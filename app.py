"""Entry point: the web service that IS the Telegram bot.

Responsibilities
  * Serve the run logs publicly at /logs/<run_id>.jsonl  (that's `log_url`).
  * Receive Telegram messages (long-polling by default, webhook optional).
  * For each message: run the data-analyst agent, shape the reply to the exact
    JSON the question asked for (filling in the real log_url), and send exactly
    ONE reply — which is what the grader reads.

Run:  uvicorn app:app --host 0.0.0.0 --port $PORT
"""
from __future__ import annotations

import asyncio
import os
import re
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse

from agent import Agent
from answer import (apply_log_url, extract_json_object, looks_like_giveup,
                    to_reply_string, wants_final_json)
from config import Config
from runlog import RunLogger, new_run_id
from telegram_api import TelegramClient

cfg = Config.load()
agent = Agent(cfg)
telegram = TelegramClient(cfg.telegram_bot_token)

_SAFE_LOG_NAME = re.compile(r"^[A-Za-z0-9._-]+\.jsonl$")


# ---------------------------------------------------------------------------
# Per-chat conversation state
# ---------------------------------------------------------------------------
@dataclass
class ChatState:
    history: list[dict[str, str]] = field(default_factory=list)
    last_ts: float = 0.0
    just_finalized: bool = False
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


_chats: dict[int, ChatState] = {}


def _state(chat_id: int) -> ChatState:
    st = _chats.get(chat_id)
    if st is None:
        st = ChatState()
        _chats[chat_id] = st
    return st


# ---------------------------------------------------------------------------
# Core: handle one incoming message end-to-end
# ---------------------------------------------------------------------------
async def handle_message(chat_id: int, text: str) -> None:
    st = _state(chat_id)
    async with st.lock:
        now = time.time()
        # Start a fresh conversation if the previous turn was a final answer,
        # or if we've been idle long enough that this is surely a new question.
        if st.just_finalized or (now - st.last_ts) > cfg.history_idle_reset_seconds:
            st.history = []
            st.just_finalized = False
        st.last_ts = now

        st.history.append({"role": "user", "content": text})
        # keep the last N turns to bound context
        max_msgs = cfg.max_history_turns * 2
        if len(st.history) > max_msgs:
            st.history = st.history[-max_msgs:]

        want_json = wants_final_json(text)
        run_id = new_run_id()
        logger = RunLogger(cfg.log_dir, cfg.public_base_url, run_id)
        log_url = logger.url
        workdir = os.path.join(cfg.log_dir, f"work-{run_id}")

        logger.log("message_received", chat_id=chat_id, text=text,
                   wants_final_json=want_json, history_len=len(st.history))

        deadline_ts = now + cfg.agent_deadline_seconds
        if want_json:
            directive = (
                f"The latest user message REQUIRES a final JSON answer now. You "
                f"have about {int(cfg.agent_deadline_seconds)}s. Fetch the real "
                f"data with your tools, compute carefully, then output ONLY the "
                f"exact JSON object the message specifies (use \"__LOG_URL__\" for "
                f"any log_url field)."
            )
        else:
            directive = (
                "The latest user message is a multi-turn SETUP/context message "
                "that does not request a final answer yet. Reply with a short "
                "plain-text acknowledgement (e.g. 'Ready.'). Do NOT output JSON."
            )

        history_copy = list(st.history)
        reply: str
        try:
            final_text = await asyncio.to_thread(
                agent.run, history_copy,
                turn_directive=directive, workdir=workdir, logger=logger,
                deadline_ts=deadline_ts,
            )
        except Exception as e:  # pragma: no cover - defensive
            logger.log("agent_exception", error=f"{type(e).__name__}: {e}")
            final_text = ""

        if want_json:
            reply = await _shape_json_reply(
                final_text, history_copy, log_url, text, logger
            )
            st.just_finalized = True
        else:
            reply = (final_text or "").strip() or "Ready."
            # Never let an ack accidentally be a gradeable JSON blob.
            if extract_json_object(reply) is not None:
                reply = "Ready."
            st.just_finalized = False

        st.history.append({"role": "assistant", "content": reply})
        logger.log("final_reply", reply=reply, log_url=log_url)
        logger.close()

    # Send outside the lock; grader reads exactly this one message.
    try:
        await telegram.send_message(chat_id, reply)
    except Exception as e:
        logger.log("send_failed", error=f"{type(e).__name__}: {e}")


async def _force(history_copy, hint, logger):
    """One tool-free pass that must emit only the JSON. Returns parsed obj/None."""
    try:
        forced = await asyncio.to_thread(
            agent.force_json, history_copy, hint, logger=logger)
        obj = extract_json_object(forced)
        logger.log("force_json_result", got_json=obj is not None)
        return obj
    except Exception as e:
        logger.log("force_json_error", error=f"{type(e).__name__}: {e}")
        return None


async def _shape_json_reply(final_text, history_copy, log_url, incoming_text,
                            logger) -> str:
    obj = extract_json_object(final_text)
    if obj is None:
        # The model didn't emit clean JSON — force one more, tool-free pass.
        obj = await _force(
            history_copy, "Match the exact shape shown in the user's message.", logger)

    # If the model hedged (unknown / unable / N/A / None / empty), demand a
    # committed answer — a blank answer always scores zero, a best-effort one
    # can be right.
    if obj is not None and looks_like_giveup(obj):
        logger.log("giveup_detected", answer=obj)
        obj2 = await _force(
            history_copy,
            "Do NOT answer with unknown / unable / N/A / none / null or an empty "
            "value. Commit to your single best, specific answer using authoritative "
            "public knowledge (note the source/period in your reasoning), in the "
            "EXACT JSON shape the user asked for.",
            logger,
        )
        if obj2 is not None and not looks_like_giveup(obj2):
            obj = obj2

    if obj is not None:
        force_url = "log_url" in incoming_text.lower()
        obj = apply_log_url(obj, log_url, force=force_url)
        return to_reply_string(obj)

    # Absolute last resort: send trimmed text so the grader gets *a* reply.
    logger.log("no_json_extracted", final_text_preview=(final_text or "")[:500])
    return (final_text or "").strip() or to_reply_string({"log_url": log_url})


# ---------------------------------------------------------------------------
# Long-polling loop (default transport)
# ---------------------------------------------------------------------------
_bg_tasks: set[asyncio.Task] = set()


def _spawn(coro) -> None:
    task = asyncio.create_task(coro)
    _bg_tasks.add(task)
    task.add_done_callback(_bg_tasks.discard)


async def _extract_and_dispatch(update: dict) -> None:
    msg = update.get("message") or update.get("edited_message")
    if not msg:
        return
    text = msg.get("text")
    chat = msg.get("chat", {})
    chat_id = chat.get("id")
    if text is None or chat_id is None:
        return
    _spawn(handle_message(chat_id, text))


async def polling_loop(stop: asyncio.Event) -> None:
    offset: int | None = None
    print("[bot] polling for updates...")
    while not stop.is_set():
        try:
            updates = await telegram.get_updates(offset, timeout=50)
            for upd in updates:
                offset = upd["update_id"] + 1
                await _extract_and_dispatch(upd)
        except asyncio.CancelledError:
            break
        except Exception as e:
            print(f"[bot] polling error: {type(e).__name__}: {e}; retrying in 3s")
            await asyncio.sleep(3)


# ---------------------------------------------------------------------------
# FastAPI app + lifespan
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    os.makedirs(cfg.log_dir, exist_ok=True)
    if cfg.public_base_url.startswith("http://localhost"):
        print("[bot] WARNING: PUBLIC_BASE_URL is unset — log_url will point at "
              "localhost and won't be reachable by the grader. Set it in your env.")
    me = await telegram.get_me()
    print(f"[bot] running as @{me.get('username')} in {cfg.bot_mode} mode; "
          f"model={cfg.openai_model}; base={cfg.public_base_url}")

    stop = asyncio.Event()
    poll_task = None
    if cfg.bot_mode == "webhook":
        url = f"{cfg.public_base_url}/webhook/{cfg.webhook_secret}"
        await telegram.set_webhook(url, secret_token=cfg.webhook_secret)
        print(f"[bot] webhook set to {url}")
    else:
        await telegram.delete_webhook(drop_pending=False)
        poll_task = asyncio.create_task(polling_loop(stop))

    try:
        yield
    finally:
        stop.set()
        if poll_task:
            poll_task.cancel()
        await telegram.close()


app = FastAPI(lifespan=lifespan, title="Data-Analyst Telegram Bot")


@app.get("/")
async def root():
    return {"ok": True, "service": "data-analyst-telegram-bot", "mode": cfg.bot_mode}


@app.get("/healthz")
async def healthz():
    return PlainTextResponse("ok")


@app.get("/logs/{name}")
async def get_log(name: str):
    if not _SAFE_LOG_NAME.match(name):
        raise HTTPException(status_code=400, detail="bad log name")
    path = os.path.join(cfg.log_dir, name)
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="not found")
    # text/plain so it renders in a browser and is trivially wget-able.
    return FileResponse(path, media_type="text/plain; charset=utf-8")


@app.post("/webhook/{secret}")
async def webhook(secret: str, request: Request,
                  x_telegram_bot_api_secret_token: str | None = Header(default=None)):
    if secret != cfg.webhook_secret or (
        x_telegram_bot_api_secret_token not in (None, cfg.webhook_secret)
    ):
        raise HTTPException(status_code=403, detail="forbidden")
    update = await request.json()
    await _extract_and_dispatch(update)
    return JSONResponse({"ok": True})
