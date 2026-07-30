"""Offline smoke test — exercise the agent + answer-shaping WITHOUT Telegram.

    OPENAI_API_KEY=sk-... python run_local.py "Your question here ... Reply with ONLY {\"x\": <n>}"

Or with no argument it runs a couple of built-in sample questions. It prints
exactly the JSON string the bot would send, and the path of the run log.
"""
from __future__ import annotations

import os
import sys

# Telegram token isn't needed for the offline path, but Config requires it.
os.environ.setdefault("TELEGRAM_BOT_TOKEN", "offline-test")
os.environ.setdefault("PUBLIC_BASE_URL", "https://example.invalid")

from agent import Agent                     # noqa: E402
from answer import (apply_log_url, extract_json_object, to_reply_string,       # noqa: E402
                    wants_final_json)
from config import Config                   # noqa: E402
from runlog import RunLogger, new_run_id    # noqa: E402

SAMPLES = [
    'Compute the mean of these numbers: [4, 8, 15, 16, 23, 42]. '
    'Reply with ONLY a JSON object like {"mean": <number rounded to 2 decimals>}',
    'From the inline table name,score\\nAda,90\\nGrace,85\\nLinus,95 , who has the '
    'highest score? Reply with ONLY {"answer": {"name": "<name>"}, "log_url": "<url>"}',
]


def answer_one(cfg: Config, agent: Agent, question: str) -> str:
    run_id = new_run_id()
    logger = RunLogger(cfg.log_dir, cfg.public_base_url, run_id)
    workdir = os.path.join(cfg.log_dir, f"work-{run_id}")
    history = [{"role": "user", "content": question}]
    want = wants_final_json(question)
    directive = ("A final JSON answer is required now; output ONLY the requested "
                 "JSON (use \"__LOG_URL__\" for any log_url)." if want else
                 "Setup message; acknowledge briefly, no JSON.")
    text = agent.run(history, turn_directive=directive, workdir=workdir,
                     logger=logger, deadline_ts=None)
    if want:
        obj = extract_json_object(text)
        if obj is not None:
            obj = apply_log_url(obj, logger.url, force="log_url" in question.lower())
            reply = to_reply_string(obj)
        else:
            reply = text.strip()
    else:
        reply = text.strip()
    logger.log("final_reply", reply=reply)
    logger.close()
    print(f"\nQ: {question}\nREPLY: {reply}\nLOG:  {logger.path}\n" + "-" * 70)
    return reply


def main() -> None:
    cfg = Config.load()
    agent = Agent(cfg)
    questions = [sys.argv[1]] if len(sys.argv) > 1 else SAMPLES
    for q in questions:
        answer_one(cfg, agent, q)


if __name__ == "__main__":
    main()
