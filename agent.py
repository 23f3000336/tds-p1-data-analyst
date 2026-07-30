"""The OpenAI data-analyst agent: a plain function-calling loop.

Given the running conversation (list of {role, content}) it lets the model
think, call `run_python` / `web_search` as many times as it needs, and returns
the model's final text. Orchestration (extracting JSON, filling log_url,
sending to Telegram) lives in app.py; this file is just "reason with tools".
"""
from __future__ import annotations

import json
import time
from typing import Any, Callable

from openai import OpenAI

from tools import TOOLS, execute_tool

SYSTEM_PROMPT = """\
You are a meticulous data-analyst agent. You receive data-analysis questions \
over Telegram. The data is either inline in the message or in a public dataset \
(often MOSPI — https://www.mospi.gov.in/ — or similar official/government/open \
sources). There are no file attachments; everything you need is reachable from \
the text.

TOOLS
- run_python: execute Python with internet access and pandas/numpy/requests/
  httpx/beautifulsoup4/lxml/openpyxl/xlrd/pdfplumber. Use it to DOWNLOAD the
  real data and COMPUTE the answer. Never fabricate a figure you could fetch.
  Only what you print() is returned. Print intermediate values so you can
  verify your work.
- web_search: find the correct dataset page or verify a current figure, then
  fetch specifics with run_python.

METHOD
- Read the question carefully: identify the dataset, the exact metric, the
  grouping, the year/period, and the exact output shape and units requested.
- Ground every number in data you actually fetched or that was given inline.
  Recompute rather than recall. If a source is unreachable, try another
  (mirror, cached copy, alternate official portal) before giving a best answer.
- Respect requested rounding, units, ordering, and the EXACT spelling/casing of
  names as they appear in the authoritative source.

OUTPUT CONTRACT — this is graded by an exact JSON match, so it is critical:
- When the message specifies a JSON shape (e.g. it shows a template like
  {"answer": {"state": "<state name>"}, "log_url": "<url>"} or {"values": [..]}),
  your FINAL message must be EXACTLY that JSON object, filled with your computed
  values — and NOTHING else. No prose, no explanation, no markdown, no code
  fences before or after it.
- Reproduce the requested structure faithfully (same keys, same nesting).
- For any "log_url" field, put the placeholder string "__LOG_URL__"; the system
  replaces it with the real public log URL. Do not invent a URL.
- If (and only if) the latest message is a multi-turn SETUP message that does
  NOT ask for a final answer yet (e.g. "build a model to forecast X"), reply
  with a short plain-text acknowledgement such as "Ready." — no JSON.

Work efficiently: you have a limited time and step budget. Once you are
confident, emit the final JSON immediately."""


class Agent:
    def __init__(self, config):
        self.cfg = config
        self.client = OpenAI(api_key=config.openai_api_key)

    def run(
        self,
        history: list[dict[str, str]],
        *,
        turn_directive: str,
        workdir: str,
        logger,
        deadline_ts: float | None = None,
    ) -> str:
        """Run the agent to completion and return the model's final text."""
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "system", "content": turn_directive},
            *history,
        ]

        deadline = deadline_ts or (time.time() + self.cfg.agent_deadline_seconds)
        last_text = ""

        for step in range(1, self.cfg.max_agent_steps + 1):
            # If we're (nearly) out of time, ask for the final answer with no
            # tools so we still return something usable.
            out_of_time = time.time() > deadline - 8
            tools_arg = None if out_of_time else TOOLS

            logger.log("model_request", step=step, model=self.cfg.openai_model,
                       tools_enabled=bool(tools_arg))
            try:
                resp = self._chat(messages, tools_arg)
            except Exception as e:
                logger.log("model_error", step=step, error=f"{type(e).__name__}: {e}")
                # brief backoff then retry once more within the same step budget
                time.sleep(2)
                try:
                    resp = self._chat(messages, tools_arg)
                except Exception as e2:
                    logger.log("model_error_fatal", step=step,
                               error=f"{type(e2).__name__}: {e2}")
                    break

            msg = resp.choices[0].message
            tool_calls = msg.tool_calls or []
            last_text = msg.content or last_text

            # Persist the assistant turn (must include tool_calls if present).
            assistant_entry: dict[str, Any] = {"role": "assistant",
                                               "content": msg.content or ""}
            if tool_calls:
                assistant_entry["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name,
                                     "arguments": tc.function.arguments},
                    }
                    for tc in tool_calls
                ]
            messages.append(assistant_entry)

            if not tool_calls:
                logger.log("model_message", step=step,
                           content=(msg.content or "")[:4000])
                return msg.content or ""

            # Execute every requested tool call and feed results back.
            for tc in tool_calls:
                name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except Exception:
                    args = {}
                logger.log("tool_call", step=step, tool=name,
                           arguments=_short(tc.function.arguments))
                result = execute_tool(
                    name, args,
                    workdir=workdir,
                    python_timeout=self.cfg.python_exec_timeout,
                    sandbox_mem_mb=self.cfg.sandbox_mem_mb,
                )
                logger.log("tool_result", step=step, tool=name,
                           output=_short(result, 4000))
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                })

            if out_of_time:
                break

        # Ran out of steps/time with tools still pending -> force a final answer.
        logger.log("finalizing", reason="step_or_time_budget_reached")
        messages.append({
            "role": "system",
            "content": "Stop using tools. Output ONLY the final JSON object "
                       "requested (filled with your best computed answer), or a "
                       "brief acknowledgement if no answer was requested.",
        })
        try:
            resp = self._chat(messages, None)
            return resp.choices[0].message.content or last_text
        except Exception as e:
            logger.log("finalize_error", error=f"{type(e).__name__}: {e}")
            return last_text

    def force_json(self, history: list[dict[str, str]], template_hint: str,
                   *, logger) -> str:
        """Last-ditch: the run finished without valid JSON but one was required.
        Ask the model to emit only the JSON, no tools."""
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            *history,
            {"role": "system", "content":
                "Output ONLY the JSON object the user asked for, filled with "
                "your answer. No prose, no code fences. " + template_hint},
        ]
        logger.log("force_json_request")
        resp = self._chat(messages, None)
        return resp.choices[0].message.content or ""

    # -- OpenAI call, tolerant of models that reject some params -----------
    def _chat(self, messages, tools_arg):
        kwargs: dict[str, Any] = {"model": self.cfg.openai_model, "messages": messages}
        if tools_arg:
            kwargs["tools"] = tools_arg
            kwargs["tool_choice"] = "auto"
        try:
            kwargs_t = dict(kwargs, temperature=self.cfg.openai_temperature)
            return self.client.chat.completions.create(**kwargs_t)
        except Exception as e:
            # Some reasoning models reject a custom temperature; retry default.
            if "temperature" in str(e).lower():
                return self.client.chat.completions.create(**kwargs)
            raise


def _short(text: str | None, limit: int = 1200) -> str:
    text = text or ""
    return text if len(text) <= limit else text[:limit] + f"...[+{len(text)-limit} chars]"
