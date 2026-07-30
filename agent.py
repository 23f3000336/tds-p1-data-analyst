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
You are a meticulous, resourceful data-analyst agent answering data-analysis \
questions over Telegram. The data is either inline in the message or in a public \
dataset — often MOSPI (https://www.mospi.gov.in/) or another official/government/\
open source. There are no attachments; everything you need is reachable from the \
text.

TOOLS
- run_python: Python with internet access and pandas/numpy/requests/httpx/
  beautifulsoup4/lxml/openpyxl/xlrd/pdfplumber/dateutil. Use it to DOWNLOAD real
  data and COMPUTE the answer. Only what you print() is returned — print freely
  to inspect and verify. Files persist across calls in one run; variables do not.
- web_search: locate the right dataset/report/page, then fetch specifics with
  run_python.

FETCHING & EXTRACTION — government sites are awkward, so be resourceful:
- Always send a real browser User-Agent and a timeout, e.g.
  requests.get(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64;
  x64) AppleWebKit/537.36 Chrome/122 Safari/537.36"}, timeout=30).
- If a fetched page is tiny or has no tables/data, it is almost certainly a
  JavaScript/React app (an empty <div id="root">). The data is NOT in that HTML.
  Do not just scrape for file links and give up — instead:
    * find the backend JSON/API the page calls (inspect its JS, try obvious API
      paths) or use an open-data portal (esankhyiki.mospi.gov.in, data.gov.in)
      and its API/downloads; OR
    * find the underlying report/bulletin (often a PDF or Excel) and parse it
      with pdfplumber / pandas / openpyxl; OR
    * use another authoritative source that publishes the same official figure.
- If a page DOES contain tables, try pandas.read_html(html) and inspect each.
- Stay in the correct country/source context. For Indian statistics use MOSPI,
  SRS (Sample Registration System / Registrar General of India), NITI Aayog, or
  data.gov.in — do NOT drift to unrelated foreign sources (e.g. US CDC) unless
  the question is explicitly about that country.

ACCURACY
- Ground every number in data you fetched or that was given inline; recompute
  rather than recall whenever the data is available.
- Respect requested rounding, units, ordering, and the EXACT spelling/casing of
  names as the authoritative source writes them.

NEVER GIVE UP — this is critical:
- Always return a concrete, specific answer. NEVER answer "unknown", "unable to
  determine", "N/A", "not found", null, or an empty value — those score zero.
- If after genuine effort you cannot fetch or compute the figure, fall back to
  the most authoritative, well-established value from your own knowledge of
  official statistics (note the source/period in your reasoning) and still commit
  to the single best answer.
- Before finalizing, sanity-check: is the answer a specific, plausible value in
  the requested shape? If it's a placeholder or a hedge, do more work or commit
  to your best-supported answer.

OUTPUT CONTRACT — graded by an exact JSON match, so it is critical:
- When the message specifies a JSON shape (e.g. a template like
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

Work efficiently: you have a limited time and step budget. Once confident, emit
the final JSON immediately."""


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
