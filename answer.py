"""Turning the model's final text into the exact reply the grader expects.

The grader does `json.loads(replies[-1].strip())` and compares to an expected
value, so the final Telegram message must be **one JSON object and nothing
else**. These helpers:

  * decide whether a given incoming message actually demands a final JSON answer
    (vs. being a multi-turn setup message that just needs an acknowledgement),
  * robustly pull the JSON object out of the model's final text (tolerating code
    fences / stray prose the model might add), and
  * force the correct `log_url` in whenever the question asked for one.
"""
from __future__ import annotations

import json
import re
from typing import Any

_JSON_HINTS = (
    "log_url",
    "reply with only",
    "reply with just",
    "respond with only",
    "json object",
    "as a json",
    "in json",
    "single json",
)


def wants_final_json(text: str) -> bool:
    """Does this message ask for a final JSON answer now?

    True for real questions (which spell out a JSON shape); False for pure
    multi-turn setup messages like "build a model to forecast X".
    """
    t = text.lower()
    if any(h in t for h in _JSON_HINTS):
        return True
    # A JSON template embedded in the message, e.g. {"state": "<...>"}
    if re.search(r'\{\s*"', text):
        return True
    return False


def _strip_fences(text: str) -> str:
    text = text.strip()
    # ```json ... ```  or  ``` ... ```
    fence = re.match(r"^```[a-zA-Z0-9_]*\s*(.*?)\s*```$", text, re.S)
    if fence:
        return fence.group(1).strip()
    return text


def _balanced_json_span(text: str) -> str | None:
    """Return the first top-level {...} span, respecting strings/escapes."""
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        c = text[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
        else:
            if c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
    return None


def extract_json_object(text: str) -> Any | None:
    """Best-effort parse of a single JSON object from arbitrary model text."""
    if not text:
        return None
    candidate = _strip_fences(text)
    for attempt in (candidate, text):
        try:
            obj = json.loads(attempt.strip())
            if isinstance(obj, (dict, list)):
                return obj
        except Exception:
            pass
        span = _balanced_json_span(attempt)
        if span:
            try:
                return json.loads(span)
            except Exception:
                continue
    return None


def apply_log_url(obj: Any, log_url: str, *, force: bool) -> Any:
    """Ensure the reply carries the real log_url.

    If the object already has a top-level `log_url` key, overwrite it (the model
    only ever sees a placeholder). If it doesn't but the question demanded one
    (`force`), add it. Nested occurrences are left to the model's structure.
    """
    if isinstance(obj, dict):
        if "log_url" in obj or force:
            obj["log_url"] = log_url
    return obj


def to_reply_string(obj: Any) -> str:
    """Compact, single-object JSON — exactly what the grader will json.loads()."""
    return json.dumps(obj, ensure_ascii=False)
