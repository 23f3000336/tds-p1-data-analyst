"""Per-run JSONL logger.

The grader wants `log_url` to point at "your agent's run log as JSONL, one JSON
object per line". We give every answer its own file `logs/<run_id>.jsonl`,
served publicly by the app at `<PUBLIC_BASE_URL>/logs/<run_id>.jsonl`.

The URL is known *before* the run starts (it's derived from run_id), so we can
put it straight into the reply while the run is still being written. We flush on
every line so a crash still leaves a readable, wget-able log.
"""
from __future__ import annotations

import json
import os
import time
import uuid
from datetime import datetime, timezone


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def new_run_id() -> str:
    # time-sortable + random suffix, filesystem/URL safe
    return f"{int(time.time())}-{uuid.uuid4().hex[:8]}"


class RunLogger:
    def __init__(self, log_dir: str, public_base_url: str, run_id: str | None = None):
        self.run_id = run_id or new_run_id()
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.path = os.path.join(log_dir, f"{self.run_id}.jsonl")
        self.url = f"{public_base_url.rstrip('/')}/logs/{self.run_id}.jsonl"
        self._fh = open(self.path, "a", encoding="utf-8")

    def log(self, event: str, **fields) -> None:
        record = {"ts": _now_iso(), "event": event, "run_id": self.run_id}
        record.update(fields)
        self._fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        self._fh.flush()

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:
            pass
