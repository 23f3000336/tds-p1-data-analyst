"""Central configuration, read once from the environment.

Every knob has an env var so the same image runs unchanged in local dev and in
production — you only ever edit `.env` (local) or the platform's dashboard.
"""
from __future__ import annotations

import os
from dataclasses import dataclass


def _get(name: str, default: str | None = None, required: bool = False) -> str:
    val = os.environ.get(name, default)
    if required and not val:
        raise RuntimeError(
            f"Missing required environment variable {name!r}. "
            f"Copy .env.example to .env and fill it in."
        )
    return val or ""


@dataclass(frozen=True)
class Config:
    # --- OpenAI ---------------------------------------------------------
    openai_api_key: str
    openai_model: str
    openai_temperature: float

    # --- Telegram -------------------------------------------------------
    telegram_bot_token: str
    bot_mode: str            # "polling" (default) or "webhook"
    webhook_secret: str      # path segment guarding the webhook route

    # --- Hosting / logs -------------------------------------------------
    public_base_url: str     # e.g. https://mybot.fly.dev  (no trailing slash)
    log_dir: str
    port: int

    # --- Agent behaviour -----------------------------------------------
    max_agent_steps: int
    agent_deadline_seconds: float   # wall-clock budget for one answer
    python_exec_timeout: int        # per run_python call
    sandbox_mem_mb: int
    history_idle_reset_seconds: float
    max_history_turns: int

    @staticmethod
    def load() -> "Config":
        base = _get("PUBLIC_BASE_URL", "").rstrip("/")
        port = int(_get("PORT", "8080"))
        if not base:
            # Fall back to localhost so the process still boots; log_url will be
            # wrong until PUBLIC_BASE_URL is set. We warn loudly at startup.
            base = f"http://localhost:{port}"
        return Config(
            openai_api_key=_get("OPENAI_API_KEY", required=True),
            openai_model=_get("OPENAI_MODEL", "gpt-4o"),
            openai_temperature=float(_get("OPENAI_TEMPERATURE", "0")),
            telegram_bot_token=_get("TELEGRAM_BOT_TOKEN", required=True),
            bot_mode=_get("BOT_MODE", "polling").strip().lower(),
            webhook_secret=_get("WEBHOOK_SECRET", "hook"),
            public_base_url=base,
            log_dir=_get("LOG_DIR", "logs"),
            port=port,
            max_agent_steps=int(_get("MAX_AGENT_STEPS", "12")),
            agent_deadline_seconds=float(_get("AGENT_DEADLINE_SECONDS", "240")),
            python_exec_timeout=int(_get("PYTHON_EXEC_TIMEOUT", "70")),
            sandbox_mem_mb=int(_get("SANDBOX_MEM_MB", "1536")),
            history_idle_reset_seconds=float(_get("HISTORY_IDLE_RESET_SECONDS", "900")),
            max_history_turns=int(_get("MAX_HISTORY_TURNS", "10")),
        )
