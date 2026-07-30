"""Minimal async Telegram Bot API client (httpx). Just what this bot needs:
long-poll for updates, send a text reply, and manage the webhook. Avoiding the
python-telegram-bot dependency keeps the surface tiny and version-stable.
"""
from __future__ import annotations

from typing import Any

import httpx


class TelegramClient:
    def __init__(self, token: str):
        self._base = f"https://api.telegram.org/bot{token}"
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(65.0))

    async def close(self) -> None:
        await self._client.aclose()

    async def _call(self, method: str, **params) -> dict[str, Any]:
        r = await self._client.post(f"{self._base}/{method}", json=params)
        r.raise_for_status()
        data = r.json()
        if not data.get("ok"):
            raise RuntimeError(f"Telegram {method} failed: {data}")
        return data.get("result")

    async def get_me(self) -> dict[str, Any]:
        return await self._call("getMe")

    async def get_updates(self, offset: int | None, timeout: int = 50) -> list[dict]:
        # long-poll; keep client timeout > server timeout (set to 65 above)
        params: dict[str, Any] = {"timeout": timeout, "allowed_updates": ["message"]}
        if offset is not None:
            params["offset"] = offset
        return await self._call("getUpdates", **params)

    async def send_message(self, chat_id: int, text: str) -> dict[str, Any]:
        # No parse_mode: the reply is raw JSON and must not be interpreted as
        # Markdown/HTML. Telegram's hard limit is 4096 chars per message.
        return await self._call(
            "sendMessage", chat_id=chat_id, text=text[:4096],
            disable_web_page_preview=True,
        )

    async def set_webhook(self, url: str, secret_token: str | None = None) -> Any:
        params: dict[str, Any] = {"url": url, "allowed_updates": ["message"],
                                  "drop_pending_updates": False}
        if secret_token:
            params["secret_token"] = secret_token
        return await self._call("setWebhook", **params)

    async def delete_webhook(self, drop_pending: bool = False) -> Any:
        return await self._call("deleteWebhook", drop_pending_updates=drop_pending)
