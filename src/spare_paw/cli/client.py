"""HTTP client that talks to a remote spare-paw webhook API."""

from __future__ import annotations

import json
import logging
import uuid
from collections.abc import AsyncIterator
from typing import Any

import aiohttp

logger = logging.getLogger(__name__)


class RemoteClient:
    """Thin HTTP client for the spare-paw webhook API."""

    def __init__(self, url: str, secret: str = "") -> None:
        self._url = url.rstrip("/")
        self._secret = secret
        self._session_id = uuid.uuid4().hex
        self._session: aiohttp.ClientSession | None = None

    def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            headers: dict[str, str] = {
                "X-Session-Id": self._session_id,
            }
            if self._secret:
                headers["Authorization"] = f"Bearer {self._secret}"
            self._session = aiohttp.ClientSession(headers=headers)
        return self._session

    async def send_message(self, text: str, image_b64: str | None = None) -> None:
        """POST /message with user text and optional base64 image."""
        session = self._get_session()
        payload: dict[str, Any] = {"text": text}
        if image_b64:
            payload["image"] = image_b64
        async with session.post(
            f"{self._url}/message", json=payload
        ) as resp:
            if resp.status == 401:
                raise PermissionError("Authentication failed — check remote.secret")
            resp.raise_for_status()

    async def poll(self, timeout: float = 30) -> list[dict[str, Any]]:
        """GET /poll, returns list of response events."""
        session = self._get_session()
        async with session.get(
            f"{self._url}/poll",
            params={"timeout": str(timeout)},
            timeout=aiohttp.ClientTimeout(total=timeout + 5),
        ) as resp:
            if resp.status == 401:
                raise PermissionError("Authentication failed — check remote.secret")
            resp.raise_for_status()
            data = await resp.json()
            return data.get("messages", [])

    async def stream(self) -> AsyncIterator[dict[str, Any]]:
        """GET /stream, yields SSE events in real-time (persistent connection)."""
        session = self._get_session()
        async with session.get(
            f"{self._url}/stream",
            timeout=aiohttp.ClientTimeout(total=0),
        ) as resp:
            if resp.status == 401:
                raise PermissionError("Authentication failed — check remote.secret")
            resp.raise_for_status()
            async for line in resp.content:
                decoded = line.decode("utf-8").strip()
                if not decoded.startswith("data: "):
                    continue
                try:
                    yield json.loads(decoded[6:])
                except json.JSONDecodeError:
                    continue

    async def stream_response(self) -> AsyncIterator[dict[str, Any]]:
        """Stream SSE events for a single request/response cycle.

        Opens an SSE connection, yields events, and auto-closes after
        receiving a ``type: text`` event (final response).
        """
        session = self._get_session()
        async with session.get(
            f"{self._url}/stream",
            timeout=aiohttp.ClientTimeout(total=0),
        ) as resp:
            if resp.status == 401:
                raise PermissionError("Authentication failed — check remote.secret")
            resp.raise_for_status()
            async for line in resp.content:
                decoded = line.decode("utf-8").strip()
                if not decoded.startswith("data: "):
                    continue
                try:
                    event = json.loads(decoded[6:])
                except json.JSONDecodeError:
                    continue
                yield event
                if event.get("type") == "text":
                    return

    async def health(self) -> bool:
        """GET /health, returns True if server is up."""
        try:
            session = self._get_session()
            async with session.get(
                f"{self._url}/health",
                timeout=aiohttp.ClientTimeout(total=5),
            ) as resp:
                return resp.status == 200
        except (aiohttp.ClientError, OSError):
            return False

    async def status(self) -> dict[str, Any]:
        """GET /status, returns server status including model name."""
        session = self._get_session()
        async with session.get(
            f"{self._url}/status",
            timeout=aiohttp.ClientTimeout(total=5),
        ) as resp:
            if resp.status == 401:
                raise PermissionError("Authentication failed — check remote.secret")
            resp.raise_for_status()
            return await resp.json()

    async def history(self, limit: int = 10) -> list[dict[str, Any]]:
        """GET /history, returns recent conversation messages."""
        session = self._get_session()
        async with session.get(
            f"{self._url}/history",
            params={"limit": str(limit)},
            timeout=aiohttp.ClientTimeout(total=10),
        ) as resp:
            if resp.status == 401:
                raise PermissionError("Authentication failed — check remote.secret")
            resp.raise_for_status()
            data = await resp.json()
            return data.get("messages", [])

    async def close(self) -> None:
        if self._session is not None and not self._session.closed:
            await self._session.close()
            self._session = None
