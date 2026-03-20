"""OpenRouter API client with retry and exponential backoff."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from typing import Any

import aiohttp

logger = logging.getLogger(__name__)

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# Retry configuration
_RETRYABLE_STATUSES = frozenset({429, 500, 502, 503, 504})
_NON_RETRYABLE_STATUSES = frozenset({400, 401, 403})
_MAX_RETRIES = 3
_BASE_DELAY = 1.0  # seconds
_MAX_DELAY = 30.0  # seconds


class OpenRouterError(Exception):
    """Raised on non-retryable OpenRouter API failures."""

    def __init__(self, status: int, message: str) -> None:
        self.status = status
        self.message = message
        super().__init__(f"OpenRouter API error {status}: {message}")


class OpenRouterClient:
    """Async client for the OpenRouter chat completions API.

    All calls are gated by a shared semaphore to serialize concurrent
    model requests (user messages and cron jobs).
    """

    def __init__(self, api_key: str, semaphore: asyncio.Semaphore) -> None:
        self._api_key = api_key
        self._semaphore = semaphore
        self._session: aiohttp.ClientSession | None = None

    def _get_session(self) -> aiohttp.ClientSession:
        """Lazily create and return the aiohttp session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "HTTP-Referer": "https://github.com/spare-paw/spare-paw",
                    "X-Title": "spare-paw",
                    "Content-Type": "application/json",
                },
            )
        return self._session

    async def chat(
        self,
        messages: list[dict[str, Any]],
        model: str,
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Send a chat completion request to OpenRouter.

        Args:
            messages: OpenAI-format message list.
            model: Model identifier (e.g. "google/gemini-2.0-flash").
            tools: Optional list of tool JSON schemas for function calling.

        Returns:
            The full completion response JSON.

        Raises:
            OpenRouterError: On non-retryable HTTP errors (400, 401, 403).
            aiohttp.ClientError: On connection-level failures after retries.
        """
        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
        }
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"

        async with self._semaphore:
            return await self._request_with_retry(body)

    async def _request_with_retry(self, body: dict[str, Any]) -> dict[str, Any]:
        """Execute the HTTP request with exponential backoff on transient errors."""
        session = self._get_session()
        last_exception: Exception | None = None

        for attempt in range(_MAX_RETRIES + 1):
            try:
                async with session.post(OPENROUTER_URL, json=body) as resp:
                    # Non-retryable client errors — fail immediately
                    if resp.status in _NON_RETRYABLE_STATUSES:
                        text = await resp.text()
                        raise OpenRouterError(resp.status, text)

                    # Retryable server / rate-limit errors
                    if resp.status in _RETRYABLE_STATUSES:
                        text = await resp.text()
                        last_exception = OpenRouterError(resp.status, text)
                        if attempt < _MAX_RETRIES:
                            delay = min(
                                _BASE_DELAY * (2 ** attempt),
                                _MAX_DELAY,
                            )
                            logger.warning(
                                "OpenRouter %d on attempt %d/%d, retrying in %.1fs",
                                resp.status,
                                attempt + 1,
                                _MAX_RETRIES + 1,
                                delay,
                            )
                            await asyncio.sleep(delay)
                            continue
                        # Exhausted retries
                        raise last_exception

                    # Unexpected non-2xx status
                    if resp.status >= 300:
                        text = await resp.text()
                        raise OpenRouterError(resp.status, text)

                    # Success
                    data: dict[str, Any] = await resp.json()
                    return data

            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                last_exception = exc
                if attempt < _MAX_RETRIES:
                    delay = min(_BASE_DELAY * (2 ** attempt), _MAX_DELAY)
                    logger.warning(
                        "OpenRouter connection error on attempt %d/%d (%s), "
                        "retrying in %.1fs",
                        attempt + 1,
                        _MAX_RETRIES + 1,
                        exc,
                        delay,
                    )
                    await asyncio.sleep(delay)
                    continue

        # Should only reach here on connection-level failures after all retries
        assert last_exception is not None
        raise last_exception

    async def chat_stream(
        self,
        messages: list[dict[str, Any]],
        model: str,
        tools: list[dict[str, Any]] | None = None,
    ) -> AsyncIterator[str]:
        """Yield text deltas via SSE streaming."""
        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": True,
        }
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"

        session = self._get_session()
        async with self._semaphore:
            async with session.post(OPENROUTER_URL, json=body) as resp:
                if resp.status >= 400:
                    text = await resp.text()
                    raise OpenRouterError(resp.status, text)

                async for line in resp.content:
                    decoded = line.decode("utf-8").strip()
                    if not decoded.startswith("data: "):
                        continue
                    payload = decoded[6:]
                    if payload == "[DONE]":
                        break
                    try:
                        chunk = json.loads(payload)
                        delta = chunk["choices"][0].get("delta", {})
                        content = delta.get("content")
                        if content:
                            yield content
                    except (json.JSONDecodeError, KeyError, IndexError):
                        continue

    async def close(self) -> None:
        """Close the underlying aiohttp session."""
        if self._session is not None and not self._session.closed:
            await self._session.close()
            self._session = None
