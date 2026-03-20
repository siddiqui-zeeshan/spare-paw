"""Tests for chat_stream() SSE streaming in OpenRouterClient."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from spare_paw.router.openrouter import OpenRouterClient, OpenRouterError


def _make_sse_lines(chunks: list[str]) -> list[bytes]:
    """Build raw SSE byte lines from a list of text deltas."""
    lines = []
    for i, text in enumerate(chunks):
        import json
        data = {
            "choices": [{"delta": {"content": text}}]
        }
        lines.append(f"data: {json.dumps(data)}\n".encode())
    lines.append(b"data: [DONE]\n")
    return lines


class _FakeContent:
    """Async iterable over byte lines."""

    def __init__(self, lines: list[bytes]) -> None:
        self._lines = lines

    def __aiter__(self):
        return self

    async def __anext__(self) -> bytes:
        if not self._lines:
            raise StopAsyncIteration
        return self._lines.pop(0)


class TestChatStream:
    @pytest.mark.asyncio
    async def test_yields_text_deltas(self):
        """chat_stream() should yield each text delta from the SSE response."""
        sse_lines = _make_sse_lines(["Hello", " ", "world"])

        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.content = _FakeContent(sse_lines)

        mock_ctx = AsyncMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_ctx.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.closed = False
        mock_session.post = MagicMock(return_value=mock_ctx)

        client = OpenRouterClient(api_key="test", semaphore=asyncio.Semaphore(1))
        client._session = mock_session

        deltas = []
        async for delta in client.chat_stream(
            messages=[{"role": "user", "content": "hi"}],
            model="test/model",
        ):
            deltas.append(delta)

        assert deltas == ["Hello", " ", "world"]

    @pytest.mark.asyncio
    async def test_raises_on_error_status(self):
        """chat_stream() should raise OpenRouterError on HTTP errors."""
        mock_resp = AsyncMock()
        mock_resp.status = 401
        mock_resp.text = AsyncMock(return_value="unauthorized")

        mock_ctx = AsyncMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_ctx.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.closed = False
        mock_session.post = MagicMock(return_value=mock_ctx)

        client = OpenRouterClient(api_key="bad", semaphore=asyncio.Semaphore(1))
        client._session = mock_session

        with pytest.raises(OpenRouterError):
            async for _ in client.chat_stream(messages=[], model="m"):
                pass

    @pytest.mark.asyncio
    async def test_sends_stream_true_in_body(self):
        """chat_stream() should include stream=True in the request body."""
        sse_lines = _make_sse_lines(["ok"])

        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.content = _FakeContent(sse_lines)

        mock_ctx = AsyncMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_ctx.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.closed = False
        mock_session.post = MagicMock(return_value=mock_ctx)

        client = OpenRouterClient(api_key="k", semaphore=asyncio.Semaphore(1))
        client._session = mock_session

        async for _ in client.chat_stream(messages=[], model="m"):
            pass

        body = mock_session.post.call_args[1]["json"]
        assert body["stream"] is True

    @pytest.mark.asyncio
    async def test_handles_malformed_sse_lines(self):
        """chat_stream() should skip malformed SSE lines gracefully."""
        import json
        lines = [
            b":\n",  # comment line
            b"\n",  # empty line
            b"data: not-json\n",  # malformed JSON
            f'data: {json.dumps({"choices": [{"delta": {"content": "ok"}}]})}\n'.encode(),
            b"data: [DONE]\n",
        ]

        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.content = _FakeContent(lines)

        mock_ctx = AsyncMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_ctx.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.closed = False
        mock_session.post = MagicMock(return_value=mock_ctx)

        client = OpenRouterClient(api_key="k", semaphore=asyncio.Semaphore(1))
        client._session = mock_session

        deltas = []
        async for delta in client.chat_stream(messages=[], model="m"):
            deltas.append(delta)

        assert deltas == ["ok"]
