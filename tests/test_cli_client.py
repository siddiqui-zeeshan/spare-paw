"""Tests for the RemoteClient HTTP client."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from spare_paw.cli.client import RemoteClient


def _make_response(status: int, json_data: dict | None = None):
    """Create a mock aiohttp response context manager."""
    resp = AsyncMock()
    resp.status = status
    if json_data is not None:
        resp.json = AsyncMock(return_value=json_data)
    resp.raise_for_status = MagicMock()
    if status >= 400:
        resp.raise_for_status = MagicMock(side_effect=Exception(f"HTTP {status}"))

    ctx = AsyncMock()
    ctx.__aenter__ = AsyncMock(return_value=resp)
    ctx.__aexit__ = AsyncMock(return_value=False)
    return ctx


class TestRemoteClient:
    @pytest.mark.asyncio
    async def test_health_returns_true_on_200(self):
        client = RemoteClient(url="http://localhost:8080")
        mock_session = MagicMock()
        mock_session.closed = False
        mock_session.get = MagicMock(
            return_value=_make_response(200, {"status": "ok"})
        )
        client._session = mock_session

        result = await client.health()
        assert result is True

    @pytest.mark.asyncio
    async def test_health_returns_false_on_error(self):
        client = RemoteClient(url="http://localhost:8080")
        mock_session = MagicMock()
        mock_session.closed = False
        mock_session.get = MagicMock(side_effect=OSError("connection refused"))
        client._session = mock_session

        result = await client.health()
        assert result is False

    @pytest.mark.asyncio
    async def test_send_message_posts_text(self):
        client = RemoteClient(url="http://localhost:8080")
        mock_session = MagicMock()
        mock_session.closed = False
        mock_session.post = MagicMock(
            return_value=_make_response(200, {"status": "accepted"})
        )
        client._session = mock_session

        await client.send_message("hello")

        mock_session.post.assert_called_once()
        call_args = mock_session.post.call_args
        assert "/message" in call_args[0][0]
        assert call_args[1]["json"] == {"text": "hello"}

    @pytest.mark.asyncio
    async def test_send_message_raises_on_401(self):
        client = RemoteClient(url="http://localhost:8080")
        mock_session = MagicMock()
        mock_session.closed = False

        resp = AsyncMock()
        resp.status = 401
        ctx = AsyncMock()
        ctx.__aenter__ = AsyncMock(return_value=resp)
        ctx.__aexit__ = AsyncMock(return_value=False)
        mock_session.post = MagicMock(return_value=ctx)
        client._session = mock_session

        with pytest.raises(PermissionError, match="Authentication failed"):
            await client.send_message("hello")

    @pytest.mark.asyncio
    async def test_poll_returns_messages(self):
        client = RemoteClient(url="http://localhost:8080")
        mock_session = MagicMock()
        mock_session.closed = False
        mock_session.get = MagicMock(
            return_value=_make_response(
                200,
                {"messages": [{"type": "text", "text": "hi"}]},
            )
        )
        client._session = mock_session

        messages = await client.poll(timeout=5)
        assert len(messages) == 1
        assert messages[0]["type"] == "text"
        assert messages[0]["text"] == "hi"

    @pytest.mark.asyncio
    async def test_auth_header_set_when_secret_provided(self):
        client = RemoteClient(url="http://localhost:8080", secret="mysecret")
        session = client._get_session()
        assert session._default_headers["Authorization"] == "Bearer mysecret"
        await session.close()

    @pytest.mark.asyncio
    async def test_close_cleans_up_session(self):
        client = RemoteClient(url="http://localhost:8080")
        client._get_session()
        assert client._session is not None
        await client.close()
        assert client._session is None


class TestRemoteClientSessionId:
    def test_session_id_is_hex_uuid(self):
        client = RemoteClient(url="http://localhost:8080")
        assert isinstance(client._session_id, str)
        assert len(client._session_id) == 32
        # Should be all hex chars
        int(client._session_id, 16)

    def test_two_clients_have_different_session_ids(self):
        c1 = RemoteClient(url="http://localhost:8080")
        c2 = RemoteClient(url="http://localhost:8080")
        assert c1._session_id != c2._session_id

    @pytest.mark.asyncio
    async def test_session_header_contains_session_id(self):
        client = RemoteClient(url="http://localhost:8080")
        session = client._get_session()
        assert session._default_headers["X-Session-Id"] == client._session_id
        await session.close()

    @pytest.mark.asyncio
    async def test_session_id_header_sent_without_secret(self):
        client = RemoteClient(url="http://localhost:8080")
        session = client._get_session()
        assert "X-Session-Id" in session._default_headers
        assert "Authorization" not in session._default_headers
        await session.close()


class TestRemoteClientStatus:
    @pytest.mark.asyncio
    async def test_status_returns_json(self):
        client = RemoteClient(url="http://localhost:8080", secret="s")
        mock_session = MagicMock()
        mock_session.closed = False
        mock_session.get = MagicMock(
            return_value=_make_response(
                200,
                {"status": "ok", "model": "claude-3-haiku", "tools": 5, "uptime": 42},
            )
        )
        client._session = mock_session

        data = await client.status()
        assert data["model"] == "claude-3-haiku"
        assert data["tools"] == 5
        assert data["uptime"] == 42

    @pytest.mark.asyncio
    async def test_status_raises_on_401(self):
        client = RemoteClient(url="http://localhost:8080", secret="bad")
        mock_session = MagicMock()
        mock_session.closed = False

        resp = AsyncMock()
        resp.status = 401
        ctx = AsyncMock()
        ctx.__aenter__ = AsyncMock(return_value=resp)
        ctx.__aexit__ = AsyncMock(return_value=False)
        mock_session.get = MagicMock(return_value=ctx)
        client._session = mock_session

        with pytest.raises(PermissionError, match="Authentication failed"):
            await client.status()

    @pytest.mark.asyncio
    async def test_status_calls_status_endpoint(self):
        client = RemoteClient(url="http://localhost:8080")
        mock_session = MagicMock()
        mock_session.closed = False
        mock_session.get = MagicMock(
            return_value=_make_response(200, {"status": "ok", "model": "m"})
        )
        client._session = mock_session

        await client.status()

        call_url = mock_session.get.call_args[0][0]
        assert call_url.endswith("/status")


def _make_async_iter(lines: list[bytes]):
    """Return an async iterable over the given byte lines."""

    async def _gen():
        for line in lines:
            yield line

    return _gen()


class TestStreamResponse:
    @pytest.mark.asyncio
    async def test_stream_response_yields_events_and_stops_on_text(self):
        client = RemoteClient(url="http://localhost:8080")

        lines = [
            b'data: {"type": "token", "token": "hi"}\n',
            b"\n",
            b'data: {"type": "text", "text": "hi there"}\n',
            b"\n",
            b'data: {"type": "token", "token": "extra"}\n',
        ]

        resp = AsyncMock()
        resp.status = 200
        resp.raise_for_status = MagicMock()
        resp.content = _make_async_iter(lines)

        ctx = AsyncMock()
        ctx.__aenter__ = AsyncMock(return_value=resp)
        ctx.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.closed = False
        mock_session.get = MagicMock(return_value=ctx)
        client._session = mock_session

        events = []
        async for event in client.stream_response():
            events.append(event)

        # Should receive token and text, but stop after text
        types = [e["type"] for e in events]
        assert "token" in types
        assert "text" in types
        # Nothing after the text event
        extra = [e for e in events if e.get("token") == "extra"]
        assert extra == []

    @pytest.mark.asyncio
    async def test_stream_response_raises_on_401(self):
        client = RemoteClient(url="http://localhost:8080")
        mock_session = MagicMock()
        mock_session.closed = False

        resp = AsyncMock()
        resp.status = 401
        ctx = AsyncMock()
        ctx.__aenter__ = AsyncMock(return_value=resp)
        ctx.__aexit__ = AsyncMock(return_value=False)
        mock_session.get = MagicMock(return_value=ctx)
        client._session = mock_session

        with pytest.raises(PermissionError, match="Authentication failed"):
            async for _ in client.stream_response():
                pass

    @pytest.mark.asyncio
    async def test_stream_response_skips_non_data_lines(self):
        client = RemoteClient(url="http://localhost:8080")

        lines = [
            b": keep-alive\n",
            b"\n",
            b'data: {"type": "text", "text": "done"}\n',
        ]

        resp = AsyncMock()
        resp.status = 200
        resp.raise_for_status = MagicMock()
        resp.content = _make_async_iter(lines)

        ctx = AsyncMock()
        ctx.__aenter__ = AsyncMock(return_value=resp)
        ctx.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.closed = False
        mock_session.get = MagicMock(return_value=ctx)
        client._session = mock_session

        events = []
        async for event in client.stream_response():
            events.append(event)

        assert len(events) == 1
        assert events[0]["type"] == "text"


class TestSendMessageWithImage:
    @pytest.mark.asyncio
    async def test_send_message_includes_image_b64(self):
        client = RemoteClient(url="http://localhost:8080")
        mock_session = MagicMock()
        mock_session.closed = False
        mock_session.post = MagicMock(
            return_value=_make_response(200, {"status": "accepted"})
        )
        client._session = mock_session

        await client.send_message("describe this", image_b64="aGVsbG8=")

        call_args = mock_session.post.call_args
        payload = call_args[1]["json"]
        assert payload["text"] == "describe this"
        assert payload["image"] == "aGVsbG8="

    @pytest.mark.asyncio
    async def test_send_message_omits_image_when_none(self):
        client = RemoteClient(url="http://localhost:8080")
        mock_session = MagicMock()
        mock_session.closed = False
        mock_session.post = MagicMock(
            return_value=_make_response(200, {"status": "accepted"})
        )
        client._session = mock_session

        await client.send_message("text only", image_b64=None)

        call_args = mock_session.post.call_args
        payload = call_args[1]["json"]
        assert "image" not in payload


class TestRemoteClientHistory:
    @pytest.mark.asyncio
    async def test_history_returns_messages(self):
        client = RemoteClient(url="http://localhost:8080")
        mock_session = MagicMock()
        mock_session.closed = False
        mock_session.get = MagicMock(
            return_value=_make_response(
                200,
                {"messages": [
                    {"role": "user", "content": "hello", "created_at": "2026-03-20T10:00:00"},
                    {"role": "assistant", "content": "hi", "created_at": "2026-03-20T10:00:01"},
                ]},
            )
        )
        client._session = mock_session

        msgs = await client.history(limit=10)
        assert len(msgs) == 2
        assert msgs[0]["role"] == "user"
        assert msgs[1]["role"] == "assistant"

    @pytest.mark.asyncio
    async def test_history_calls_history_endpoint(self):
        client = RemoteClient(url="http://localhost:8080")
        mock_session = MagicMock()
        mock_session.closed = False
        mock_session.get = MagicMock(
            return_value=_make_response(200, {"messages": []})
        )
        client._session = mock_session

        await client.history(limit=5)

        call_url = mock_session.get.call_args[0][0]
        assert "/history" in call_url

    @pytest.mark.asyncio
    async def test_history_passes_limit_param(self):
        client = RemoteClient(url="http://localhost:8080")
        mock_session = MagicMock()
        mock_session.closed = False
        mock_session.get = MagicMock(
            return_value=_make_response(200, {"messages": []})
        )
        client._session = mock_session

        await client.history(limit=15)

        call_kwargs = mock_session.get.call_args[1]
        assert call_kwargs["params"]["limit"] == "15"

    @pytest.mark.asyncio
    async def test_history_raises_on_401(self):
        client = RemoteClient(url="http://localhost:8080")
        mock_session = MagicMock()
        mock_session.closed = False

        resp = AsyncMock()
        resp.status = 401
        ctx = AsyncMock()
        ctx.__aenter__ = AsyncMock(return_value=resp)
        ctx.__aexit__ = AsyncMock(return_value=False)
        mock_session.get = MagicMock(return_value=ctx)
        client._session = mock_session

        with pytest.raises(PermissionError, match="Authentication failed"):
            await client.history()
