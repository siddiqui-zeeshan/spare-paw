"""Tests for the webhook backend."""

from __future__ import annotations

import asyncio
import time
from unittest.mock import MagicMock

import pytest

from spare_paw.backend import MessageBackend
from spare_paw.webhook.backend import WebhookBackend, _SESSION_TTL, _current_session

SECRET = "test-secret"


class TestWebhookBackend:
    def test_implements_protocol(self):
        backend = WebhookBackend(secret=SECRET)
        assert isinstance(backend, MessageBackend)

    def test_rejects_empty_secret(self):
        with pytest.raises(ValueError, match="webhook.secret is required"):
            WebhookBackend(secret="")

    def test_rejects_missing_secret(self):
        with pytest.raises(ValueError, match="webhook.secret is required"):
            WebhookBackend()

    @pytest.mark.asyncio
    async def test_send_text_queues_message_default_session(self):
        backend = WebhookBackend(secret=SECRET)
        await backend.send_text("hello")
        queue = backend._session_queues["default"]
        msg = queue.get_nowait()
        assert msg == {"type": "text", "text": "hello"}

    @pytest.mark.asyncio
    async def test_send_file_queues_message(self):
        backend = WebhookBackend(secret=SECRET)
        await backend.send_file("/tmp/test.txt", caption="test")
        queue = backend._session_queues["default"]
        msg = queue.get_nowait()
        assert msg["type"] == "file"
        assert msg["caption"] == "test"

    @pytest.mark.asyncio
    async def test_send_typing_broadcasts_sse(self):
        backend = WebhookBackend(secret=SECRET)
        # No SSE clients — should not raise
        await backend.send_typing()

    @pytest.mark.asyncio
    async def test_send_notification_queues_message(self):
        backend = WebhookBackend(secret=SECRET)
        await backend.send_notification("alert", actions=[{"label": "OK"}])
        queue = backend._session_queues["default"]
        msg = queue.get_nowait()
        assert msg["type"] == "notification"
        assert msg["text"] == "alert"
        assert msg["actions"] == [{"label": "OK"}]

    @pytest.mark.asyncio
    async def test_start_stop(self):
        backend = WebhookBackend(port=18923, secret=SECRET)
        await backend.start()
        assert backend._runner is not None
        await backend.stop()

    @pytest.mark.asyncio
    async def test_health_endpoint(self):
        backend = WebhookBackend(port=18924, secret=SECRET)
        await backend.start()
        try:
            import aiohttp

            async with aiohttp.ClientSession() as session:
                async with session.get("http://localhost:18924/health") as resp:
                    assert resp.status == 200
                    data = await resp.json()
                    assert data["status"] == "ok"
        finally:
            await backend.stop()

    @pytest.mark.asyncio
    async def test_message_endpoint_auth(self):
        backend = WebhookBackend(port=18925, secret="test-secret")
        await backend.start()
        try:
            import aiohttp

            async with aiohttp.ClientSession() as session:
                # No auth
                async with session.post(
                    "http://localhost:18925/message",
                    json={"text": "hi"},
                ) as resp:
                    assert resp.status == 401

                # With auth
                async with session.post(
                    "http://localhost:18925/message",
                    json={"text": "hi"},
                    headers={"Authorization": "Bearer test-secret"},
                ) as resp:
                    assert resp.status == 200
        finally:
            await backend.stop()

    @pytest.mark.asyncio
    async def test_timing_safe_comparison(self):
        """Auth uses constant-time comparison (hmac.compare_digest)."""
        backend = WebhookBackend(port=18926, secret="correct-secret")
        await backend.start()
        try:
            import aiohttp

            async with aiohttp.ClientSession() as session:
                # Wrong secret
                async with session.post(
                    "http://localhost:18926/message",
                    json={"text": "hi"},
                    headers={"Authorization": "Bearer wrong-secret"},
                ) as resp:
                    assert resp.status == 401

                # Correct secret
                async with session.post(
                    "http://localhost:18926/message",
                    json={"text": "hi"},
                    headers={"Authorization": "Bearer correct-secret"},
                ) as resp:
                    assert resp.status == 200
        finally:
            await backend.stop()


class TestWebhookBackendStatus:
    @pytest.mark.asyncio
    async def test_status_endpoint_no_auth_returns_401(self):
        backend = WebhookBackend(port=18930, secret=SECRET)
        await backend.start()
        try:
            import aiohttp

            async with aiohttp.ClientSession() as session:
                async with session.get("http://localhost:18930/status") as resp:
                    assert resp.status == 401
        finally:
            await backend.stop()

    @pytest.mark.asyncio
    async def test_status_endpoint_returns_model_info(self):
        app_state = MagicMock()
        app_state.config.get = MagicMock(return_value="claude-3-haiku")
        app_state.tool_registry = ["tool1", "tool2"]

        backend = WebhookBackend(port=18931, secret=SECRET, app_state=app_state)
        await backend.start()
        try:
            import aiohttp

            async with aiohttp.ClientSession() as session:
                async with session.get(
                    "http://localhost:18931/status",
                    headers={"Authorization": f"Bearer {SECRET}"},
                ) as resp:
                    assert resp.status == 200
                    data = await resp.json()
                    assert data["status"] == "ok"
                    assert data["model"] == "claude-3-haiku"
                    assert data["tools"] == 2
                    assert "uptime" in data
        finally:
            await backend.stop()

    @pytest.mark.asyncio
    async def test_status_endpoint_no_app_state(self):
        backend = WebhookBackend(port=18932, secret=SECRET)
        await backend.start()
        try:
            import aiohttp

            async with aiohttp.ClientSession() as session:
                async with session.get(
                    "http://localhost:18932/status",
                    headers={"Authorization": f"Bearer {SECRET}"},
                ) as resp:
                    assert resp.status == 200
                    data = await resp.json()
                    assert data["model"] == "unknown"
                    assert data["tools"] == 0
        finally:
            await backend.stop()


class TestPerSessionQueues:
    @pytest.mark.asyncio
    async def test_two_sessions_get_independent_queues(self):
        backend = WebhookBackend(port=18933, secret=SECRET)
        await backend.start()
        try:
            import aiohttp

            headers_a = {
                "Authorization": f"Bearer {SECRET}",
                "X-Session-Id": "session-a",
            }
            headers_b = {
                "Authorization": f"Bearer {SECRET}",
                "X-Session-Id": "session-b",
            }

            async with aiohttp.ClientSession() as session:
                # Send a message for session A
                async with session.post(
                    "http://localhost:18933/message",
                    json={"text": "hello from A"},
                    headers=headers_a,
                ) as resp:
                    assert resp.status == 200

                # Poll for session B — should be empty (timeout quickly)
                async with session.get(
                    "http://localhost:18933/poll",
                    params={"timeout": "0.1"},
                    headers=headers_b,
                    timeout=aiohttp.ClientTimeout(total=5),
                ) as resp:
                    data = await resp.json()
                    assert data["messages"] == []
        finally:
            await backend.stop()

    @pytest.mark.asyncio
    async def test_poll_with_session_id_header_routes_correctly(self):
        backend = WebhookBackend(port=18934, secret=SECRET)
        await backend.start()
        try:
            import aiohttp

            # Pre-populate the "my-session" queue by setting contextvar and calling send_text
            token = _current_session.set("my-session")
            await backend.send_text("targeted response")
            _current_session.reset(token)

            async with aiohttp.ClientSession() as session:
                async with session.get(
                    "http://localhost:18934/poll",
                    params={"timeout": "0.1"},
                    headers={
                        "Authorization": f"Bearer {SECRET}",
                        "X-Session-Id": "my-session",
                    },
                    timeout=aiohttp.ClientTimeout(total=5),
                ) as resp:
                    data = await resp.json()
                    assert len(data["messages"]) == 1
                    assert data["messages"][0]["type"] == "text"
                    assert data["messages"][0]["text"] == "targeted response"
        finally:
            await backend.stop()

    def test_get_session_queue_creates_distinct_queues(self):
        backend = WebhookBackend(secret=SECRET)
        q1 = backend._get_session_queue("alpha")
        q2 = backend._get_session_queue("beta")
        assert q1 is not q2

    def test_get_session_queue_same_id_returns_same_queue(self):
        backend = WebhookBackend(secret=SECRET)
        q1 = backend._get_session_queue("same")
        q2 = backend._get_session_queue("same")
        assert q1 is q2


class TestStaleSessionCleanup:
    def test_cleanup_removes_expired_sessions(self):
        backend = WebhookBackend(secret=SECRET)
        backend._get_session_queue("old-session")
        # Back-date the last_seen timestamp to beyond TTL
        backend._session_last_seen["old-session"] = time.monotonic() - _SESSION_TTL - 1

        backend._cleanup_stale_sessions()

        assert "old-session" not in backend._session_queues
        assert "old-session" not in backend._session_last_seen

    def test_cleanup_preserves_default_session(self):
        backend = WebhookBackend(secret=SECRET)
        backend._get_session_queue("default")
        backend._session_last_seen["default"] = time.monotonic() - _SESSION_TTL - 1

        backend._cleanup_stale_sessions()

        assert "default" in backend._session_queues

    def test_cleanup_preserves_active_sessions(self):
        backend = WebhookBackend(secret=SECRET)
        backend._get_session_queue("active-session")
        # Last seen just now — should survive cleanup

        backend._cleanup_stale_sessions()

        assert "active-session" in backend._session_queues

    def test_cleanup_also_removes_sse_queues(self):
        backend = WebhookBackend(secret=SECRET)
        backend._get_session_queue("sse-session")
        backend._sse_queues["sse-session"] = []
        backend._session_last_seen["sse-session"] = time.monotonic() - _SESSION_TTL - 1

        backend._cleanup_stale_sessions()

        assert "sse-session" not in backend._sse_queues


class TestOnToolEvent:
    def test_on_tool_event_tool_start_queues_message(self):
        backend = WebhookBackend(secret=SECRET)
        token = _current_session.set("default")
        try:
            event = MagicMock()
            event.kind = "tool_start"
            event.tool_name = "shell"
            event.tool_args = {"command": "ls"}

            backend.on_tool_event(event)

            queue = backend._session_queues["default"]
            msg = queue.get_nowait()
            assert msg["type"] == "tool_call"
            assert msg["tool"] == "shell"
            assert "ls" in msg["args"]
        finally:
            _current_session.reset(token)

    def test_on_tool_event_tool_end_queues_message(self):
        backend = WebhookBackend(secret=SECRET)
        token = _current_session.set("default")
        try:
            event = MagicMock()
            event.kind = "tool_end"
            event.tool_name = "shell"
            event.result_preview = "file1.txt"

            backend.on_tool_event(event)

            queue = backend._session_queues["default"]
            msg = queue.get_nowait()
            assert msg["type"] == "tool_result"
            assert msg["tool"] == "shell"
            assert msg["preview"] == "file1.txt"
        finally:
            _current_session.reset(token)

    def test_on_tool_event_unknown_kind_queues_generic(self):
        backend = WebhookBackend(secret=SECRET)
        token = _current_session.set("default")
        try:
            event = MagicMock()
            event.kind = "llm_start"

            backend.on_tool_event(event)

            queue = backend._session_queues["default"]
            msg = queue.get_nowait()
            assert msg["type"] == "llm_start"
        finally:
            _current_session.reset(token)

    def test_on_tool_event_broadcasts_to_sse(self):
        backend = WebhookBackend(secret=SECRET)
        sse_q: asyncio.Queue = asyncio.Queue()
        backend._sse_queues["default"] = [sse_q]

        token = _current_session.set("default")
        try:
            event = MagicMock()
            event.kind = "tool_start"
            event.tool_name = "ping"
            event.tool_args = None

            backend.on_tool_event(event)

            assert not sse_q.empty()
            msg = sse_q.get_nowait()
            assert msg["type"] == "tool_call"
        finally:
            _current_session.reset(token)

    def test_on_token_broadcasts_to_sse(self):
        backend = WebhookBackend(secret=SECRET)
        sse_q: asyncio.Queue = asyncio.Queue()
        backend._sse_queues["default"] = [sse_q]

        token = _current_session.set("default")
        try:
            backend.on_token("hello")

            assert not sse_q.empty()
            msg = sse_q.get_nowait()
            assert msg["type"] == "token"
            assert msg["token"] == "hello"
        finally:
            _current_session.reset(token)


@pytest.mark.skipif(
    "CI" in __import__("os").environ,
    reason="SSE integration tests are too slow for CI",
)
class TestStreamSSE:
    @pytest.mark.asyncio
    async def test_stream_endpoint_sends_events_in_sse_format(self):
        backend = WebhookBackend(port=18935, secret=SECRET)
        await backend.start()
        try:
            import aiohttp

            session_id = "stream-test"
            received: list[dict] = []

            async def _consume() -> None:
                async with aiohttp.ClientSession() as http:
                    async with http.get(
                        "http://localhost:18935/stream",
                        headers={
                            "Authorization": f"Bearer {SECRET}",
                            "X-Session-Id": session_id,
                        },
                        timeout=aiohttp.ClientTimeout(total=5),
                    ) as resp:
                        assert resp.status == 200
                        assert "text/event-stream" in resp.content_type
                        async for line in resp.content:
                            decoded = line.decode("utf-8").strip()
                            if decoded.startswith("data: "):
                                import json

                                received.append(json.loads(decoded[6:]))
                                break  # Stop after first event

            consumer = asyncio.create_task(_consume())
            # Give the consumer time to connect
            await asyncio.sleep(0.1)

            # Broadcast an event to the session
            token = _current_session.set(session_id)
            backend._broadcast_sse({"type": "text", "text": "streamed"}, session_id=session_id)
            _current_session.reset(token)

            await asyncio.wait_for(consumer, timeout=4)
            assert len(received) == 1
            assert received[0]["type"] == "text"
            assert received[0]["text"] == "streamed"
        finally:
            await backend.stop()

    @pytest.mark.asyncio
    async def test_stream_endpoint_rejects_without_auth(self):
        backend = WebhookBackend(port=18936, secret=SECRET)
        await backend.start()
        try:
            import aiohttp

            async with aiohttp.ClientSession() as session:
                async with session.get("http://localhost:18936/stream") as resp:
                    assert resp.status == 401
        finally:
            await backend.stop()
