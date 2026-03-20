"""Integration tests for the webhook backend HTTP API."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import MagicMock

import aiohttp
import pytest

from spare_paw.webhook.backend import WebhookBackend, _current_session

SECRET = "integration-secret"

# Port range: 18940–18960 (avoid collision with other test files)
_PORT_BASE = 18940
_port_counter = 0


def _next_port() -> int:
    global _port_counter
    _port_counter += 1
    return _PORT_BASE + _port_counter


class TestStatusEndpoint:
    @pytest.mark.asyncio
    async def test_status_returns_model_name(self):
        port = _next_port()
        app_state = MagicMock()
        app_state.config.get = MagicMock(return_value="claude-3-sonnet")
        app_state.tool_registry = list(range(3))

        backend = WebhookBackend(port=port, secret=SECRET, app_state=app_state)
        await backend.start()
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"http://localhost:{port}/status",
                    headers={"Authorization": f"Bearer {SECRET}"},
                ) as resp:
                    assert resp.status == 200
                    data = await resp.json()
                    assert data["status"] == "ok"
                    assert data["model"] == "claude-3-sonnet"
                    assert data["tools"] == 3
                    assert isinstance(data["uptime"], int)
        finally:
            await backend.stop()

    @pytest.mark.asyncio
    async def test_status_uptime_is_non_negative(self):
        port = _next_port()
        backend = WebhookBackend(port=port, secret=SECRET)
        await backend.start()
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"http://localhost:{port}/status",
                    headers={"Authorization": f"Bearer {SECRET}"},
                ) as resp:
                    data = await resp.json()
                    assert data["uptime"] >= 0
        finally:
            await backend.stop()

    @pytest.mark.asyncio
    async def test_status_without_auth_returns_401(self):
        port = _next_port()
        backend = WebhookBackend(port=port, secret=SECRET)
        await backend.start()
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"http://localhost:{port}/status") as resp:
                    assert resp.status == 401
        finally:
            await backend.stop()


class TestPollWithSessionId:
    @pytest.mark.asyncio
    async def test_different_session_ids_get_independent_queues(self):
        port = _next_port()
        backend = WebhookBackend(port=port, secret=SECRET)
        await backend.start()
        try:
            # Pre-populate two different sessions directly
            tok_a = _current_session.set("poll-session-a")
            await backend.send_text("message for A")
            _current_session.reset(tok_a)

            tok_b = _current_session.set("poll-session-b")
            await backend.send_text("message for B")
            _current_session.reset(tok_b)

            async with aiohttp.ClientSession() as session:
                # Poll session A
                async with session.get(
                    f"http://localhost:{port}/poll",
                    params={"timeout": "0.1"},
                    headers={
                        "Authorization": f"Bearer {SECRET}",
                        "X-Session-Id": "poll-session-a",
                    },
                    timeout=aiohttp.ClientTimeout(total=5),
                ) as resp:
                    data_a = await resp.json()

                # Poll session B
                async with session.get(
                    f"http://localhost:{port}/poll",
                    params={"timeout": "0.1"},
                    headers={
                        "Authorization": f"Bearer {SECRET}",
                        "X-Session-Id": "poll-session-b",
                    },
                    timeout=aiohttp.ClientTimeout(total=5),
                ) as resp:
                    data_b = await resp.json()

            assert len(data_a["messages"]) == 1
            assert data_a["messages"][0]["text"] == "message for A"

            assert len(data_b["messages"]) == 1
            assert data_b["messages"][0]["text"] == "message for B"
        finally:
            await backend.stop()

    @pytest.mark.asyncio
    async def test_poll_empty_when_no_messages(self):
        port = _next_port()
        backend = WebhookBackend(port=port, secret=SECRET)
        await backend.start()
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"http://localhost:{port}/poll",
                    params={"timeout": "0.1"},
                    headers={
                        "Authorization": f"Bearer {SECRET}",
                        "X-Session-Id": "empty-session",
                    },
                    timeout=aiohttp.ClientTimeout(total=5),
                ) as resp:
                    data = await resp.json()
                    assert data["messages"] == []
        finally:
            await backend.stop()

    @pytest.mark.asyncio
    async def test_poll_drains_all_queued_messages(self):
        port = _next_port()
        backend = WebhookBackend(port=port, secret=SECRET)
        await backend.start()
        try:
            tok = _current_session.set("drain-session")
            await backend.send_text("msg1")
            await backend.send_text("msg2")
            await backend.send_text("msg3")
            _current_session.reset(tok)

            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"http://localhost:{port}/poll",
                    params={"timeout": "0.5"},
                    headers={
                        "Authorization": f"Bearer {SECRET}",
                        "X-Session-Id": "drain-session",
                    },
                    timeout=aiohttp.ClientTimeout(total=5),
                ) as resp:
                    data = await resp.json()

            assert len(data["messages"]) == 3
        finally:
            await backend.stop()

    @pytest.mark.asyncio
    async def test_poll_without_auth_returns_401(self):
        port = _next_port()
        backend = WebhookBackend(port=port, secret=SECRET)
        await backend.start()
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"http://localhost:{port}/poll",
                    params={"timeout": "0.1"},
                    timeout=aiohttp.ClientTimeout(total=5),
                ) as resp:
                    assert resp.status == 401
        finally:
            await backend.stop()


@pytest.mark.skipif(
    "CI" in __import__("os").environ,
    reason="SSE integration tests are too slow for CI",
)
class TestStreamSSEEndpoint:
    @pytest.mark.asyncio
    async def test_stream_returns_sse_content_type(self):
        port = _next_port()
        backend = WebhookBackend(port=port, secret=SECRET)
        await backend.start()
        try:
            received: list[dict] = []

            async def _consume() -> None:
                async with aiohttp.ClientSession() as http:
                    async with http.get(
                        f"http://localhost:{port}/stream",
                        headers={
                            "Authorization": f"Bearer {SECRET}",
                            "X-Session-Id": "sse-type-test",
                        },
                        timeout=aiohttp.ClientTimeout(total=5),
                    ) as resp:
                        assert resp.status == 200
                        assert "text/event-stream" in resp.content_type
                        async for line in resp.content:
                            decoded = line.decode("utf-8").strip()
                            if decoded.startswith("data: "):
                                received.append(json.loads(decoded[6:]))
                                break

            consumer = asyncio.create_task(_consume())
            await asyncio.sleep(0.5)

            backend._broadcast_sse(
                {"type": "text", "text": "sse works"},
                session_id="sse-type-test",
            )

            await asyncio.wait_for(consumer, timeout=10)
            assert len(received) == 1
            assert received[0]["type"] == "text"
        finally:
            await backend.stop()

    @pytest.mark.asyncio
    async def test_stream_events_in_data_colon_format(self):
        port = _next_port()
        backend = WebhookBackend(port=port, secret=SECRET)
        await backend.start()
        try:
            raw_lines: list[str] = []

            async def _consume() -> None:
                async with aiohttp.ClientSession() as http:
                    async with http.get(
                        f"http://localhost:{port}/stream",
                        headers={
                            "Authorization": f"Bearer {SECRET}",
                            "X-Session-Id": "sse-format-test",
                        },
                        timeout=aiohttp.ClientTimeout(total=5),
                    ) as resp:
                        async for line in resp.content:
                            decoded = line.decode("utf-8").strip()
                            if decoded:
                                raw_lines.append(decoded)
                                break

            consumer = asyncio.create_task(_consume())
            await asyncio.sleep(0.5)

            backend._broadcast_sse(
                {"type": "token", "token": "x"},
                session_id="sse-format-test",
            )

            await asyncio.wait_for(consumer, timeout=10)
            assert any(line.startswith("data: ") for line in raw_lines)
        finally:
            await backend.stop()

    @pytest.mark.asyncio
    async def test_stream_without_auth_returns_401(self):
        port = _next_port()
        backend = WebhookBackend(port=port, secret=SECRET)
        await backend.start()
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"http://localhost:{port}/stream"
                ) as resp:
                    assert resp.status == 401
        finally:
            await backend.stop()

    @pytest.mark.asyncio
    async def test_stream_multiple_sessions_independent(self):
        """Two SSE clients on different sessions receive only their own events."""
        port = _next_port()
        backend = WebhookBackend(port=port, secret=SECRET)
        await backend.start()
        try:
            received_a: list[dict] = []
            received_b: list[dict] = []

            async def _consume(session_id: str, store: list) -> None:
                async with aiohttp.ClientSession() as http:
                    async with http.get(
                        f"http://localhost:{port}/stream",
                        headers={
                            "Authorization": f"Bearer {SECRET}",
                            "X-Session-Id": session_id,
                        },
                        timeout=aiohttp.ClientTimeout(total=5),
                    ) as resp:
                        async for line in resp.content:
                            decoded = line.decode("utf-8").strip()
                            if decoded.startswith("data: "):
                                store.append(json.loads(decoded[6:]))
                                break

            task_a = asyncio.create_task(_consume("multi-a", received_a))
            task_b = asyncio.create_task(_consume("multi-b", received_b))
            await asyncio.sleep(0.5)

            backend._broadcast_sse({"type": "text", "text": "for A"}, session_id="multi-a")
            backend._broadcast_sse({"type": "text", "text": "for B"}, session_id="multi-b")

            await asyncio.wait_for(task_a, timeout=10)
            await asyncio.wait_for(task_b, timeout=10)

            assert received_a[0]["text"] == "for A"
            assert received_b[0]["text"] == "for B"
        finally:
            await backend.stop()
