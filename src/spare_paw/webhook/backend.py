"""HTTP webhook backend implementing MessageBackend.

Provides a REST API for sending/receiving messages without Telegram.
Supports per-session response routing, SSE streaming, and tool event visibility.
"""

from __future__ import annotations

import asyncio
import base64
import contextvars
import hmac
import json
import logging
import time
from typing import Any

from aiohttp import web

from spare_paw.backend import IncomingMessage

logger = logging.getLogger(__name__)

# Contextvar to thread session ID through process_message → send_text/on_tool_event
_current_session: contextvars.ContextVar[str] = contextvars.ContextVar(
    "session_id", default="default"
)

_SESSION_TTL = 600  # 10 minutes


class WebhookBackend:
    """HTTP-based message backend with per-session response delivery."""

    def __init__(self, port: int = 8080, secret: str = "", app_state: Any = None) -> None:
        if not secret:
            raise ValueError(
                "webhook.secret is required. Set a strong secret in config.yaml "
                "to protect the webhook API."
            )
        self._port = port
        self._secret = secret
        self._app_state = app_state
        self._session_queues: dict[str, asyncio.Queue[dict[str, Any]]] = {}
        self._session_last_seen: dict[str, float] = {}
        self._sse_queues: dict[str, list[asyncio.Queue[dict[str, Any]]]] = {}
        self._web_app: web.Application | None = None
        self._runner: web.AppRunner | None = None
        self._start_time: float = time.monotonic()

    def _check_auth(self, request: web.Request) -> bool:
        auth = request.headers.get("Authorization", "")
        expected = f"Bearer {self._secret}"
        return hmac.compare_digest(auth.encode(), expected.encode())

    _MAX_SESSIONS = 50

    def _get_session_queue(self, session_id: str) -> asyncio.Queue[dict[str, Any]]:
        if session_id not in self._session_queues:
            if len(self._session_queues) >= self._MAX_SESSIONS:
                self._cleanup_stale_sessions()
            if len(self._session_queues) >= self._MAX_SESSIONS:
                oldest = min(self._session_last_seen, key=self._session_last_seen.get)  # type: ignore[arg-type]
                self._session_queues.pop(oldest, None)
                self._session_last_seen.pop(oldest, None)
                self._sse_queues.pop(oldest, None)
            self._session_queues[session_id] = asyncio.Queue()
        self._session_last_seen[session_id] = time.monotonic()
        return self._session_queues[session_id]

    def _cleanup_stale_sessions(self) -> None:
        now = time.monotonic()
        stale = [
            sid for sid, ts in self._session_last_seen.items()
            if now - ts > _SESSION_TTL and sid != "default"
        ]
        for sid in stale:
            self._session_queues.pop(sid, None)
            self._session_last_seen.pop(sid, None)
            self._sse_queues.pop(sid, None)

    async def _handle_message(self, request: web.Request) -> web.Response:
        if not self._check_auth(request):
            return web.json_response({"error": "Unauthorized"}, status=401)

        try:
            data = await request.json()
        except (json.JSONDecodeError, Exception):
            return web.json_response({"error": "Invalid JSON"}, status=400)

        text = data.get("text")
        image_b64 = data.get("image")
        voice_b64 = data.get("voice")

        image_bytes = base64.b64decode(image_b64) if image_b64 else None
        voice_bytes = base64.b64decode(voice_b64) if voice_b64 else None

        msg = IncomingMessage(
            text=text,
            image_bytes=image_bytes,
            voice_bytes=voice_bytes,
        )

        session_id = request.headers.get("X-Session-Id", "default")
        # Touch session to keep it alive
        self._get_session_queue(session_id)

        if self._app_state is not None:
            asyncio.create_task(self._process_message(msg, session_id))

        # Lazy cleanup
        self._cleanup_stale_sessions()

        return web.json_response({"status": "accepted"})

    async def _process_message(self, msg: IncomingMessage, session_id: str = "default") -> None:
        """Route an incoming message through the engine."""
        _current_session.set(session_id)
        try:
            from spare_paw.core.engine import process_message

            await process_message(self._app_state, msg, self)
        except Exception:
            logger.exception("Error processing webhook message")

    async def _handle_poll(self, request: web.Request) -> web.Response:
        if not self._check_auth(request):
            return web.json_response({"error": "Unauthorized"}, status=401)

        session_id = request.headers.get("X-Session-Id", "default")
        queue = self._get_session_queue(session_id)
        timeout = float(request.query.get("timeout", "30"))
        messages: list[dict[str, Any]] = []

        try:
            msg = await asyncio.wait_for(queue.get(), timeout=timeout)
            messages.append(msg)
            while not queue.empty():
                messages.append(queue.get_nowait())
        except asyncio.TimeoutError:
            pass

        return web.json_response({"messages": messages})

    async def _handle_stream(self, request: web.Request) -> web.StreamResponse:
        """SSE endpoint that streams events in real-time."""
        if not self._check_auth(request):
            return web.json_response({"error": "Unauthorized"}, status=401)

        session_id = request.headers.get("X-Session-Id", "default")

        response = web.StreamResponse(
            status=200,
            reason="OK",
            headers={
                "Content-Type": "text/event-stream",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )
        await response.prepare(request)

        sse_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        if session_id not in self._sse_queues:
            self._sse_queues[session_id] = []
        self._sse_queues[session_id].append(sse_queue)
        try:
            while True:
                event = await sse_queue.get()
                data = json.dumps(event)
                await response.write(f"data: {data}\n\n".encode())
        except (asyncio.CancelledError, ConnectionResetError):
            pass
        finally:
            if session_id in self._sse_queues:
                try:
                    self._sse_queues[session_id].remove(sse_queue)
                except ValueError:
                    pass
                if not self._sse_queues[session_id]:
                    del self._sse_queues[session_id]

        return response

    async def _handle_health(self, _request: web.Request) -> web.Response:
        return web.json_response({"status": "ok"})

    async def _handle_status(self, request: web.Request) -> web.Response:
        if not self._check_auth(request):
            return web.json_response({"error": "Unauthorized"}, status=401)
        model = "unknown"
        tool_count = 0
        if self._app_state is not None:
            model = self._app_state.config.get("models.main_agent", "unknown")
            if self._app_state.tool_registry is not None:
                tool_count = len(self._app_state.tool_registry)
        uptime = int(time.monotonic() - self._start_time)
        return web.json_response({
            "status": "ok",
            "model": model,
            "tools": tool_count,
            "uptime": uptime,
        })

    def _broadcast_sse(self, event: dict[str, Any], session_id: str | None = None) -> None:
        """Send an event to SSE clients for the given session."""
        sid = session_id or _current_session.get()
        if sid in self._sse_queues:
            for q in self._sse_queues[sid]:
                q.put_nowait(event)

    def _put_session_queue(self, msg: dict[str, Any]) -> None:
        """Put a message on the current session's queue."""
        session_id = _current_session.get()
        queue = self._get_session_queue(session_id)
        queue.put_nowait(msg)

    # -- Tool event callback (duck-typed by engine) --

    def on_tool_event(self, event: Any) -> None:
        """Queue a tool event for poll and SSE clients."""
        msg: dict[str, Any]
        if event.kind == "tool_start":
            msg = {
                "type": "tool_call",
                "tool": event.tool_name,
                "args": str(event.tool_args)[:200] if event.tool_args else "",
            }
        elif event.kind == "tool_end":
            msg = {
                "type": "tool_result",
                "tool": event.tool_name,
                "preview": event.result_preview or "",
            }
        else:
            msg = {"type": event.kind}

        self._put_session_queue(msg)
        self._broadcast_sse(msg)

    def on_token(self, token: str) -> None:
        """Queue a streaming token for SSE clients."""
        event = {"type": "token", "token": token}
        self._broadcast_sse(event)

    # -- MessageBackend protocol --

    async def send_text(self, text: str) -> None:
        msg = {"type": "text", "text": text}
        self._put_session_queue(msg)
        self._broadcast_sse(msg)

    async def send_file(self, path: str, caption: str = "") -> None:
        msg = {
            "type": "file",
            "url": f"/files/{path}",
            "caption": caption,
        }
        self._put_session_queue(msg)
        self._broadcast_sse(msg)

    async def send_typing(self) -> None:
        self._broadcast_sse({"type": "typing"})

    async def send_notification(
        self, text: str, actions: list[dict] | None = None
    ) -> None:
        msg = {
            "type": "notification",
            "text": text,
            "actions": actions or [],
        }
        self._put_session_queue(msg)
        self._broadcast_sse(msg)

    async def _handle_history(self, request: web.Request) -> web.Response:
        if not self._check_auth(request):
            return web.json_response({"error": "Unauthorized"}, status=401)
        limit = min(int(request.query.get("limit", "10")), 100)

        from spare_paw.context import get_or_create_conversation, recent

        conversation_id = await get_or_create_conversation()
        messages = await recent(conversation_id, limit=limit)
        return web.json_response({"messages": messages})

    async def start(self) -> None:
        self._start_time = time.monotonic()
        self._web_app = web.Application()
        self._web_app.router.add_post("/message", self._handle_message)
        self._web_app.router.add_get("/poll", self._handle_poll)
        self._web_app.router.add_get("/stream", self._handle_stream)
        self._web_app.router.add_get("/health", self._handle_health)
        self._web_app.router.add_get("/status", self._handle_status)
        self._web_app.router.add_get("/history", self._handle_history)

        self._runner = web.AppRunner(self._web_app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, "0.0.0.0", self._port)
        await site.start()
        logger.info("Webhook backend listening on port %d", self._port)

    async def stop(self) -> None:
        if self._runner is not None:
            await self._runner.cleanup()
            logger.info("Webhook backend stopped")
