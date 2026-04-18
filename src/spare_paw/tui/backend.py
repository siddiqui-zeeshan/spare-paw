"""TUIBackend: MessageBackend adapter that bridges engine events to widgets.

The engine calls ``on_token`` / ``on_tool_event`` from background tasks.
These must be marshalled onto the Textual thread via ``App.post_message``.
"""

from __future__ import annotations

import uuid
from typing import Any

from spare_paw.tui.events import (
    AppendLog,
    StreamEnd,
    StreamToken,
    ToolCallEnd,
    ToolCallStart,
)


class TUIBackend:
    """MessageBackend implementation that posts Textual messages to an App."""

    def __init__(self, app: Any) -> None:
        self._app = app
        # Map tool_name (within one turn) -> call_id, so tool_end can correlate
        # with the corresponding tool_start. ToolEvent does not carry a call_id.
        self._active_tool_calls: dict[str, str] = {}

    async def send_text(self, text: str) -> None:
        """Called by the engine once streaming is complete with the final text.

        Posts a StreamEnd so MessageView can finalize, then appends the rendered
        markdown in case no streaming occurred (e.g., Telegram backends).
        """
        from rich.markdown import Markdown

        self._app.post_message(StreamEnd())
        self._app.post_message(AppendLog(Markdown(text)))

    async def send_file(self, path: str, caption: str = "") -> None:
        msg = f"File: {path}" + (f" \u2014 {caption}" if caption else "")
        self._app.post_message(AppendLog(f"[dim]{msg}[/dim]"))

    async def send_typing(self) -> None:
        return None

    async def send_notification(
        self, text: str, actions: list[dict] | None = None
    ) -> None:
        self._app.post_message(AppendLog(f"[yellow]{text}[/yellow]"))

    async def start(self) -> None:
        return None

    async def stop(self) -> None:
        return None

    def on_token(self, token: str) -> None:
        self._app.post_message(StreamToken(token))

    def on_tool_event(self, event: Any) -> None:
        kind = getattr(event, "kind", None)
        tool = getattr(event, "tool_name", None)
        if kind == "tool_start" and tool:
            call_id = uuid.uuid4().hex[:8]
            self._active_tool_calls[tool] = call_id
            args = getattr(event, "tool_args", None) or {}
            self._app.post_message(
                ToolCallStart(call_id=call_id, tool=tool, args=args)
            )
        elif kind == "tool_end" and tool:
            call_id = self._active_tool_calls.pop(tool, uuid.uuid4().hex[:8])
            preview = getattr(event, "result_preview", "") or ""
            success = not preview.lower().startswith("error")
            self._app.post_message(
                ToolCallEnd(
                    call_id=call_id,
                    success=success,
                    duration_ms=0,
                    preview=preview,
                )
            )
