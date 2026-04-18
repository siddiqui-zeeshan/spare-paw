"""Textual Message dataclasses for TUI inter-widget communication.

All messages posted via ``App.post_message`` from background tasks must be
instances of ``textual.message.Message`` to be thread-safe.
"""

from __future__ import annotations

from typing import Any

from textual.message import Message


class AppendLog(Message):
    """Append arbitrary rich content to the chat log."""

    def __init__(self, content: Any) -> None:
        super().__init__()
        self.content = content


class AppendError(Message):
    """Append a visible, dim-red error row to the chat log."""

    def __init__(self, text: str) -> None:
        super().__init__()
        self.text = text


class UpdateStatus(Message):
    """Update the bottom status bar text."""

    def __init__(self, text: str) -> None:
        super().__init__()
        self.text = text


class StreamToken(Message):
    """A single streamed text token from the model."""

    def __init__(self, token: str) -> None:
        super().__init__()
        self.token = token


class StreamEnd(Message):
    """Signal that streaming has finished for the current assistant turn."""


class ToolCallStart(Message):
    """A tool call has started executing."""

    def __init__(self, call_id: str, tool: str, args: dict) -> None:
        super().__init__()
        self.call_id = call_id
        self.tool = tool
        self.args = args


class ToolCallEnd(Message):
    """A tool call has finished (success or failure)."""

    def __init__(
        self, call_id: str, success: bool, duration_ms: int, preview: str
    ) -> None:
        super().__init__()
        self.call_id = call_id
        self.success = success
        self.duration_ms = duration_ms
        self.preview = preview


class ConnectionStateChanged(Message):
    """Remote-mode connection state changed (connected | reconnecting | disconnected)."""

    def __init__(self, state: str, detail: str = "") -> None:
        super().__init__()
        self.state = state
        self.detail = detail
