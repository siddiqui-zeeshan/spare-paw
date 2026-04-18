"""Tests for TUIBackend and TUI message classes."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from spare_paw.tui.backend import TUIBackend
from spare_paw.tui.events import (
    AppendLog,
    StreamEnd,
    StreamToken,
    ToolCallEnd,
    ToolCallStart,
    UpdateStatus,
)


class TestAppendLog:
    def test_stores_content(self):
        msg = AppendLog("hello")
        assert msg.content == "hello"

    def test_accepts_any_content_type(self):
        rich_obj = MagicMock()
        msg = AppendLog(rich_obj)
        assert msg.content is rich_obj

    def test_accepts_empty_string(self):
        msg = AppendLog("")
        assert msg.content == ""


class TestUpdateStatus:
    def test_stores_text(self):
        msg = UpdateStatus("Connected | Model: gpt-4")
        assert msg.text == "Connected | Model: gpt-4"

    def test_accepts_empty_string(self):
        msg = UpdateStatus("")
        assert msg.text == ""


class TestStreamToken:
    def test_stores_token(self):
        msg = StreamToken("hello")
        assert msg.token == "hello"

    def test_accepts_empty_string(self):
        msg = StreamToken("")
        assert msg.token == ""


class TestStreamEnd:
    def test_can_instantiate(self):
        msg = StreamEnd()
        assert isinstance(msg, StreamEnd)


class TestTUIBackend:
    def _make_backend(self):
        app = MagicMock()
        app.post_message = MagicMock()
        return TUIBackend(app), app

    @pytest.mark.asyncio
    async def test_send_text_posts_messages(self):
        backend, app = self._make_backend()
        await backend.send_text("hello world")

        assert app.post_message.called
        # New backend: StreamEnd first, then AppendLog(Markdown)
        calls = [c[0][0] for c in app.post_message.call_args_list]
        assert any(isinstance(m, StreamEnd) for m in calls)
        assert any(isinstance(m, AppendLog) for m in calls)

    @pytest.mark.asyncio
    async def test_send_file_posts_append_log(self):
        backend, app = self._make_backend()
        await backend.send_file("/tmp/report.pdf", caption="my report")

        app.post_message.assert_called_once()
        msg = app.post_message.call_args[0][0]
        assert isinstance(msg, AppendLog)
        assert "report.pdf" in str(msg.content)
        assert "my report" in str(msg.content)

    @pytest.mark.asyncio
    async def test_send_typing_does_not_post(self):
        backend, app = self._make_backend()
        await backend.send_typing()
        app.post_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_send_notification_posts_append_log(self):
        backend, app = self._make_backend()
        await backend.send_notification("something happened")

        app.post_message.assert_called_once()
        msg = app.post_message.call_args[0][0]
        assert isinstance(msg, AppendLog)
        assert "something happened" in str(msg.content)

    def test_on_tool_event_tool_start_posts_tool_call_start(self):
        backend, app = self._make_backend()
        event = MagicMock()
        event.kind = "tool_start"
        event.tool_name = "shell"
        event.tool_args = {"command": "ls"}

        backend.on_tool_event(event)

        app.post_message.assert_called_once()
        msg = app.post_message.call_args[0][0]
        assert isinstance(msg, ToolCallStart)
        assert msg.tool == "shell"
        assert msg.args == {"command": "ls"}

    def test_on_tool_event_tool_start_forwards_args_dict(self):
        backend, app = self._make_backend()
        event = MagicMock()
        event.kind = "tool_start"
        event.tool_name = "read_file"
        event.tool_args = {"path": "/etc/hosts"}

        backend.on_tool_event(event)

        msg = app.post_message.call_args[0][0]
        assert isinstance(msg, ToolCallStart)
        assert msg.tool == "read_file"
        assert msg.args == {"path": "/etc/hosts"}

    def test_on_tool_event_non_tool_start_does_not_post_start(self):
        backend, app = self._make_backend()
        event = MagicMock()
        event.kind = "tool_end"
        event.tool_name = "shell"
        event.result_preview = "done"

        backend.on_tool_event(event)

        # tool_end dispatches ToolCallEnd, not ToolCallStart
        msg = app.post_message.call_args[0][0]
        assert isinstance(msg, ToolCallEnd)

    def test_on_tool_event_tool_start_no_name_does_not_post(self):
        backend, app = self._make_backend()
        event = MagicMock()
        event.kind = "tool_start"
        event.tool_name = None

        backend.on_tool_event(event)

        app.post_message.assert_not_called()

    def test_on_token_posts_stream_token(self):
        backend, app = self._make_backend()
        backend.on_token("hello")

        app.post_message.assert_called_once()
        msg = app.post_message.call_args[0][0]
        assert isinstance(msg, StreamToken)
        assert msg.token == "hello"

    @pytest.mark.asyncio
    async def test_start_does_not_post(self):
        backend, app = self._make_backend()
        await backend.start()
        app.post_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_stop_does_not_post(self):
        backend, app = self._make_backend()
        await backend.stop()
        app.post_message.assert_not_called()


class _CapturingApp:
    def __init__(self):
        self.messages = []

    def post_message(self, msg):
        self.messages.append(msg)


def test_new_tuibackend_on_token_dispatches_stream_token():
    app = _CapturingApp()
    backend = TUIBackend(app)
    backend.on_token("hello")
    assert len(app.messages) == 1
    assert isinstance(app.messages[0], StreamToken)
    assert app.messages[0].token == "hello"


def test_new_tuibackend_tool_start_dispatches_tool_call_start():
    app = _CapturingApp()
    backend = TUIBackend(app)

    class _Evt:
        kind = "tool_start"
        tool_name = "read_file"
        tool_args = {"path": "foo.py"}
        iteration = 1
        result_preview = None

    backend.on_tool_event(_Evt())
    assert any(isinstance(m, ToolCallStart) for m in app.messages)
    start = next(m for m in app.messages if isinstance(m, ToolCallStart))
    assert start.tool == "read_file"
    assert start.args == {"path": "foo.py"}


def test_new_tuibackend_tool_end_dispatches_tool_call_end():
    app = _CapturingApp()
    backend = TUIBackend(app)

    class _StartEvt:
        kind = "tool_start"
        tool_name = "shell"
        tool_args = {"cmd": "ls"}
        iteration = 1
        result_preview = None

    class _EndEvt:
        kind = "tool_end"
        tool_name = "shell"
        tool_args = None
        iteration = 1
        result_preview = "file1\nfile2"

    backend.on_tool_event(_StartEvt())
    backend.on_tool_event(_EndEvt())
    ends = [m for m in app.messages if isinstance(m, ToolCallEnd)]
    assert len(ends) == 1
    assert ends[0].success is True
    assert ends[0].preview == "file1\nfile2"
