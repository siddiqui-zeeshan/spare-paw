"""Tests for TUIBackend and TUI message classes."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock

import pytest

from spare_paw.tui.app import (
    SLASH_COMMANDS,
    SPINNER_FRAMES,
    THINKING_VERBS,
    TOOL_ICONS,
    AppendLog,
    StreamEnd,
    StreamToken,
    ToolCallEvent,
    TUIBackend,
    UpdateStatus,
    _format_timestamp,
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


class TestToolCallEvent:
    def test_stores_tool_and_args(self):
        msg = ToolCallEvent("web_search", 'query="weather"')
        assert msg.tool == "web_search"
        assert msg.args == 'query="weather"'

    def test_accepts_empty_args(self):
        msg = ToolCallEvent("shell", "")
        assert msg.tool == "shell"
        assert msg.args == ""


class TestFormatTimestamp:
    def test_returns_hhmm_am_pm(self):
        dt = datetime(2026, 3, 20, 14, 5)
        result = _format_timestamp(dt)
        assert result == "2:05 PM"

    def test_midnight(self):
        dt = datetime(2026, 3, 20, 0, 0)
        result = _format_timestamp(dt)
        assert result == "12:00 AM"

    def test_noon(self):
        dt = datetime(2026, 3, 20, 12, 0)
        result = _format_timestamp(dt)
        assert result == "12:00 PM"

    def test_defaults_to_now(self):
        result = _format_timestamp()
        assert "AM" in result or "PM" in result


class TestTUIBackend:
    def _make_backend(self):
        app = MagicMock()
        app.post_message = MagicMock()
        return TUIBackend(app), app

    @pytest.mark.asyncio
    async def test_send_text_posts_append_log(self):
        backend, app = self._make_backend()
        await backend.send_text("hello world")

        assert app.post_message.called
        # First call should be AppendLog with Markdown content
        first_call_arg = app.post_message.call_args_list[0][0][0]
        assert isinstance(first_call_arg, AppendLog)

    @pytest.mark.asyncio
    async def test_send_text_posts_two_messages(self):
        """send_text posts Markdown content and then an empty separator."""
        backend, app = self._make_backend()
        await backend.send_text("hello")

        assert app.post_message.call_count == 2
        second_call_arg = app.post_message.call_args_list[1][0][0]
        assert isinstance(second_call_arg, AppendLog)
        assert second_call_arg.content == ""

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

    def test_on_tool_event_tool_start_posts_tool_call_event(self):
        backend, app = self._make_backend()
        event = MagicMock()
        event.kind = "tool_start"
        event.tool_name = "shell"
        event.tool_args = {"command": "ls"}

        backend.on_tool_event(event)

        app.post_message.assert_called_once()
        msg = app.post_message.call_args[0][0]
        assert isinstance(msg, ToolCallEvent)
        assert msg.tool == "shell"
        assert "command" in msg.args

    def test_on_tool_event_tool_start_formats_args(self):
        backend, app = self._make_backend()
        event = MagicMock()
        event.kind = "tool_start"
        event.tool_name = "read_file"
        event.tool_args = {"path": "/etc/hosts"}

        backend.on_tool_event(event)

        msg = app.post_message.call_args[0][0]
        assert isinstance(msg, ToolCallEvent)
        assert msg.tool == "read_file"
        assert "path" in msg.args

    def test_on_tool_event_non_tool_start_does_not_post(self):
        backend, app = self._make_backend()
        event = MagicMock()
        event.kind = "tool_end"
        event.tool_name = "shell"

        backend.on_tool_event(event)

        app.post_message.assert_not_called()

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

    def test_on_tool_event_truncates_args_to_three(self):
        backend, app = self._make_backend()
        event = MagicMock()
        event.kind = "tool_start"
        event.tool_name = "big_tool"
        event.tool_args = {
            "a": 1,
            "b": 2,
            "c": 3,
            "d": 4,
            "e": 5,
        }

        backend.on_tool_event(event)

        msg = app.post_message.call_args[0][0]
        assert isinstance(msg, ToolCallEvent)
        # At most 3 key=value pairs rendered
        assert msg.args.count("=") <= 3


class TestSpinnerFrames:
    def test_frames_exist(self):
        assert len(SPINNER_FRAMES) > 0

    def test_frames_are_strings(self):
        for frame in SPINNER_FRAMES:
            assert isinstance(frame, str)

    def test_mirror_pattern(self):
        n = len(SPINNER_FRAMES)
        for i in range(1, n // 2):
            assert SPINNER_FRAMES[i] == SPINNER_FRAMES[n - i]


class TestThinkingVerbs:
    def test_verbs_exist(self):
        assert len(THINKING_VERBS) > 0

    def test_all_end_with_ing(self):
        for verb in THINKING_VERBS:
            assert verb.endswith("ing")

    def test_contains_purring(self):
        assert "Purring" in THINKING_VERBS


class TestToolIcons:
    def test_icons_exist(self):
        assert len(TOOL_ICONS) > 0

    def test_shell_has_icon(self):
        assert "shell" in TOOL_ICONS

    def test_read_file_has_icon(self):
        assert "read_file" in TOOL_ICONS


class TestSlashCommands:
    def test_commands_list_not_empty(self):
        assert len(SLASH_COMMANDS) > 0

    def test_all_start_with_slash(self):
        for cmd in SLASH_COMMANDS:
            assert cmd.startswith("/")

    def test_contains_help(self):
        assert "/help" in SLASH_COMMANDS

    def test_contains_exit(self):
        assert "/exit" in SLASH_COMMANDS

    def test_contains_forget(self):
        assert "/forget" in SLASH_COMMANDS


from spare_paw.tui.backend import TUIBackend as NewTUIBackend  # noqa: E402
from spare_paw.tui.events import StreamToken as EvStreamToken  # noqa: E402
from spare_paw.tui.events import ToolCallEnd as EvToolCallEnd  # noqa: E402
from spare_paw.tui.events import ToolCallStart as EvToolCallStart  # noqa: E402


class _CapturingApp:
    def __init__(self):
        self.messages = []
    def post_message(self, msg):
        self.messages.append(msg)


def test_new_tuibackend_on_token_dispatches_stream_token():
    app = _CapturingApp()
    backend = NewTUIBackend(app)
    backend.on_token("hello")
    assert len(app.messages) == 1
    assert isinstance(app.messages[0], EvStreamToken)
    assert app.messages[0].token == "hello"


def test_new_tuibackend_tool_start_dispatches_tool_call_start():
    app = _CapturingApp()
    backend = NewTUIBackend(app)

    class _Evt:
        kind = "tool_start"
        tool_name = "read_file"
        tool_args = {"path": "foo.py"}
        iteration = 1
        result_preview = None

    backend.on_tool_event(_Evt())
    assert any(isinstance(m, EvToolCallStart) for m in app.messages)
    start = next(m for m in app.messages if isinstance(m, EvToolCallStart))
    assert start.tool == "read_file"
    assert start.args == {"path": "foo.py"}


def test_new_tuibackend_tool_end_dispatches_tool_call_end():
    app = _CapturingApp()
    backend = NewTUIBackend(app)

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
    ends = [m for m in app.messages if isinstance(m, EvToolCallEnd)]
    assert len(ends) == 1
    assert ends[0].success is True
    assert ends[0].preview == "file1\nfile2"
