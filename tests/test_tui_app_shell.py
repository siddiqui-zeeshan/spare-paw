from __future__ import annotations

import pytest

from spare_paw.tui.app import SparePawTUI
from spare_paw.tui.events import StreamEnd, StreamToken, ToolCallEnd, ToolCallStart
from spare_paw.tui.widgets.chat_log import ChatLog
from spare_paw.tui.widgets.composer import Composer
from spare_paw.tui.widgets.message_view import MessageView
from spare_paw.tui.widgets.status_bar import StatusBar


@pytest.mark.asyncio
async def test_app_composes_core_widgets():
    app = SparePawTUI(client=None, app_state=None)
    async with app.run_test():
        assert app.query_one(ChatLog) is not None
        assert app.query_one(Composer) is not None
        assert app.query_one(StatusBar) is not None


@pytest.mark.asyncio
async def test_stream_token_renders_into_active_message():
    app = SparePawTUI(client=None, app_state=None)
    async with app.run_test() as pilot:
        log = app.query_one(ChatLog)
        turn = MessageView(role="assistant")
        log.mount_turn(turn)
        await pilot.pause()
        app.post_message(StreamToken("hello"))
        await pilot.pause(0.05)
        assert "hello" in turn.live_text
        app.post_message(StreamToken(" world"))
        await pilot.pause(0.05)
        assert "hello world" in turn.live_text
        app.post_message(StreamEnd())
        await pilot.pause(0.05)
        assert turn.finalized is True


@pytest.mark.asyncio
async def test_tool_call_mounts_inline_row():
    app = SparePawTUI(client=None, app_state=None)
    async with app.run_test() as pilot:
        log = app.query_one(ChatLog)
        turn = MessageView(role="assistant")
        log.mount_turn(turn)
        await pilot.pause()
        app.post_message(ToolCallStart(call_id="c1", tool="read_file", args={"path": "x"}))
        await pilot.pause(0.05)
        assert turn.tool_row_count() == 1
        app.post_message(ToolCallEnd(call_id="c1", success=True, duration_ms=100, preview="ok"))
        await pilot.pause(0.05)
