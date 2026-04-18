from __future__ import annotations

import pytest
from textual.app import App

from spare_paw.tui.widgets.status_bar import StatusBar


class _Host(App):
    def compose(self):
        yield StatusBar(id="sb")


@pytest.mark.asyncio
async def test_status_bar_renders_all_fields():
    app = _Host()
    async with app.run_test() as pilot:
        sb = app.query_one("#sb", StatusBar)
        sb.set_state(
            connection="connected", url="http://x", model="m",
            msg_count=3, tool_count=1,
        )
        await pilot.pause()
        assert "m" in sb.render_text()
        assert "3 msgs" in sb.render_text()
        assert "1 tools" in sb.render_text()


@pytest.mark.asyncio
async def test_status_bar_connection_color():
    app = _Host()
    async with app.run_test() as pilot:
        sb = app.query_one("#sb", StatusBar)
        sb.set_state(connection="disconnected", url="x", model="m", msg_count=0, tool_count=0)
        await pilot.pause()
        assert "[red]" in sb.render_text() or "disconnected" in sb.render_text()
