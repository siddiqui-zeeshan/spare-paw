from __future__ import annotations

import pytest

from spare_paw.tui.app import SparePawTUI
from spare_paw.tui.events import ConnectionStateChanged
from spare_paw.tui.widgets.status_bar import StatusBar


@pytest.mark.asyncio
async def test_connection_changed_updates_status_bar():
    app = SparePawTUI(client=None, app_state=None)
    async with app.run_test() as pilot:
        app.post_message(ConnectionStateChanged(state="reconnecting", detail="test"))
        await pilot.pause(0.05)
        bar = app.query_one(StatusBar)
        rendered = bar.render_text()
        assert "reconnecting" in rendered or "yellow" in rendered
