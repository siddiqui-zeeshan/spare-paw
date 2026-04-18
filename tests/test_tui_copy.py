import pytest

from spare_paw.tui.app import SparePawTUI
from spare_paw.tui.widgets.chat_log import ChatLog
from spare_paw.tui.widgets.message_view import MessageView


@pytest.mark.asyncio
async def test_copy_last_message_returns_text(monkeypatch):
    copied: list[str] = []

    def fake_copy(text: str) -> None:
        copied.append(text)

    app = SparePawTUI(client=None, app_state=None)
    monkeypatch.setattr("spare_paw.tui.app._copy_to_clipboard", fake_copy)
    async with app.run_test() as _pilot:
        log = app.query_one(ChatLog)
        log.mount_turn(MessageView(role="assistant", initial_text="final answer", historical=True))
        app.action_copy_last()
        assert copied == ["final answer"]
