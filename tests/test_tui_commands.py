from __future__ import annotations

import pytest

from spare_paw.tui.commands import CommandResult, SlashCommandRouter  # noqa: F401


class _FakeAppState:
    def __init__(self):
        self.config = type("C", (), {"get": lambda self, k, d=None: "model-x"})()


@pytest.mark.asyncio
async def test_help_returns_help_text():
    router = SlashCommandRouter(app_state=None)
    res = await router.dispatch("/help")
    assert res.kind == "text"
    assert "Commands:" in res.content


@pytest.mark.asyncio
async def test_exit_returns_quit():
    router = SlashCommandRouter(app_state=None)
    res = await router.dispatch("/exit")
    assert res.kind == "quit"


@pytest.mark.asyncio
async def test_unknown_slash_returns_hint():
    router = SlashCommandRouter(app_state=None)
    res = await router.dispatch("/nonexistent")
    assert res.kind == "text"
    assert "Unknown command" in res.content


@pytest.mark.asyncio
async def test_plain_text_returns_send():
    router = SlashCommandRouter(app_state=None)
    res = await router.dispatch("hello world")
    assert res.kind == "send"
    assert res.text == "hello world"
    assert res.plan is False


@pytest.mark.asyncio
async def test_plan_prefix_sets_plan_flag():
    router = SlashCommandRouter(app_state=None)
    res = await router.dispatch("/plan do a thing")
    assert res.kind == "send"
    assert res.text == "do a thing"
    assert res.plan is True


@pytest.mark.asyncio
async def test_plan_without_body_returns_hint():
    router = SlashCommandRouter(app_state=None)
    res = await router.dispatch("/plan")
    assert res.kind == "text"
    assert "Usage:" in res.content


@pytest.mark.asyncio
async def test_image_missing_file_returns_error():
    router = SlashCommandRouter(app_state=None)
    res = await router.dispatch("/image /does/not/exist.png")
    assert res.kind == "text"
    assert "not found" in res.content.lower()
