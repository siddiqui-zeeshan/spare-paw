"""Branch coverage for engine voice delivery after run_tool_loop returns."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio

from spare_paw import context as ctx
from spare_paw.backend import IncomingMessage
from spare_paw.core import engine
from spare_paw.db import close_db, init_db


@pytest_asyncio.fixture(autouse=True)
async def _fresh_db(tmp_path, monkeypatch):
    db_file = tmp_path / "spare-paw.db"
    monkeypatch.setattr("spare_paw.db.DB_PATH", db_file)
    monkeypatch.setattr("spare_paw.db.DB_DIR", tmp_path)
    monkeypatch.setattr("spare_paw.db._connection", None)
    await init_db()
    yield
    await close_db()


def _app_state(tts_enabled=True, tts_voice="nova", tts_max_chars=2000):
    cfg = MagicMock()
    values = {
        "voice.tts_enabled": tts_enabled,
        "voice.tts_voice": tts_voice,
        "voice.tts_max_chars": tts_max_chars,
        "agent.system_prompt": "base",
        "agent.max_tool_iterations": 1,
        "context.max_messages": 64,
        "context.token_budget": 120000,
        "context.safety_margin": 0.85,
        "models.main_agent": "test/model",
        "models.summary": "test/model",
        "models.vision": "test/model",
    }
    cfg.get = lambda k, default=None: values.get(k, default)
    cfg.data = values
    registry = MagicMock()
    registry.get_schemas = lambda: []
    return SimpleNamespace(
        config=cfg,
        router_client=MagicMock(),
        tool_registry=registry,
        executor=None,
        current_request=None,
    )


def _mock_backend():
    backend = MagicMock()
    backend.send_text = AsyncMock()
    backend.send_voice = AsyncMock()
    backend.send_typing = AsyncMock()
    return backend


@pytest.fixture
def patch_tool_loop(monkeypatch):
    async def _fake_tool_loop(**kwargs):
        return "fake response text"
    monkeypatch.setattr(engine, "run_tool_loop", _fake_tool_loop)


@pytest.fixture
def patch_compact(monkeypatch):
    async def _noop(*a, **kw):
        return None
    monkeypatch.setattr(engine, "compact_with_retry", _noop)


@pytest.mark.asyncio
async def test_text_in_talk_mode_off_sends_text(patch_tool_loop, patch_compact, monkeypatch):
    monkeypatch.setattr(
        engine, "render_voice_note", AsyncMock(side_effect=AssertionError("should not be called")),
    )
    backend = _mock_backend()
    msg = IncomingMessage(text="hi")
    await engine.process_message(_app_state(), msg, backend)
    backend.send_text.assert_awaited_once_with("fake response text")
    backend.send_voice.assert_not_awaited()


@pytest.mark.asyncio
async def test_voice_in_talk_mode_off_sends_voice(patch_tool_loop, patch_compact, monkeypatch):
    monkeypatch.setattr(
        "spare_paw.core.voice.transcribe", AsyncMock(return_value="hello"),
    )
    monkeypatch.setattr(
        engine, "render_voice_note", AsyncMock(return_value=b"ogg-bytes"),
    )
    backend = _mock_backend()
    msg = IncomingMessage(voice_bytes=b"any")
    await engine.process_message(_app_state(), msg, backend)
    backend.send_voice.assert_awaited_once_with(b"ogg-bytes")
    backend.send_text.assert_not_awaited()


@pytest.mark.asyncio
async def test_text_in_talk_mode_on_sends_voice(patch_tool_loop, patch_compact, monkeypatch):
    monkeypatch.setattr(
        engine, "render_voice_note", AsyncMock(return_value=b"ogg-bytes"),
    )
    backend = _mock_backend()
    convo = await ctx.new_conversation()
    await ctx.set_conversation_meta(convo, "talk_mode", True)
    msg = IncomingMessage(text="hi")
    await engine.process_message(_app_state(), msg, backend)
    backend.send_voice.assert_awaited_once_with(b"ogg-bytes")
    backend.send_text.assert_not_awaited()


@pytest.mark.asyncio
async def test_reply_over_max_chars_talk_on_sends_text_with_banner(
    patch_tool_loop, patch_compact, monkeypatch,
):
    async def long_loop(**kwargs):
        return "x" * 5000
    monkeypatch.setattr(engine, "run_tool_loop", long_loop)
    monkeypatch.setattr(
        engine, "render_voice_note", AsyncMock(side_effect=AssertionError("should not be called")),
    )
    backend = _mock_backend()
    convo = await ctx.new_conversation()
    await ctx.set_conversation_meta(convo, "talk_mode", True)
    msg = IncomingMessage(text="hi")
    await engine.process_message(_app_state(tts_max_chars=2000), msg, backend)
    call_arg = backend.send_text.await_args.args[0]
    assert "too long for voice" in call_arg.lower()


@pytest.mark.asyncio
async def test_render_voice_note_failure_falls_back_to_text(
    patch_tool_loop, patch_compact, monkeypatch,
):
    from spare_paw.core.voice_out import VoiceRenderError
    monkeypatch.setattr(
        engine, "render_voice_note",
        AsyncMock(side_effect=VoiceRenderError("tts broken")),
    )
    backend = _mock_backend()
    convo = await ctx.new_conversation()
    await ctx.set_conversation_meta(convo, "talk_mode", True)
    msg = IncomingMessage(text="hi")
    await engine.process_message(_app_state(), msg, backend)
    # Two text sends: one-time failure notice + actual reply.
    assert backend.send_text.await_count == 2
    texts = [c.args[0] for c in backend.send_text.await_args_list]
    assert any("Voice" in t or "voice" in t for t in texts)
    assert "fake response text" in texts


@pytest.mark.asyncio
async def test_tts_disabled_talk_on_sends_text_only(
    patch_tool_loop, patch_compact, monkeypatch,
):
    monkeypatch.setattr(
        engine, "render_voice_note", AsyncMock(side_effect=AssertionError("should not be called")),
    )
    backend = _mock_backend()
    convo = await ctx.new_conversation()
    await ctx.set_conversation_meta(convo, "talk_mode", True)
    msg = IncomingMessage(text="hi")
    await engine.process_message(_app_state(tts_enabled=False), msg, backend)
    backend.send_text.assert_awaited_once()
    backend.send_voice.assert_not_awaited()
