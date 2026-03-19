"""Tests for core/engine.py — platform-agnostic message processor."""

from __future__ import annotations

import ast
import inspect
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from spare_paw.backend import IncomingMessage
from spare_paw.core.engine import process_agent_callback, process_message, split_text


class TestSplitText:
    def test_short_text_single_chunk(self):
        assert split_text("hello", 100) == ["hello"]

    def test_exact_length(self):
        text = "a" * 50
        assert split_text(text, 50) == [text]

    def test_splits_at_newline(self):
        text = "line1\nline2\nline3"
        chunks = split_text(text, 12)
        assert all(len(c) <= 12 for c in chunks)
        assert "line1" in chunks[0]

    def test_hard_cut_when_no_newline(self):
        text = "a" * 100
        chunks = split_text(text, 30)
        assert all(len(c) <= 30 for c in chunks)
        assert "".join(chunks) == text

    def test_preserves_all_content(self):
        text = "first\nsecond\nthird\nfourth"
        chunks = split_text(text, 10)
        joined = "\n".join(chunks)
        assert "first" in joined
        assert "second" in joined
        assert "third" in joined
        assert "fourth" in joined

    def test_empty_string(self):
        assert split_text("", 100) == [""]

    def test_multiple_newlines_stripped(self):
        """Leading newlines on subsequent chunks should be stripped."""
        text = "aaa\n\n\nbbb"
        chunks = split_text(text, 5)
        for chunk in chunks:
            assert not chunk.startswith("\n")


def _make_app_state(response_text="Bot response."):
    """Build a mock app_state with router_client, tool_registry, config, executor."""
    app_state = MagicMock()
    app_state.config.get = lambda key, default=None: {
        "models.default": "test-model",
        "agent.max_tool_iterations": 5,
        "agent.system_prompt": "You are a test bot.",
        "context.summary_model": "summary-model",
    }.get(key, default)
    app_state.tool_registry.get_schemas.return_value = []
    app_state.router_client = MagicMock()
    app_state.executor = None
    return app_state


def _make_backend():
    """Build a mock MessageBackend."""
    backend = AsyncMock()
    return backend


class TestProcessMessage:
    @pytest.mark.asyncio
    async def test_text_message(self):
        """Text-only message: ingest, assemble, tool loop, send_text."""
        app_state = _make_app_state()
        backend = _make_backend()
        msg = IncomingMessage(text="hello")

        with patch("spare_paw.core.engine.ctx_module") as mock_ctx, \
             patch("spare_paw.core.engine.run_tool_loop", new_callable=AsyncMock, return_value="Bot reply") as _mock_loop, \
             patch("spare_paw.core.engine.build_system_prompt", new_callable=AsyncMock, return_value="system prompt"), \
             patch("spare_paw.core.engine.compact_with_retry", new_callable=AsyncMock):
            mock_ctx.get_or_create_conversation = AsyncMock(return_value="conv-1")
            mock_ctx.ingest = AsyncMock(return_value="msg-1")
            mock_ctx.assemble = AsyncMock(return_value=[
                {"role": "system", "content": "system prompt"},
                {"role": "user", "content": "hello"},
            ])

            await process_message(app_state, msg, backend)

        # Verify user message ingested
        mock_ctx.ingest.assert_any_await("conv-1", "user", "hello")
        # Verify assistant response ingested
        mock_ctx.ingest.assert_any_await("conv-1", "assistant", "Bot reply")
        # Verify backend received markdown (not HTML)
        backend.send_text.assert_awaited_once_with("Bot reply")

    @pytest.mark.asyncio
    async def test_voice_message(self):
        """Voice message: transcribe bytes, then proceed as text."""
        app_state = _make_app_state()
        backend = _make_backend()
        msg = IncomingMessage(voice_bytes=b"\x00\x01\x02")

        with patch("spare_paw.core.engine.ctx_module") as mock_ctx, \
             patch("spare_paw.core.engine.run_tool_loop", new_callable=AsyncMock, return_value="Voice reply"), \
             patch("spare_paw.core.engine.build_system_prompt", new_callable=AsyncMock, return_value="sys"), \
             patch("spare_paw.core.engine.transcribe", new_callable=AsyncMock, return_value="transcribed text"), \
             patch("spare_paw.core.engine.compact_with_retry", new_callable=AsyncMock):
            mock_ctx.get_or_create_conversation = AsyncMock(return_value="conv-1")
            mock_ctx.ingest = AsyncMock(return_value="msg-1")
            mock_ctx.assemble = AsyncMock(return_value=[
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "transcribed text"},
            ])

            await process_message(app_state, msg, backend)

        mock_ctx.ingest.assert_any_await("conv-1", "user", "transcribed text")
        backend.send_text.assert_awaited_once_with("Voice reply")

    @pytest.mark.asyncio
    async def test_image_message(self):
        """Image message: base64 encode, build multimodal content."""
        app_state = _make_app_state()
        backend = _make_backend()
        msg = IncomingMessage(image_bytes=b"\xff\xd8", caption="what is this?")

        assembled = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "what is this?"},
        ]

        with patch("spare_paw.core.engine.ctx_module") as mock_ctx, \
             patch("spare_paw.core.engine.run_tool_loop", new_callable=AsyncMock, return_value="It's a photo") as _mock_loop, \
             patch("spare_paw.core.engine.build_system_prompt", new_callable=AsyncMock, return_value="sys"), \
             patch("spare_paw.core.engine.compact_with_retry", new_callable=AsyncMock):
            mock_ctx.get_or_create_conversation = AsyncMock(return_value="conv-1")
            mock_ctx.ingest = AsyncMock(return_value="msg-1")
            mock_ctx.assemble = AsyncMock(return_value=assembled)

            await process_message(app_state, msg, backend)

        # Check that the last user message was made multimodal
        last_user = assembled[-1]
        assert isinstance(last_user["content"], list)
        assert last_user["content"][0]["type"] == "text"
        assert last_user["content"][1]["type"] == "image_url"
        assert "base64" in last_user["content"][1]["image_url"]["url"]

    @pytest.mark.asyncio
    async def test_cron_context_injected(self):
        """Cron context is appended to assembled messages."""
        app_state = _make_app_state()
        backend = _make_backend()
        msg = IncomingMessage(text="looks good", cron_context="cron output here")

        assembled = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "looks good"},
        ]

        with patch("spare_paw.core.engine.ctx_module") as mock_ctx, \
             patch("spare_paw.core.engine.run_tool_loop", new_callable=AsyncMock, return_value="Noted") as _mock_loop, \
             patch("spare_paw.core.engine.build_system_prompt", new_callable=AsyncMock, return_value="sys"), \
             patch("spare_paw.core.engine.compact_with_retry", new_callable=AsyncMock):
            mock_ctx.get_or_create_conversation = AsyncMock(return_value="conv-1")
            mock_ctx.ingest = AsyncMock(return_value="msg-1")
            mock_ctx.assemble = AsyncMock(return_value=assembled)

            await process_message(app_state, msg, backend)

        # Check cron context was injected
        cron_msgs = [m for m in assembled if "cron" in m.get("content", "").lower()]
        assert len(cron_msgs) >= 1
        assert "cron output here" in cron_msgs[0]["content"]

    @pytest.mark.asyncio
    async def test_empty_text_returns_early(self):
        """Message with no text/voice/image should return without calling backend."""
        app_state = _make_app_state()
        backend = _make_backend()
        msg = IncomingMessage()

        await process_message(app_state, msg, backend)

        backend.send_text.assert_not_awaited()


class TestProcessAgentCallback:
    @pytest.mark.asyncio
    async def test_ingests_and_sends(self):
        """Agent callback: ingest augmented text, run tool loop, send result."""
        app_state = _make_app_state()
        backend = _make_backend()

        with patch("spare_paw.core.engine.ctx_module") as mock_ctx, \
             patch("spare_paw.core.engine.run_tool_loop", new_callable=AsyncMock, return_value="Synthesized response"), \
             patch("spare_paw.core.engine.build_system_prompt", new_callable=AsyncMock, return_value="sys"), \
             patch("spare_paw.core.engine.compact_with_retry", new_callable=AsyncMock):
            mock_ctx.get_or_create_conversation = AsyncMock(return_value="conv-1")
            mock_ctx.ingest = AsyncMock(return_value="msg-1")
            mock_ctx.assemble = AsyncMock(return_value=[
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "agent results"},
            ])

            await process_agent_callback(app_state, "agent results", backend)

        # Verify augmented text was ingested
        ingest_calls = mock_ctx.ingest.await_args_list
        user_calls = [c for c in ingest_calls if c.args[1] == "user"]
        assert len(user_calls) >= 1
        assert "agent results" in user_calls[0].args[2]

        # Verify response ingested
        assistant_calls = [c for c in ingest_calls if c.args[1] == "assistant"]
        assert len(assistant_calls) >= 1

        # Verify backend received the synthesized response
        backend.send_text.assert_awaited_once_with("Synthesized response")

    @pytest.mark.asyncio
    async def test_error_sends_fallback(self):
        """On failure, sends error message via backend."""
        app_state = _make_app_state()
        backend = _make_backend()

        with patch("spare_paw.core.engine.ctx_module") as mock_ctx, \
             patch("spare_paw.core.engine.run_tool_loop", new_callable=AsyncMock, side_effect=Exception("boom")), \
             patch("spare_paw.core.engine.build_system_prompt", new_callable=AsyncMock, return_value="sys"):
            mock_ctx.get_or_create_conversation = AsyncMock(return_value="conv-1")
            mock_ctx.ingest = AsyncMock(return_value="msg-1")
            mock_ctx.assemble = AsyncMock(return_value=[])

            await process_agent_callback(app_state, "agent results", backend)

        # Should have sent an error message
        backend.send_text.assert_awaited_once()
        sent_text = backend.send_text.await_args.args[0]
        assert "failed" in sent_text.lower() or "error" in sent_text.lower()


class TestNoTelegramImport:
    def test_no_telegram_import_in_engine(self):
        import spare_paw.core.engine as mod

        source = inspect.getsource(mod)
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert not alias.name.startswith("telegram"), (
                        f"Found 'import {alias.name}' in core/engine.py"
                    )
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    assert not node.module.startswith("telegram"), (
                        f"Found 'from {node.module}' in core/engine.py"
                    )
