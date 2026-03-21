"""Tests for core/planner.py — deep thinking planning phase."""

from __future__ import annotations

import ast
import inspect
from unittest.mock import AsyncMock, MagicMock

import pytest

from spare_paw.core.planner import PLANNING_SYSTEM_PROMPT, create_plan


def _make_router_client(plan_text: str = "## Plan\n1. Search the web"):
    """Build a mock router client that returns a plan response."""
    client = AsyncMock()
    client.chat = AsyncMock(return_value={
        "choices": [{"message": {"content": plan_text}}],
    })
    return client


def _make_config(overrides: dict | None = None):
    store = {
        "models.main_agent": "test-model",
        "models.planner": "cheap-model",
        **(overrides or {}),
    }
    config = MagicMock()
    config.get = lambda key, default=None: store.get(key, default)
    return config


class TestCreatePlan:
    @pytest.mark.asyncio
    async def test_returns_plan_text(self):
        """create_plan returns the LLM's plan as a string."""
        client = _make_router_client("## Plan\n1. Do X\n2. Do Y")
        config = _make_config()
        messages = [
            {"role": "system", "content": "You are a bot."},
            {"role": "user", "content": "research AI and summarize"},
        ]

        result = await create_plan(messages, config, client)

        assert "Do X" in result
        assert "Do Y" in result

    @pytest.mark.asyncio
    async def test_uses_planning_model(self):
        """create_plan uses the planning.model config key."""
        client = _make_router_client()
        config = _make_config({"models.planner": "google/gemini-2.0-flash"})
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "do something complex"},
        ]

        await create_plan(messages, config, client)

        call_args = client.chat.call_args
        assert call_args[0][1] == "google/gemini-2.0-flash"

    @pytest.mark.asyncio
    async def test_falls_back_to_default_model(self):
        """When models.planner is not set, falls back to models.main_agent."""
        client = _make_router_client()
        config = _make_config({"models.planner": None})
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "do something"},
        ]

        await create_plan(messages, config, client)

        call_args = client.chat.call_args
        assert call_args[0][1] == "test-model"

    @pytest.mark.asyncio
    async def test_no_tools_passed_to_llm(self):
        """Planning call should NOT include tool schemas."""
        client = _make_router_client()
        config = _make_config()
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "plan this"},
        ]

        await create_plan(messages, config, client)

        call_args = client.chat.call_args
        # chat(messages, model) — no tools arg
        assert len(call_args[0]) == 2 or call_args[0][2] is None

    @pytest.mark.asyncio
    async def test_planning_prompt_is_system_message(self):
        """The planning system prompt should replace the original system message."""
        client = _make_router_client()
        config = _make_config()
        messages = [
            {"role": "system", "content": "original system prompt"},
            {"role": "user", "content": "complex task"},
        ]

        await create_plan(messages, config, client)

        sent_messages = client.chat.call_args[0][0]
        system_msgs = [m for m in sent_messages if m["role"] == "system"]
        assert len(system_msgs) == 1
        assert PLANNING_SYSTEM_PROMPT in system_msgs[0]["content"]

    @pytest.mark.asyncio
    async def test_includes_user_messages(self):
        """The planning call should include user messages for context."""
        client = _make_router_client()
        config = _make_config()
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "earlier context"},
            {"role": "assistant", "content": "some reply"},
            {"role": "user", "content": "now do this complex thing"},
        ]

        await create_plan(messages, config, client)

        sent_messages = client.chat.call_args[0][0]
        user_msgs = [m for m in sent_messages if m["role"] == "user"]
        assert any("complex thing" in m["content"] for m in user_msgs)

    @pytest.mark.asyncio
    async def test_returns_empty_on_error(self):
        """On LLM error, create_plan returns an empty string (graceful degradation)."""
        client = AsyncMock()
        client.chat = AsyncMock(side_effect=Exception("API error"))
        config = _make_config()
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "do stuff"},
        ]

        result = await create_plan(messages, config, client)

        assert result == ""


class TestNoTelegramImport:
    def test_no_telegram_import_in_planner(self):
        import spare_paw.core.planner as mod

        source = inspect.getsource(mod)
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert not alias.name.startswith("telegram")
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    assert not node.module.startswith("telegram")
