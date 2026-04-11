"""Tests for router/tool_loop.py — timeouts, circuit breaker, structured logging."""

from __future__ import annotations

import asyncio
import json
import logging
from unittest.mock import AsyncMock, MagicMock

import pytest

from spare_paw.router.tool_loop import run_tool_loop


def _make_response(content: str = "", tool_calls: list | None = None, usage: dict | None = None):
    """Helper to build an OpenRouter-style response."""
    msg: dict = {"content": content}
    if tool_calls:
        msg["tool_calls"] = tool_calls
    return {
        "choices": [{"message": msg, "finish_reason": "stop"}],
        "usage": usage or {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


def _make_tool_call(name: str, args: dict, call_id: str = "call_1"):
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(args)},
    }


class TestToolExecutionTimeout:
    @pytest.mark.asyncio
    async def test_tool_timeout_returns_error_to_model(self):
        """A tool that exceeds the timeout gets an error fed back to the model."""
        client = AsyncMock()
        # First call: model requests a tool call
        client.chat = AsyncMock(side_effect=[
            _make_response(tool_calls=[_make_tool_call("slow_tool", {})]),
            _make_response(content="Done after timeout"),
        ])

        registry = AsyncMock()

        async def slow_tool(*args, **kwargs):
            await asyncio.sleep(999)

        registry.execute = slow_tool
        registry.get_schemas = MagicMock(return_value=[])

        result = await run_tool_loop(
            client=client,
            messages=[{"role": "user", "content": "test"}],
            model="test-model",
            tools=[{"type": "function", "function": {"name": "slow_tool"}}],
            tool_registry=registry,
            tool_timeout=0.1,
        )

        assert result == "Done after timeout"
        # Check the tool result message contains timeout error
        tool_msg = [m for m in client.chat.call_args_list[1][0][0] if m.get("role") == "tool"]
        assert len(tool_msg) == 1
        assert "timed out" in tool_msg[0]["content"].lower()

    @pytest.mark.asyncio
    async def test_llm_call_timeout(self):
        """LLM call that exceeds timeout raises and is handled."""
        client = AsyncMock()

        async def slow_chat(*args, **kwargs):
            await asyncio.sleep(999)

        client.chat = slow_chat

        registry = AsyncMock()
        registry.get_schemas = MagicMock(return_value=[])

        result = await run_tool_loop(
            client=client,
            messages=[{"role": "user", "content": "test"}],
            model="test-model",
            tools=[],
            tool_registry=registry,
            llm_timeout=0.1,
        )

        assert "timed out" in result.lower() or "timeout" in result.lower()


class TestTokenBudgetCircuitBreaker:
    @pytest.mark.asyncio
    async def test_aborts_when_budget_exceeded(self):
        """Loop aborts when cumulative tokens exceed the budget."""
        client = AsyncMock()
        # Each call uses 20k tokens; budget is 30k, so should abort after 2 iterations
        client.chat = AsyncMock(side_effect=[
            _make_response(
                tool_calls=[_make_tool_call("shell", {"command": "echo 1"})],
                usage={"prompt_tokens": 15000, "completion_tokens": 5000, "total_tokens": 20000},
            ),
            _make_response(
                tool_calls=[_make_tool_call("shell", {"command": "echo 2"})],
                usage={"prompt_tokens": 15000, "completion_tokens": 5000, "total_tokens": 20000},
            ),
        ])

        registry = AsyncMock()
        registry.execute = AsyncMock(return_value="ok")

        result = await run_tool_loop(
            client=client,
            messages=[{"role": "user", "content": "test"}],
            model="test-model",
            tools=[{"type": "function", "function": {"name": "shell"}}],
            tool_registry=registry,
            token_budget=30000,
        )

        assert "token budget" in result.lower() or "token limit" in result.lower()
        # Should have stopped after 2 iterations, not continued
        assert client.chat.call_count == 2

    @pytest.mark.asyncio
    async def test_no_abort_under_budget(self):
        """Loop completes normally when under token budget."""
        client = AsyncMock()
        client.chat = AsyncMock(side_effect=[
            _make_response(
                tool_calls=[_make_tool_call("shell", {"command": "echo 1"})],
                usage={"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
            ),
            _make_response(content="Final answer", usage={"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}),
        ])

        registry = AsyncMock()
        registry.execute = AsyncMock(return_value="ok")

        result = await run_tool_loop(
            client=client,
            messages=[{"role": "user", "content": "test"}],
            model="test-model",
            tools=[{"type": "function", "function": {"name": "shell"}}],
            tool_registry=registry,
            token_budget=50000,
        )

        assert result == "Final answer"


class TestMissingChoices:
    @pytest.mark.asyncio
    async def test_missing_choices_returns_error(self):
        """LLM response without 'choices' key returns error instead of crashing."""
        client = AsyncMock()
        client.chat = AsyncMock(return_value={"usage": {"prompt_tokens": 10, "completion_tokens": 0, "total_tokens": 10}})

        registry = AsyncMock()

        result = await run_tool_loop(
            client=client,
            messages=[{"role": "user", "content": "test"}],
            model="test-model",
            tools=[],
            tool_registry=registry,
        )

        assert "invalid response" in result.lower()

    @pytest.mark.asyncio
    async def test_empty_choices_returns_error(self):
        """LLM response with empty choices list returns error instead of crashing."""
        client = AsyncMock()
        client.chat = AsyncMock(return_value={"choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 0, "total_tokens": 10}})

        registry = AsyncMock()

        result = await run_tool_loop(
            client=client,
            messages=[{"role": "user", "content": "test"}],
            model="test-model",
            tools=[],
            tool_registry=registry,
        )

        assert "invalid response" in result.lower()


class TestStructuredLogging:
    @pytest.mark.asyncio
    async def test_logs_contain_structured_fields(self, caplog):
        """Log entries include iteration, tool_name, and duration_ms."""
        client = AsyncMock()
        client.chat = AsyncMock(side_effect=[
            _make_response(
                tool_calls=[_make_tool_call("shell", {"command": "echo hi"})],
                usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            ),
            _make_response(content="Done"),
        ])

        registry = AsyncMock()
        registry.execute = AsyncMock(return_value="hi")

        with caplog.at_level(logging.INFO, logger="spare_paw.router.tool_loop"):
            await run_tool_loop(
                client=client,
                messages=[{"role": "user", "content": "test"}],
                model="test-model",
                tools=[{"type": "function", "function": {"name": "shell"}}],
                tool_registry=registry,
            )

        tool_log = [r for r in caplog.records if "tool shell" in r.message.lower() or "shell executed" in r.message.lower()]
        assert len(tool_log) > 0
        # Check that duration info is in the log
        log_text = " ".join(r.message for r in caplog.records)
        assert "ms" in log_text or "duration" in log_text.lower()

    @pytest.mark.asyncio
    async def test_logs_token_budget_abort(self, caplog):
        """Token budget abort is logged with warning level."""
        client = AsyncMock()
        client.chat = AsyncMock(return_value=_make_response(
            tool_calls=[_make_tool_call("shell", {"command": "echo"})],
            usage={"prompt_tokens": 30000, "completion_tokens": 10000, "total_tokens": 40000},
        ))

        registry = AsyncMock()
        registry.execute = AsyncMock(return_value="ok")

        with caplog.at_level(logging.WARNING, logger="spare_paw.router.tool_loop"):
            await run_tool_loop(
                client=client,
                messages=[{"role": "user", "content": "test"}],
                model="test-model",
                tools=[{"type": "function", "function": {"name": "shell"}}],
                tool_registry=registry,
                token_budget=30000,
            )

        warning_logs = [r for r in caplog.records if r.levelno >= logging.WARNING]
        budget_warnings = [r for r in warning_logs if "token" in r.message.lower() and "budget" in r.message.lower()]
        assert len(budget_warnings) > 0
