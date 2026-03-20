"""Tests for ToolEvent and on_event/on_token callbacks in the tool loop."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from spare_paw.router.tool_loop import ToolEvent, run_tool_loop


def _text_response(content: str = "done") -> dict:
    return {
        "choices": [{"message": {"role": "assistant", "content": content}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


def _tool_call_response(name: str, arguments: dict, call_id: str = "call_1") -> dict:
    return {
        "choices": [{
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": json.dumps(arguments),
                    },
                }],
            }
        }],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


class TestToolEvent:
    def test_dataclass_fields(self):
        event = ToolEvent(kind="tool_start", tool_name="shell", iteration=1)
        assert event.kind == "tool_start"
        assert event.tool_name == "shell"
        assert event.tool_args is None
        assert event.result_preview is None
        assert event.iteration == 1

    def test_defaults(self):
        event = ToolEvent(kind="llm_start")
        assert event.tool_name is None
        assert event.iteration == 0


class TestOnEventCallback:
    @pytest.mark.asyncio
    async def test_fires_llm_start_and_end_events(self):
        """on_event receives llm_start and llm_end when no tool calls."""
        events: list[ToolEvent] = []
        on_event = MagicMock(side_effect=lambda e: events.append(e))

        mock_client = AsyncMock()
        mock_client.chat = AsyncMock(return_value=_text_response("hi"))

        await run_tool_loop(
            client=mock_client,
            messages=[{"role": "user", "content": "hello"}],
            model="m",
            tools=[],
            tool_registry=AsyncMock(),
            on_event=on_event,
        )

        assert len(events) == 2
        assert events[0].kind == "llm_start"
        assert events[1].kind == "llm_end"

    @pytest.mark.asyncio
    async def test_fires_tool_start_and_end_events(self):
        """on_event receives tool_start/tool_end around tool execution."""
        events: list[ToolEvent] = []
        on_event = MagicMock(side_effect=lambda e: events.append(e))

        mock_client = AsyncMock()
        mock_client.chat = AsyncMock(side_effect=[
            _tool_call_response("shell", {"command": "ls"}),
            _text_response("done"),
        ])
        mock_registry = AsyncMock()
        mock_registry.execute = AsyncMock(return_value="file_list")

        await run_tool_loop(
            client=mock_client,
            messages=[{"role": "user", "content": "list"}],
            model="m",
            tools=[{"type": "function", "function": {"name": "shell"}}],
            tool_registry=mock_registry,
            on_event=on_event,
        )

        kinds = [e.kind for e in events]
        assert "tool_start" in kinds
        assert "tool_end" in kinds
        tool_start = next(e for e in events if e.kind == "tool_start")
        assert tool_start.tool_name == "shell"
        assert tool_start.tool_args == {"command": "ls"}

    @pytest.mark.asyncio
    async def test_no_event_callback_is_safe(self):
        """on_event=None should not cause errors."""
        mock_client = AsyncMock()
        mock_client.chat = AsyncMock(return_value=_text_response("hi"))

        result = await run_tool_loop(
            client=mock_client,
            messages=[],
            model="m",
            tools=[],
            tool_registry=AsyncMock(),
            on_event=None,
        )
        assert result == "hi"


class TestOnTokenCallback:
    @pytest.mark.asyncio
    async def test_fires_tokens_for_final_text(self):
        """on_token receives word-chunked tokens of the final response."""
        tokens: list[str] = []
        on_token = MagicMock(side_effect=lambda t: tokens.append(t))

        mock_client = AsyncMock()
        mock_client.chat = AsyncMock(return_value=_text_response("hello world"))

        result = await run_tool_loop(
            client=mock_client,
            messages=[],
            model="m",
            tools=[],
            tool_registry=AsyncMock(),
            on_token=on_token,
        )

        assert result == "hello world"
        assert tokens == ["hello ", "world "]

    @pytest.mark.asyncio
    async def test_no_token_callback_is_safe(self):
        """on_token=None should not cause errors."""
        mock_client = AsyncMock()
        mock_client.chat = AsyncMock(return_value=_text_response("hi"))

        result = await run_tool_loop(
            client=mock_client,
            messages=[],
            model="m",
            tools=[],
            tool_registry=AsyncMock(),
            on_token=None,
        )
        assert result == "hi"
