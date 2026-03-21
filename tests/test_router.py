"""Tests for OpenRouter client and tool loop."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from spare_paw.router.openrouter import (
    OPENROUTER_URL,
    OpenRouterClient,
    OpenRouterError,
)
from spare_paw.router.tool_loop import run_tool_loop


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_response_ctx(status: int, json_data: dict | None = None, text: str = ""):
    """Create a mock async context manager that mimics aiohttp response."""
    resp = AsyncMock()
    resp.status = status
    resp.text = AsyncMock(return_value=text)
    if json_data is not None:
        resp.json = AsyncMock(return_value=json_data)

    ctx = AsyncMock()
    ctx.__aenter__ = AsyncMock(return_value=resp)
    ctx.__aexit__ = AsyncMock(return_value=False)
    return ctx


def _success_json(content: str = "Hello!") -> dict:
    return {
        "choices": [
            {"message": {"role": "assistant", "content": content}}
        ]
    }


# ---------------------------------------------------------------------------
# OpenRouterClient.chat
# ---------------------------------------------------------------------------

class TestOpenRouterClientChat:
    """Tests for OpenRouterClient.chat()."""

    @pytest.mark.asyncio
    async def test_chat_makes_correct_api_call(self):
        """chat() sends the right payload and returns parsed JSON."""
        expected = _success_json("Hi there")
        mock_session = MagicMock()
        mock_session.closed = False
        mock_session.post = MagicMock(
            return_value=_make_response_ctx(200, json_data=expected)
        )

        client = OpenRouterClient(api_key="test-key", semaphore=asyncio.Semaphore(1))
        client._session = mock_session

        messages = [{"role": "user", "content": "ping"}]
        result = await client.chat(messages, model="test/model")

        # Verify the POST was called with the right URL and body
        mock_session.post.assert_called_once()
        call_args = mock_session.post.call_args
        assert call_args[0][0] == OPENROUTER_URL
        body = call_args[1]["json"]
        assert body["model"] == "test/model"
        assert body["messages"] == messages
        assert "tools" not in body

        assert result == expected

    @pytest.mark.asyncio
    async def test_chat_sends_tools_when_provided(self):
        """chat() includes tools and tool_choice in the body when tools are given."""
        mock_session = MagicMock()
        mock_session.closed = False
        mock_session.post = MagicMock(
            return_value=_make_response_ctx(200, json_data=_success_json())
        )

        client = OpenRouterClient(api_key="k", semaphore=asyncio.Semaphore(1))
        client._session = mock_session

        tools = [{"type": "function", "function": {"name": "foo"}}]
        await client.chat([], model="m", tools=tools)

        body = mock_session.post.call_args[1]["json"]
        assert body["tools"] == tools
        assert body["tool_choice"] == "auto"


# ---------------------------------------------------------------------------
# Retry logic
# ---------------------------------------------------------------------------

class TestRetryLogic:
    """Tests for exponential backoff and retry behaviour."""

    @pytest.mark.asyncio
    async def test_retries_on_429_then_succeeds(self):
        """A 429 on the first attempt should be retried; success on second attempt."""
        success_data = _success_json("recovered")
        mock_session = MagicMock()
        mock_session.closed = False
        mock_session.post = MagicMock(
            side_effect=[
                _make_response_ctx(429, text="rate limited"),
                _make_response_ctx(200, json_data=success_data),
            ]
        )

        client = OpenRouterClient(api_key="k", semaphore=asyncio.Semaphore(1))
        client._session = mock_session

        with patch("spare_paw.router.openrouter.asyncio.sleep", new_callable=AsyncMock):
            result = await client.chat([], model="m")

        assert result == success_data
        assert mock_session.post.call_count == 2

    @pytest.mark.asyncio
    async def test_non_retryable_401_raises_immediately(self):
        """A 401 should raise OpenRouterError without any retry."""
        mock_session = MagicMock()
        mock_session.closed = False
        mock_session.post = MagicMock(
            return_value=_make_response_ctx(401, text="unauthorized")
        )

        client = OpenRouterClient(api_key="bad", semaphore=asyncio.Semaphore(1))
        client._session = mock_session

        with pytest.raises(OpenRouterError) as exc_info:
            await client.chat([], model="m")

        assert exc_info.value.status == 401
        assert "unauthorized" in exc_info.value.message
        # Only one attempt — no retries
        assert mock_session.post.call_count == 1


# ---------------------------------------------------------------------------
# run_tool_loop
# ---------------------------------------------------------------------------

def _tool_call_response(name: str, arguments: dict, call_id: str = "call_1") -> dict:
    """Build a model response that contains a tool call."""
    return {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": name,
                                "arguments": json.dumps(arguments),
                            },
                        }
                    ],
                }
            }
        ]
    }


def _text_response(content: str = "done") -> dict:
    return {"choices": [{"message": {"role": "assistant", "content": content}}]}


class TestRunToolLoop:
    """Tests for the tool-calling execution loop."""

    @pytest.mark.asyncio
    async def test_executes_tool_and_returns_final_text(self):
        """Loop calls a tool, feeds the result back, then returns final text."""
        mock_client = AsyncMock()
        mock_client.chat = AsyncMock(
            side_effect=[
                _tool_call_response("greet", {"name": "Alice"}),
                _text_response("Hello Alice!"),
            ]
        )

        mock_registry = AsyncMock()
        mock_registry.execute = AsyncMock(return_value="greeting sent")

        messages: list[dict] = [{"role": "user", "content": "greet Alice"}]
        tools = [{"type": "function", "function": {"name": "greet"}}]

        result = await run_tool_loop(
            client=mock_client,
            messages=messages,
            model="m",
            tools=tools,
            tool_registry=mock_registry,
        )

        assert result == "Hello Alice!"
        # Tool was executed once
        mock_registry.execute.assert_called_once_with("greet", {"name": "Alice"}, None)
        # Client was called twice (tool call round + final text round)
        assert mock_client.chat.call_count == 2

    @pytest.mark.asyncio
    async def test_max_iterations_returns_fallback(self):
        """When the model keeps calling tools, the loop caps at max_iterations."""
        mock_client = AsyncMock()
        # Always return a tool call — never a plain text response
        mock_client.chat = AsyncMock(
            side_effect=[
                _tool_call_response("tick", {}, call_id=f"c{i}")
                for i in range(5)
            ]
            # After exhausting iterations, the loop makes one final call without tools
            + [_text_response("gave up")]
        )

        mock_registry = AsyncMock()
        mock_registry.execute = AsyncMock(return_value="ok")

        result = await run_tool_loop(
            client=mock_client,
            messages=[{"role": "user", "content": "loop"}],
            model="m",
            tools=[{"type": "function", "function": {"name": "tick"}}],
            tool_registry=mock_registry,
            max_iterations=5,
        )

        assert result == "gave up"
        # 5 iterations with tool calls + 1 final call = 6 total
        assert mock_client.chat.call_count == 6

    @pytest.mark.asyncio
    async def test_no_tool_calls_returns_immediately(self):
        """If the model responds with text on the first call, return immediately."""
        mock_client = AsyncMock()
        mock_client.chat = AsyncMock(return_value=_text_response("immediate"))

        result = await run_tool_loop(
            client=mock_client,
            messages=[],
            model="m",
            tools=[],
            tool_registry=AsyncMock(),
        )

        assert result == "immediate"
        assert mock_client.chat.call_count == 1

    @pytest.mark.asyncio
    async def test_tool_execution_error_is_fed_back(self):
        """If a tool raises, the error string is sent back to the model."""
        mock_client = AsyncMock()
        mock_client.chat = AsyncMock(
            side_effect=[
                _tool_call_response("fail_tool", {}),
                _text_response("handled error"),
            ]
        )

        mock_registry = AsyncMock()
        mock_registry.execute = AsyncMock(side_effect=RuntimeError("boom"))

        messages: list[dict] = [{"role": "user", "content": "try"}]

        result = await run_tool_loop(
            client=mock_client,
            messages=messages,
            model="m",
            tools=[{"type": "function", "function": {"name": "fail_tool"}}],
            tool_registry=mock_registry,
        )

        assert result == "handled error"
        # The tool result message should contain the error
        _ = messages[-1]  # last message before final call
        # Find the tool result message in the messages list
        tool_results = [m for m in messages if m.get("role") == "tool"]
        assert len(tool_results) == 1
        assert "Error executing tool fail_tool" in tool_results[0]["content"]
        assert "boom" in tool_results[0]["content"]

    @pytest.mark.asyncio
    async def test_spawn_agent_calls_in_same_batch_share_group_id(self):
        """Multiple spawn_agent calls in one model response get the same group_id."""
        # Model returns two spawn_agent calls in one response, then text
        batch_response = {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_a",
                            "type": "function",
                            "function": {
                                "name": "spawn_agent",
                                "arguments": json.dumps({"name": "r1", "prompt": "research"}),
                            },
                        },
                        {
                            "id": "call_b",
                            "type": "function",
                            "function": {
                                "name": "spawn_agent",
                                "arguments": json.dumps({"name": "r2", "prompt": "analyze"}),
                            },
                        },
                    ],
                }
            }]
        }

        mock_client = AsyncMock()
        mock_client.chat = AsyncMock(return_value=batch_response)

        # Capture the args passed to execute to verify group_id injection
        captured_args: list[dict] = []

        async def _capture_execute(name, arguments, executor=None):
            captured_args.append({"name": name, "args": arguments})
            return json.dumps({"__stop_turn__": True, "reply": "spawned"})

        mock_registry = AsyncMock()
        mock_registry.execute = AsyncMock(side_effect=_capture_execute)

        await run_tool_loop(
            client=mock_client,
            messages=[{"role": "user", "content": "research two topics"}],
            model="m",
            tools=[{"type": "function", "function": {"name": "spawn_agent"}}],
            tool_registry=mock_registry,
        )

        # Both spawn_agent calls should have been given a group_id
        spawn_calls = [c for c in captured_args if c["name"] == "spawn_agent"]
        assert len(spawn_calls) == 2
        assert "group_id" in spawn_calls[0]["args"], "spawn_agent should receive group_id"
        assert "group_id" in spawn_calls[1]["args"], "spawn_agent should receive group_id"
        # And they should share the SAME group_id
        assert spawn_calls[0]["args"]["group_id"] == spawn_calls[1]["args"]["group_id"]

    @pytest.mark.asyncio
    async def test_non_spawn_tools_not_injected_with_group_id(self):
        """Non-spawn_agent tools should NOT get a group_id injected."""
        batch_response = {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_a",
                            "type": "function",
                            "function": {
                                "name": "shell",
                                "arguments": json.dumps({"command": "ls"}),
                            },
                        },
                    ],
                }
            }]
        }

        mock_client = AsyncMock()
        mock_client.chat = AsyncMock(
            side_effect=[batch_response, _text_response("done")]
        )

        captured_args: list[dict] = []

        async def _capture_execute(name, arguments, executor=None):
            captured_args.append({"name": name, "args": arguments})
            return "file_list"

        mock_registry = AsyncMock()
        mock_registry.execute = AsyncMock(side_effect=_capture_execute)

        await run_tool_loop(
            client=mock_client,
            messages=[{"role": "user", "content": "list files"}],
            model="m",
            tools=[{"type": "function", "function": {"name": "shell"}}],
            tool_registry=mock_registry,
        )

        assert len(captured_args) == 1
        assert "group_id" not in captured_args[0]["args"]

    @pytest.mark.asyncio
    async def test_spawn_calls_in_different_batches_get_different_group_ids(self):
        """spawn_agent calls in separate model responses get different group_ids."""
        # First response: one spawn_agent call (returns stop_turn)
        # We need two separate iterations that each have a spawn_agent
        # But __stop_turn__ ends the loop. So we test via two separate
        # run_tool_loop invocations simulating two user turns.

        captured_args: list[dict] = []

        async def _capture_execute(name, arguments, executor=None):
            captured_args.append({"name": name, "args": arguments})
            return json.dumps({"__stop_turn__": True, "reply": "spawned"})

        # First turn
        mock_client_1 = AsyncMock()
        mock_client_1.chat = AsyncMock(return_value={
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "spawn_agent",
                            "arguments": json.dumps({"name": "a1", "prompt": "t1"}),
                        },
                    }],
                }
            }]
        })
        mock_registry = AsyncMock()
        mock_registry.execute = AsyncMock(side_effect=_capture_execute)

        await run_tool_loop(
            client=mock_client_1, messages=[], model="m",
            tools=[], tool_registry=mock_registry,
        )

        # Second turn
        mock_client_2 = AsyncMock()
        mock_client_2.chat = AsyncMock(return_value={
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": "call_2",
                        "type": "function",
                        "function": {
                            "name": "spawn_agent",
                            "arguments": json.dumps({"name": "a2", "prompt": "t2"}),
                        },
                    }],
                }
            }]
        })
        mock_registry_2 = AsyncMock()
        mock_registry_2.execute = AsyncMock(side_effect=_capture_execute)

        await run_tool_loop(
            client=mock_client_2, messages=[], model="m",
            tools=[], tool_registry=mock_registry_2,
        )

        spawn_calls = [c for c in captured_args if c["name"] == "spawn_agent"]
        assert len(spawn_calls) == 2
        assert spawn_calls[0]["args"]["group_id"] != spawn_calls[1]["args"]["group_id"]

    @pytest.mark.asyncio
    async def test_tool_rate_limit_returns_error_instead_of_execution(self):
        """A tool exceeding its per-turn limit gets an error result, not execution."""
        # Model calls "web_search" twice in separate iterations; limit is 1
        mock_client = AsyncMock()
        mock_client.chat = AsyncMock(
            side_effect=[
                _tool_call_response("web_search", {"query": "first"}, call_id="c1"),
                _tool_call_response("web_search", {"query": "second"}, call_id="c2"),
                _text_response("done"),
            ]
        )

        mock_registry = AsyncMock()
        mock_registry.execute = AsyncMock(return_value="search result")

        messages: list[dict] = [{"role": "user", "content": "search twice"}]
        tools = [{"type": "function", "function": {"name": "web_search"}}]

        result = await run_tool_loop(
            client=mock_client,
            messages=messages,
            model="m",
            tools=tools,
            tool_registry=mock_registry,
            tool_limits={"web_search": 1},
        )

        assert result == "done"
        # Only the first call should have been executed
        assert mock_registry.execute.call_count == 1
        # The second tool result message should contain the rate limit error
        tool_results = [m for m in messages if m.get("role") == "tool"]
        assert len(tool_results) == 2
        assert "Rate limit" in tool_results[1]["content"]
        assert "web_search" in tool_results[1]["content"]
        assert "2/1" in tool_results[1]["content"]

    @pytest.mark.asyncio
    async def test_call_counts_reset_between_run_tool_loop_calls(self):
        """Each invocation of run_tool_loop has independent call counts."""
        mock_registry = AsyncMock()
        mock_registry.execute = AsyncMock(return_value="ok")

        async def _run_once() -> str:
            client = AsyncMock()
            client.chat = AsyncMock(
                side_effect=[
                    _tool_call_response("shell", {"command": "ls"}, call_id="cx"),
                    _text_response("done"),
                ]
            )
            return await run_tool_loop(
                client=client,
                messages=[{"role": "user", "content": "run"}],
                model="m",
                tools=[{"type": "function", "function": {"name": "shell"}}],
                tool_registry=mock_registry,
                tool_limits={"shell": 1},
            )

        # Run twice — if counts leaked, second invocation would hit the limit
        result1 = await _run_once()
        result2 = await _run_once()

        assert result1 == "done"
        assert result2 == "done"
        # Both calls executed successfully (no rate-limit skip)
        assert mock_registry.execute.call_count == 2

    @pytest.mark.asyncio
    async def test_tools_without_limits_are_unlimited(self):
        """A tool not present in the limits dict is never rate-limited."""
        call_count = 3
        mock_client = AsyncMock()
        mock_client.chat = AsyncMock(
            side_effect=[
                _tool_call_response("custom_tool", {}, call_id=f"c{i}")
                for i in range(call_count)
            ]
            + [_text_response("done")]
        )

        mock_registry = AsyncMock()
        mock_registry.execute = AsyncMock(return_value="result")

        # custom_tool is absent from DEFAULT_TOOL_LIMITS, so it has no limit
        result = await run_tool_loop(
            client=mock_client,
            messages=[{"role": "user", "content": "go"}],
            model="m",
            tools=[{"type": "function", "function": {"name": "custom_tool"}}],
            tool_registry=mock_registry,
        )

        assert result == "done"
        assert mock_registry.execute.call_count == call_count

    @pytest.mark.asyncio
    async def test_none_limit_removes_default(self):
        """Setting a tool limit to None removes the default, making it unlimited."""
        # shell has a default limit of 10; override with None to remove it
        call_count = 12
        mock_client = AsyncMock()
        mock_client.chat = AsyncMock(
            side_effect=[
                _tool_call_response("shell", {"command": "ls"}, call_id=f"c{i}")
                for i in range(call_count)
            ]
            + [_text_response("done")]
        )

        mock_registry = AsyncMock()
        mock_registry.execute = AsyncMock(return_value="ok")

        result = await run_tool_loop(
            client=mock_client,
            messages=[{"role": "user", "content": "go"}],
            model="m",
            tools=[{"type": "function", "function": {"name": "shell"}}],
            tool_registry=mock_registry,
            tool_limits={"shell": None},
        )

        assert result == "done"
        # All 12 calls executed — default limit of 10 was removed
        assert mock_registry.execute.call_count == call_count


# ---------------------------------------------------------------------------
# OpenRouterClient.list_models
# ---------------------------------------------------------------------------

class TestListModels:
    """Tests for OpenRouterClient.list_models()."""

    @pytest.mark.asyncio
    async def test_list_models_fetches_and_caches(self):
        """list_models() fetches from API and caches the result."""
        models_data = {
            "data": [
                {"id": "google/gemini-2.0-flash", "name": "Gemini 2.0 Flash"},
                {"id": "anthropic/claude-sonnet", "name": "Claude Sonnet"},
            ]
        }
        mock_session = MagicMock()
        mock_session.closed = False
        mock_session.get = MagicMock(
            return_value=_make_response_ctx(200, json_data=models_data)
        )

        client = OpenRouterClient(api_key="k", semaphore=asyncio.Semaphore(1))
        client._session = mock_session

        result = await client.list_models()
        assert len(result) == 2
        assert result[0]["id"] == "google/gemini-2.0-flash"

        # Second call should use cache (no new HTTP call)
        result2 = await client.list_models()
        assert result2 == result
        assert mock_session.get.call_count == 1

    @pytest.mark.asyncio
    async def test_list_models_force_refresh(self):
        """force_refresh=True bypasses the cache."""
        models_data = {"data": [{"id": "m1", "name": "Model 1"}]}
        mock_session = MagicMock()
        mock_session.closed = False
        mock_session.get = MagicMock(
            return_value=_make_response_ctx(200, json_data=models_data)
        )

        client = OpenRouterClient(api_key="k", semaphore=asyncio.Semaphore(1))
        client._session = mock_session

        await client.list_models()
        await client.list_models(force_refresh=True)
        assert mock_session.get.call_count == 2

    @pytest.mark.asyncio
    async def test_list_models_error_raises(self):
        """list_models() raises OpenRouterError on HTTP errors."""
        mock_session = MagicMock()
        mock_session.closed = False
        mock_session.get = MagicMock(
            return_value=_make_response_ctx(401, text="unauthorized")
        )

        client = OpenRouterClient(api_key="bad", semaphore=asyncio.Semaphore(1))
        client._session = mock_session

        with pytest.raises(OpenRouterError):
            await client.list_models()
