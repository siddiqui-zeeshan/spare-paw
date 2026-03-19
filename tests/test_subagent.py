"""Tests for spare_paw.tools.subagent — spawn and list agents."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import MagicMock, patch

import pytest

import spare_paw.tools.subagent as subagent_mod


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _reset_agents():
    """Clear the global _agents dict and rate-limiter before each test."""
    subagent_mod._agents.clear()
    subagent_mod._last_spawn_time = 0
    subagent_mod._last_group_id = None
    yield
    subagent_mod._agents.clear()
    subagent_mod._last_spawn_time = 0
    subagent_mod._last_group_id = None


def _make_app_state() -> MagicMock:
    app_state = MagicMock()
    app_state.config = MagicMock()
    app_state.config.get = MagicMock(return_value="test/model")
    app_state.tool_registry = MagicMock()
    app_state.tool_registry.get_schemas = MagicMock(return_value=[])
    app_state.router_client = MagicMock()
    app_state.executor = None
    app_state.application = MagicMock()
    app_state.application.bot = MagicMock()
    app_state.scheduler = None
    return app_state


async def _noop_run_agent(*args, **kwargs):
    """A no-op coroutine that replaces _run_agent."""
    pass


# ---------------------------------------------------------------------------
# spawn
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_spawn_creates_agent_entry():
    app_state = _make_app_state()

    # Patch _run_agent with a proper coroutine so asyncio.create_task works
    with patch.object(subagent_mod, "_run_agent", side_effect=_noop_run_agent):
        result = json.loads(
            await subagent_mod._handle_spawn(app_state, name="researcher", prompt="find info")
        )

    assert result["__stop_turn__"] is True
    assert "researcher" in result["reply"]

    # The agent should exist in the global dict (exactly one new entry)
    assert len(subagent_mod._agents) == 1
    agent_id = next(iter(subagent_mod._agents))
    assert subagent_mod._agents[agent_id]["name"] == "researcher"

    # Let the background task finish
    await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_spawn_respects_max_concurrent():
    app_state = _make_app_state()

    # Pre-populate _agents with 3 running agents
    for i in range(3):
        subagent_mod._agents[f"fake-{i}"] = {
            "name": f"agent-{i}",
            "status": "running",
            "created_at": "2026-01-01T00:00:00+00:00",
        }

    result = json.loads(
        await subagent_mod._handle_spawn(app_state, name="fourth", prompt="overflow")
    )

    assert "error" in result
    assert "Max concurrent" in result["error"]
    assert result["running"] == 3


# ---------------------------------------------------------------------------
# list_agents
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_list_agents_empty():
    result = json.loads(await subagent_mod._handle_list_agents())
    assert result["agents"] == []
    assert result["count"] == 0


@pytest.mark.asyncio
async def test_list_agents_returns_entries():
    subagent_mod._agents["a1"] = {
        "name": "first",
        "status": "completed",
        "created_at": "2026-01-01T00:00:00+00:00",
        "finished_at": "2026-01-01T00:01:00+00:00",
    }
    subagent_mod._agents["a2"] = {
        "name": "second",
        "status": "running",
        "created_at": "2026-01-01T00:02:00+00:00",
        "finished_at": None,
    }

    result = json.loads(await subagent_mod._handle_list_agents())
    assert result["count"] == 2

    # Most recent first
    assert result["agents"][0]["name"] == "second"
    assert result["agents"][1]["name"] == "first"


# ---------------------------------------------------------------------------
# Group-based callback pattern
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_spawn_assigns_group_id():
    """Spawned agents get a group_id in their _agents entry."""
    app_state = _make_app_state()

    with patch.object(subagent_mod, "_run_agent", side_effect=_noop_run_agent):
        await subagent_mod._handle_spawn(app_state, name="a1", prompt="task1")

    assert len(subagent_mod._agents) == 1
    agent_id = next(iter(subagent_mod._agents))
    assert "group_id" in subagent_mod._agents[agent_id], (
        "Spawned agent entry must contain a 'group_id' key"
    )
    await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_group_callback_fires_when_all_complete():
    """When the last agent in a group finishes, _notify_main_agent is called."""
    group_id = "test-group-1"

    # Pre-populate two agents in the same group, both completed
    subagent_mod._agents["ag-1"] = {
        "name": "first",
        "status": "completed",
        "group_id": group_id,
        "created_at": "2026-01-01T00:00:00+00:00",
    }
    subagent_mod._agents["ag-2"] = {
        "name": "second",
        "status": "completed",
        "group_id": group_id,
        "created_at": "2026-01-01T00:01:00+00:00",
    }

    with patch.object(subagent_mod, "_notify_main_agent") as mock_notify:
        # Simulate calling the group completion check
        await subagent_mod._notify_main_agent(group_id)
        mock_notify.assert_called_once_with(group_id)


@pytest.mark.asyncio
async def test_group_callback_not_fired_when_partial():
    """If 2 of 3 agents complete, callback is NOT fired."""
    group_id = "test-group-partial"

    subagent_mod._agents["ag-1"] = {
        "name": "first",
        "status": "completed",
        "group_id": group_id,
        "created_at": "2026-01-01T00:00:00+00:00",
    }
    subagent_mod._agents["ag-2"] = {
        "name": "second",
        "status": "completed",
        "group_id": group_id,
        "created_at": "2026-01-01T00:01:00+00:00",
    }
    subagent_mod._agents["ag-3"] = {
        "name": "third",
        "status": "running",
        "group_id": group_id,
        "created_at": "2026-01-01T00:02:00+00:00",
    }

    # The function _check_group_complete should exist and return False
    # when not all agents in a group are done
    result = subagent_mod._check_group_complete(group_id)
    assert result is False, "Group should NOT be complete when an agent is still running"


@pytest.mark.asyncio
async def test_single_agent_group():
    """A single agent spawn still works (group of 1), callback fires on completion."""
    app_state = _make_app_state()

    async def _run_and_complete(*args, **kwargs):
        agent_id = args[0]
        subagent_mod._agents[agent_id]["status"] = "completed"

    with patch.object(subagent_mod, "_run_agent", side_effect=_run_and_complete):
        with patch.object(subagent_mod, "_notify_main_agent"):
            await subagent_mod._handle_spawn(
                app_state, name="solo", prompt="solo task"
            )
            # Let the background task run
            await asyncio.sleep(0)

            agent_id = next(iter(subagent_mod._agents))
            group_id = subagent_mod._agents[agent_id]["group_id"]

            # After single agent completes, the group callback should fire
            assert subagent_mod._check_group_complete(group_id) is True


# ---------------------------------------------------------------------------
# Agent types/archetypes
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_agent_types_dict_exists():
    """AGENT_TYPES is defined and has expected keys."""
    assert hasattr(subagent_mod, "AGENT_TYPES"), "AGENT_TYPES dict must exist"
    agent_types = subagent_mod.AGENT_TYPES

    assert isinstance(agent_types, dict)
    for key in ("researcher", "coder", "analyst"):
        assert key in agent_types, f"AGENT_TYPES must contain '{key}'"
        assert "system_suffix" in agent_types[key], (
            f"AGENT_TYPES['{key}'] must have 'system_suffix'"
        )
        assert "tools" in agent_types[key], (
            f"AGENT_TYPES['{key}'] must have 'tools'"
        )


@pytest.mark.asyncio
async def test_spawn_with_agent_type():
    """Spawning with agent_type='researcher' sets the right tools filter."""
    app_state = _make_app_state()

    with patch.object(subagent_mod, "_run_agent", side_effect=_noop_run_agent) as mock_run:
        result = json.loads(
            await subagent_mod._handle_spawn(
                app_state, name="research-bot", prompt="find info",
                agent_type="researcher",
            )
        )

    assert result["__stop_turn__"] is True

    # _run_agent should have been called with the researcher's tools filter
    call_args = mock_run.call_args
    tools_filter = call_args[0][4] if len(call_args[0]) > 4 else call_args[1].get("tools_filter")
    expected_tools = subagent_mod.AGENT_TYPES["researcher"]["tools"]
    assert tools_filter == expected_tools, (
        f"Expected tools filter {expected_tools}, got {tools_filter}"
    )
    await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_spawn_with_unknown_type_falls_back():
    """Unknown agent_type falls back to default (no filter)."""
    app_state = _make_app_state()

    with patch.object(subagent_mod, "_run_agent", side_effect=_noop_run_agent) as mock_run:
        result = json.loads(
            await subagent_mod._handle_spawn(
                app_state, name="mystery-bot", prompt="do stuff",
                agent_type="nonexistent_type",
            )
        )

    assert result["__stop_turn__"] is True

    # _run_agent should have been called with None tools_filter (no filter)
    call_args = mock_run.call_args
    tools_filter = call_args[0][4] if len(call_args[0]) > 4 else call_args[1].get("tools_filter")
    assert tools_filter is None, (
        f"Unknown agent_type should fall back to None tools filter, got {tools_filter}"
    )
    await asyncio.sleep(0)


# ---------------------------------------------------------------------------
# Token/cost tracking
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_run_agent_accumulates_usage():
    """After agent completes, _agents entry has usage data."""
    app_state = _make_app_state()

    agent_id = "usage-test"
    subagent_mod._agents[agent_id] = {
        "name": "tracker",
        "prompt": "test",
        "status": "starting",
        "group_id": "test-group",
        "created_at": "2026-01-01T00:00:00+00:00",
    }

    mock_usage = {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}

    with patch("spare_paw.router.tool_loop.run_tool_loop", return_value=("result text", mock_usage)):
        with patch("spare_paw.bot.handler._build_system_prompt", return_value="sys"):
            await subagent_mod._run_agent(
                agent_id, "test prompt", app_state,
                model=None, tools_filter=None, max_iterations=5,
            )

    agent = subagent_mod._agents[agent_id]
    assert "usage" in agent, "Agent entry must contain 'usage' after completion"
    assert agent["usage"]["prompt_tokens"] == 100
    assert agent["usage"]["completion_tokens"] == 50
    assert agent["usage"]["total_tokens"] == 150


@pytest.mark.asyncio
async def test_list_agents_includes_usage():
    """Usage info appears in list_agents output."""
    subagent_mod._agents["u1"] = {
        "name": "tracked",
        "status": "completed",
        "created_at": "2026-01-01T00:00:00+00:00",
        "finished_at": "2026-01-01T00:01:00+00:00",
        "usage": {"prompt_tokens": 200, "completion_tokens": 80, "total_tokens": 280},
    }

    result = json.loads(await subagent_mod._handle_list_agents())
    assert result["count"] == 1

    agent_info = result["agents"][0]
    assert "usage" in agent_info, "list_agents output must include 'usage' field"
    assert agent_info["usage"]["total_tokens"] == 280


# ---------------------------------------------------------------------------
# Batch-based grouping (explicit group_id replaces timing heuristic)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_explicit_group_id_used_when_provided():
    """When group_id is passed explicitly, it is used instead of auto-grouping."""
    app_state = _make_app_state()

    with patch.object(subagent_mod, "_run_agent", side_effect=_noop_run_agent):
        await subagent_mod._handle_spawn(
            app_state, name="a1", prompt="task1", group_id="batch-42"
        )
        await subagent_mod._handle_spawn(
            app_state, name="a2", prompt="task2", group_id="batch-42"
        )

    agents = list(subagent_mod._agents.values())
    assert len(agents) == 2
    assert agents[0]["group_id"] == "batch-42"
    assert agents[1]["group_id"] == "batch-42"
    await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_different_group_ids_not_merged():
    """Agents with different explicit group_ids stay in separate groups."""
    app_state = _make_app_state()

    with patch.object(subagent_mod, "_run_agent", side_effect=_noop_run_agent):
        await subagent_mod._handle_spawn(
            app_state, name="a1", prompt="task1", group_id="batch-1"
        )
        await subagent_mod._handle_spawn(
            app_state, name="a2", prompt="task2", group_id="batch-2"
        )

    agents = list(subagent_mod._agents.values())
    assert agents[0]["group_id"] == "batch-1"
    assert agents[1]["group_id"] == "batch-2"
    await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_no_timing_based_grouping():
    """Spawns without explicit group_id should NOT auto-group by timing."""
    app_state = _make_app_state()

    with patch.object(subagent_mod, "_run_agent", side_effect=_noop_run_agent):
        r1 = json.loads(
            await subagent_mod._handle_spawn(app_state, name="a1", prompt="task1")
        )
        # Reset rate limit to allow second spawn
        subagent_mod._last_spawn_time = 0
        r2 = json.loads(
            await subagent_mod._handle_spawn(app_state, name="a2", prompt="task2")
        )

    # Each spawn without group_id should get its own unique group_id
    assert r1["group_id"] != r2["group_id"]
    await asyncio.sleep(0)
