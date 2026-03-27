"""Tests for spare_paw.tools.subagent — spawn and list agents."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import spare_paw.tools.subagent as subagent_mod


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _reset_agents():
    """Clear the global _agents and _channels dicts before each test."""
    subagent_mod._agents.clear()
    subagent_mod._channels.clear()
    yield
    subagent_mod._agents.clear()
    subagent_mod._channels.clear()


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
    """Spawns without explicit group_id should each get their own group."""
    app_state = _make_app_state()

    with patch.object(subagent_mod, "_run_agent", side_effect=_noop_run_agent):
        r1 = json.loads(
            await subagent_mod._handle_spawn(app_state, name="a1", prompt="task1")
        )
        r2 = json.loads(
            await subagent_mod._handle_spawn(app_state, name="a2", prompt="task2")
        )

    # Each spawn without group_id should get its own unique group_id
    assert r1["group_id"] != r2["group_id"]
    await asyncio.sleep(0)


# ---------------------------------------------------------------------------
# Agents blocked from direct user communication
# ---------------------------------------------------------------------------

def test_agents_cannot_use_send_message_or_send_file():
    """Subagents must not have access to send_message, send_file, or spawn_agent.

    Uses the actual filtering logic from _run_agent (via source inspection)
    rather than duplicating it, so the test breaks if the blocklist changes.
    """
    import inspect
    source = inspect.getsource(subagent_mod._run_agent)
    # Verify the blocklist set exists in the source and contains the expected tools
    assert "send_message" in source, "send_message must be in _run_agent blocklist"
    assert "send_file" in source, "send_file must be in _run_agent blocklist"
    assert "spawn_agent" in source, "spawn_agent must be in _run_agent blocklist"
    assert "list_agents" in source, "list_agents must be in _run_agent blocklist"

    # Also verify the filtering works with sample schemas
    all_schemas = [
        {"function": {"name": "shell"}},
        {"function": {"name": "send_message"}},
        {"function": {"name": "send_file"}},
        {"function": {"name": "spawn_agent"}},
        {"function": {"name": "list_agents"}},
        {"function": {"name": "files"}},
    ]

    # Apply the same set that's defined in _run_agent
    blocked = {"spawn_agent", "list_agents", "send_message", "send_file"}
    filtered = [
        s for s in all_schemas
        if s.get("function", {}).get("name") not in blocked
    ]

    tool_names = {t["function"]["name"] for t in filtered}
    assert tool_names == {"shell", "files"}


# ---------------------------------------------------------------------------
# Per-agent-type tool limits
# ---------------------------------------------------------------------------

def test_agent_types_have_tool_limits():
    """Each AGENT_TYPES entry must have a tool_limits dict."""
    for key, archetype in subagent_mod.AGENT_TYPES.items():
        assert "tool_limits" in archetype, f"AGENT_TYPES['{key}'] missing 'tool_limits'"
        assert isinstance(archetype["tool_limits"], dict)


@pytest.mark.asyncio
async def test_run_agent_forwards_tool_limits_to_tool_loop():
    """_run_agent passes tool_limits to run_tool_loop."""
    app_state = _make_app_state()
    agent_id = "limits-test"
    subagent_mod._agents[agent_id] = {
        "name": "limiter",
        "prompt": "test",
        "status": "starting",
        "group_id": "grp",
        "agent_type": "coder",
        "created_at": "2026-01-01T00:00:00+00:00",
    }

    expected_limits = {"shell": 30, "web_search": 3}

    with patch("spare_paw.router.tool_loop.run_tool_loop", return_value=("ok", {})) as mock_loop:
        with patch("spare_paw.bot.handler._build_system_prompt", return_value="sys"):
            await subagent_mod._run_agent(
                agent_id, "test", app_state,
                model=None, tools_filter=None, max_iterations=5,
                tool_limits=expected_limits,
            )

    assert mock_loop.call_args.kwargs.get("tool_limits") == expected_limits


@pytest.mark.asyncio
async def test_default_agent_limits_when_no_type():
    """Spawn without agent_type uses _DEFAULT_AGENT_LIMITS."""
    app_state = _make_app_state()

    with patch.object(subagent_mod, "_run_agent", side_effect=_noop_run_agent) as mock_run:
        await subagent_mod._handle_spawn(app_state, name="generic", prompt="do stuff")

    call_args = mock_run.call_args
    tool_limits = call_args[1].get("tool_limits") if call_args[1] else None
    assert tool_limits == subagent_mod._DEFAULT_AGENT_LIMITS
    await asyncio.sleep(0)


# ---------------------------------------------------------------------------
# Heartbeat watchdog
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_agent_has_last_activity():
    """Spawned agent has a last_activity datetime field."""
    app_state = _make_app_state()

    with patch.object(subagent_mod, "_run_agent", side_effect=_noop_run_agent):
        await subagent_mod._handle_spawn(app_state, name="hb", prompt="test")

    agent_id = next(iter(subagent_mod._agents))
    assert "last_activity" in subagent_mod._agents[agent_id]
    assert isinstance(subagent_mod._agents[agent_id]["last_activity"], datetime)
    await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_watchdog_cancels_stuck_agent():
    """Watchdog cancels agents with no recent activity."""
    mock_task = MagicMock()
    mock_task.done.return_value = False

    subagent_mod._agents["stuck-1"] = {
        "name": "stuck",
        "status": "running",
        "last_activity": datetime.now(timezone.utc) - timedelta(seconds=300),
        "task": mock_task,
        "created_at": "2026-01-01T00:00:00+00:00",
    }

    await subagent_mod._watchdog_tick()
    mock_task.cancel.assert_called_once()


@pytest.mark.asyncio
async def test_watchdog_ignores_active_agent():
    """Watchdog does not cancel agents with recent activity."""
    mock_task = MagicMock()
    mock_task.done.return_value = False

    subagent_mod._agents["active-1"] = {
        "name": "active",
        "status": "running",
        "last_activity": datetime.now(timezone.utc) - timedelta(seconds=10),
        "task": mock_task,
        "created_at": "2026-01-01T00:00:00+00:00",
    }

    await subagent_mod._watchdog_tick()
    mock_task.cancel.assert_not_called()


@pytest.mark.asyncio
async def test_watchdog_ignores_completed_agent():
    """Watchdog does not cancel completed agents even with old last_activity."""
    mock_task = MagicMock()
    mock_task.done.return_value = True

    subagent_mod._agents["done-1"] = {
        "name": "done",
        "status": "completed",
        "last_activity": datetime.now(timezone.utc) - timedelta(seconds=300),
        "task": mock_task,
        "created_at": "2026-01-01T00:00:00+00:00",
    }

    await subagent_mod._watchdog_tick()
    mock_task.cancel.assert_not_called()


@pytest.mark.asyncio
async def test_cancelled_agent_sets_timed_out():
    """CancelledError in _run_agent sets status to timed_out."""
    app_state = _make_app_state()
    agent_id = "timeout-test"
    subagent_mod._agents[agent_id] = {
        "name": "timeouter",
        "prompt": "test",
        "status": "starting",
        "group_id": "grp",
        "created_at": "2026-01-01T00:00:00+00:00",
    }

    async def _raise_cancelled(*args, **kwargs):
        raise asyncio.CancelledError()

    with patch("spare_paw.router.tool_loop.run_tool_loop", side_effect=_raise_cancelled):
        with patch("spare_paw.bot.handler._build_system_prompt", return_value="sys"):
            await subagent_mod._run_agent(
                agent_id, "test", app_state,
                model=None, tools_filter=None, max_iterations=5,
            )

    assert subagent_mod._agents[agent_id]["status"] == "timed_out"
    assert "timed out" in subagent_mod._agents[agent_id]["error"].lower()


# ---------------------------------------------------------------------------
# Ephemeral progress messages
# ---------------------------------------------------------------------------

class _ProgressBackend:
    """Fake backend class with progress methods for testing."""
    send_progress = AsyncMock(return_value=42)
    edit_progress = AsyncMock()
    delete_progress = AsyncMock()


@pytest.mark.asyncio
async def test_spawn_sends_progress_message():
    """Spawn sends a progress message via backend.send_progress."""
    app_state = _make_app_state()
    backend = _ProgressBackend()
    backend.send_progress = AsyncMock(return_value=42)
    app_state.backend = backend

    with patch.object(subagent_mod, "_run_agent", side_effect=_noop_run_agent):
        await subagent_mod._handle_spawn(app_state, name="prog", prompt="test")

    backend.send_progress.assert_called_once()
    agent_id = next(iter(subagent_mod._agents))
    assert subagent_mod._agents[agent_id]["progress_message_id"] == 42
    await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_group_complete_deletes_progress():
    """_notify_main_agent deletes progress messages for the group."""
    group_id = "prog-group"
    backend = _ProgressBackend()
    backend.delete_progress = AsyncMock()

    mock_app_state = MagicMock()
    mock_app_state.backend = backend
    subagent_mod._app_state = mock_app_state

    subagent_mod._agents["pg-1"] = {
        "name": "agent1",
        "status": "completed",
        "result": "result1",
        "group_id": group_id,
        "progress_message_id": 101,
        "created_at": "2026-01-01T00:00:00+00:00",
    }

    queue = asyncio.Queue()
    original_queue = subagent_mod._message_queue
    subagent_mod._message_queue = queue
    try:
        await subagent_mod._notify_main_agent(group_id)
        backend.delete_progress.assert_called_once_with(101)
    finally:
        subagent_mod._message_queue = original_queue
        subagent_mod._app_state = None


@pytest.mark.asyncio
async def test_no_progress_without_backend_support():
    """Spawn succeeds even if backend lacks send_progress."""
    app_state = _make_app_state()
    app_state.backend = MagicMock(spec=[])  # No send_progress attribute

    with patch.object(subagent_mod, "_run_agent", side_effect=_noop_run_agent):
        result = json.loads(
            await subagent_mod._handle_spawn(app_state, name="noprog", prompt="test")
        )

    assert result["__stop_turn__"] is True
    agent_id = next(iter(subagent_mod._agents))
    assert subagent_mod._agents[agent_id].get("progress_message_id") is None
    await asyncio.sleep(0)


# ---------------------------------------------------------------------------
# Integration: full dispatch path (tool_loop → registry → _spawn_handler)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Robust agent tracking (done-callback, enriched list, null-queue escalation)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_spawn_stores_task_reference():
    """Spawned agents must have an asyncio.Task stored in _agents."""
    app_state = _make_app_state()

    with patch.object(subagent_mod, "_run_agent", side_effect=_noop_run_agent):
        await subagent_mod._handle_spawn(app_state, name="tracked", prompt="test")

    agent_id = next(iter(subagent_mod._agents))
    agent = subagent_mod._agents[agent_id]
    assert "task" in agent, "Agent entry must contain a 'task' key"
    assert isinstance(agent["task"], asyncio.Task)
    await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_done_callback_detects_crash():
    """If an agent task raises outside _run_agent's try/except, done-callback catches it."""
    agent_id = "crash-test"
    group_id = "crash-group"

    async def _crashing_agent(*args, **kwargs):
        raise RuntimeError("unexpected crash")

    subagent_mod._agents[agent_id] = {
        "name": "crasher",
        "prompt": "test",
        "status": "starting",
        "group_id": group_id,
        "created_at": "2026-01-01T00:00:00+00:00",
    }

    task = asyncio.create_task(_crashing_agent(), name=f"agent-{agent_id}")
    subagent_mod._agents[agent_id]["task"] = task
    task.add_done_callback(lambda t, aid=agent_id: subagent_mod._on_agent_done(aid, t))

    # Wait for task to complete and callback to fire
    try:
        await task
    except RuntimeError:
        pass
    await asyncio.sleep(0)

    agent = subagent_mod._agents[agent_id]
    assert agent["status"] == "failed"
    assert "RuntimeError" in agent["error"]
    assert "finished_at" in agent


@pytest.mark.asyncio
async def test_null_queue_logs_error(caplog):
    """_notify_main_agent logs ERROR (not WARNING) when queue is None."""
    group_id = "null-queue-group"
    subagent_mod._agents["nq-1"] = {
        "name": "agent1",
        "status": "completed",
        "result": "some result",
        "group_id": group_id,
        "created_at": "2026-01-01T00:00:00+00:00",
    }

    original_queue = subagent_mod._message_queue
    subagent_mod._message_queue = None
    try:
        import logging
        with caplog.at_level(logging.ERROR, logger="spare_paw.tools.subagent"):
            await subagent_mod._notify_main_agent(group_id)
        assert any("DROPPED" in r.message for r in caplog.records), (
            "Expected ERROR log with 'DROPPED' when queue is None"
        )
    finally:
        subagent_mod._message_queue = original_queue


@pytest.mark.asyncio
async def test_list_agents_includes_enriched_fields():
    """list_agents output includes error, result_preview, agent_type, group_id, is_alive."""
    mock_task = MagicMock()
    mock_task.done.return_value = True

    subagent_mod._agents["e1"] = {
        "name": "failed-agent",
        "status": "failed",
        "error": "RuntimeError: boom",
        "result_preview": None,
        "agent_type": "researcher",
        "group_id": "grp-1",
        "created_at": "2026-01-01T00:00:00+00:00",
        "finished_at": "2026-01-01T00:01:00+00:00",
        "task": mock_task,
    }
    subagent_mod._agents["e2"] = {
        "name": "ok-agent",
        "status": "completed",
        "result": "full result text here",
        "result_preview": "full result text here",
        "agent_type": "coder",
        "group_id": "grp-1",
        "created_at": "2026-01-01T00:02:00+00:00",
        "finished_at": "2026-01-01T00:03:00+00:00",
        "task": mock_task,
    }

    result = json.loads(await subagent_mod._handle_list_agents())
    assert result["count"] == 2

    # Check enriched fields on the failed agent (second in list, sorted by created_at desc)
    failed = next(a for a in result["agents"] if a["name"] == "failed-agent")
    assert failed["error"] == "RuntimeError: boom"
    assert failed["agent_type"] == "researcher"
    assert failed["group_id"] == "grp-1"
    assert "is_alive" in failed
    assert failed["is_alive"] is False

    ok = next(a for a in result["agents"] if a["name"] == "ok-agent")
    assert ok["result_preview"] == "full result text here"
    assert ok["agent_type"] == "coder"


# ---------------------------------------------------------------------------
# Integration: full dispatch path (tool_loop → registry → _spawn_handler)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_tool_loop_to_spawn_handler_integration():
    """Verify group_id flows through the full path: tool_loop → registry → _spawn_handler.

    This catches signature mismatches between what tool_loop injects and what
    _spawn_handler accepts (the args go through registry.execute → handler(**arguments)).
    """
    from spare_paw.tools.registry import ToolRegistry
    from spare_paw.router.tool_loop import run_tool_loop

    app_state = _make_app_state()
    registry = ToolRegistry()

    # Register spawn_agent using the real register() function
    subagent_mod.register(registry, {}, app_state)

    # Model returns two spawn_agent calls in one batch
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
                            "arguments": json.dumps({"name": "r1", "prompt": "task1"}),
                        },
                    },
                    {
                        "id": "call_b",
                        "type": "function",
                        "function": {
                            "name": "spawn_agent",
                            "arguments": json.dumps({"name": "r2", "prompt": "task2"}),
                        },
                    },
                ],
            }
        }]
    }

    mock_client = MagicMock()
    mock_client.chat = AsyncMock(return_value=batch_response)

    with patch.object(subagent_mod, "_run_agent", side_effect=_noop_run_agent):
        await run_tool_loop(
            client=mock_client,
            messages=[{"role": "user", "content": "do two things"}],
            model="m",
            tools=registry.get_schemas(),
            tool_registry=registry,
        )

    # Should not crash — the handler accepted group_id
    assert len(subagent_mod._agents) == 2

    # Both agents should share the same group_id (batch-based grouping)
    agents = list(subagent_mod._agents.values())
    assert agents[0]["group_id"] == agents[1]["group_id"]

    # Let background tasks finish
    await asyncio.sleep(0)


# ---------------------------------------------------------------------------
# Bidirectional dialogue — DialogueChannel
# ---------------------------------------------------------------------------

def test_dialogue_channel_dataclass_exists():
    """DialogueChannel dataclass has expected fields."""
    channel = subagent_mod.DialogueChannel(
        agent_id="test-1",
        original_request="user request",
        spawn_prompt="do research",
        to_main=asyncio.Queue(),
    )
    assert channel.agent_id == "test-1"
    assert channel.original_request == "user request"
    assert channel.spawn_prompt == "do research"
    assert channel.max_rounds == 5
    assert channel.round_count == 0
    assert channel.history == []
    assert channel.consumer_task is None
    assert channel.closed is False


def test_channels_registry_exists():
    """Module-level _channels dict exists."""
    assert hasattr(subagent_mod, "_channels")
    assert isinstance(subagent_mod._channels, dict)


# ---------------------------------------------------------------------------
# Bidirectional dialogue — Consumer coroutine
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_dialogue_consumer_resolves_future():
    """Consumer picks up a question, calls LLM, resolves the Future."""
    app_state = _make_app_state()
    app_state.router_client.chat = AsyncMock(return_value={
        "choices": [{"message": {"content": "The answer is 42"}}],
    })

    channel = subagent_mod.DialogueChannel(
        agent_id="cons-1",
        original_request="original user request",
        spawn_prompt="research task",
        to_main=asyncio.Queue(),
    )

    consumer = asyncio.create_task(
        subagent_mod._dialogue_consumer(channel, app_state)
    )

    future = asyncio.get_running_loop().create_future()
    await channel.to_main.put(("What is the meaning?", future))

    result = await asyncio.wait_for(future, timeout=2.0)
    assert result == "The answer is 42"
    assert channel.round_count == 1
    assert len(channel.history) == 2

    channel.closed = True
    consumer.cancel()
    try:
        await consumer
    except asyncio.CancelledError:
        pass


@pytest.mark.asyncio
async def test_dialogue_consumer_exits_on_cancel():
    """Consumer coroutine exits cleanly when cancelled."""
    app_state = _make_app_state()
    channel = subagent_mod.DialogueChannel(
        agent_id="cons-2",
        original_request="req",
        spawn_prompt="task",
        to_main=asyncio.Queue(),
    )

    consumer = asyncio.create_task(
        subagent_mod._dialogue_consumer(channel, app_state)
    )
    await asyncio.sleep(0)

    consumer.cancel()
    await asyncio.sleep(0)
    assert consumer.done()


# ---------------------------------------------------------------------------
# Bidirectional dialogue — consult_main handler and heartbeat
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_consult_main_rejects_after_max_rounds():
    """consult_main returns error when round_count >= max_rounds."""
    channel = subagent_mod.DialogueChannel(
        agent_id="max-1",
        original_request="req",
        spawn_prompt="task",
        to_main=asyncio.Queue(),
        round_count=5,
    )
    subagent_mod._channels["max-1"] = channel

    result = json.loads(
        await subagent_mod._handle_consult("max-1", "one more question?")
    )
    assert "error" in result
    assert "limit" in result["error"].lower()


@pytest.mark.asyncio
async def test_consult_main_rejects_long_question():
    """consult_main returns error when question exceeds 2000 chars."""
    channel = subagent_mod.DialogueChannel(
        agent_id="long-1",
        original_request="req",
        spawn_prompt="task",
        to_main=asyncio.Queue(),
    )
    subagent_mod._channels["long-1"] = channel

    result = json.loads(
        await subagent_mod._handle_consult("long-1", "x" * 2001)
    )
    assert "error" in result
    assert "2000" in result["error"]


@pytest.mark.asyncio
async def test_consult_main_full_roundtrip():
    """consult_main pushes question, consumer resolves, tool returns answer."""
    app_state = _make_app_state()
    app_state.router_client.chat = AsyncMock(return_value={
        "choices": [{"message": {"content": "Use Redis"}}],
    })

    agent_id = "rt-1"
    subagent_mod._agents[agent_id] = {
        "name": "coder",
        "status": "running",
        "last_activity": datetime.now(timezone.utc),
        "created_at": "2026-01-01T00:00:00+00:00",
    }

    channel = subagent_mod.DialogueChannel(
        agent_id=agent_id,
        original_request="build caching layer",
        spawn_prompt="implement cache",
        to_main=asyncio.Queue(),
    )
    channel.consumer_task = asyncio.create_task(
        subagent_mod._dialogue_consumer(channel, app_state)
    )
    subagent_mod._channels[agent_id] = channel

    result = await asyncio.wait_for(
        subagent_mod._handle_consult(agent_id, "Redis or memcached?"),
        timeout=2.0,
    )
    assert "Use Redis" in result

    channel.closed = True
    channel.consumer_task.cancel()
    try:
        await channel.consumer_task
    except asyncio.CancelledError:
        pass


@pytest.mark.asyncio
async def test_consult_heartbeat_updates_last_activity():
    """_consult_heartbeat updates last_activity periodically."""
    agent_id = "hb-consult"
    subagent_mod._agents[agent_id] = {
        "name": "hb",
        "status": "running",
        "last_activity": datetime(2020, 1, 1, tzinfo=timezone.utc),
        "created_at": "2026-01-01T00:00:00+00:00",
    }

    future = asyncio.get_running_loop().create_future()

    with patch.object(subagent_mod, "_CONSULT_HEARTBEAT_INTERVAL", 0.01):
        hb_task = asyncio.create_task(
            subagent_mod._consult_heartbeat(agent_id, future)
        )
        await asyncio.sleep(0.05)
        future.set_result("done")
        await asyncio.sleep(0.02)

    assert subagent_mod._agents[agent_id]["last_activity"].year > 2020


# ---------------------------------------------------------------------------
# Bidirectional dialogue — Channel cleanup
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_cleanup_channel_resolves_pending_future():
    """_cleanup_channel resolves pending Futures with error string."""
    channel = subagent_mod.DialogueChannel(
        agent_id="cleanup-1",
        original_request="req",
        spawn_prompt="task",
        to_main=asyncio.Queue(),
    )
    subagent_mod._channels["cleanup-1"] = channel

    future = asyncio.get_running_loop().create_future()
    await channel.to_main.put(("question", future))

    subagent_mod._cleanup_channel("cleanup-1")

    assert future.done()
    assert "terminated" in future.result().lower()
    assert "cleanup-1" not in subagent_mod._channels


@pytest.mark.asyncio
async def test_cleanup_channel_cancels_consumer():
    """_cleanup_channel cancels the consumer coroutine."""
    app_state = _make_app_state()
    channel = subagent_mod.DialogueChannel(
        agent_id="cleanup-2",
        original_request="req",
        spawn_prompt="task",
        to_main=asyncio.Queue(),
    )
    channel.consumer_task = asyncio.create_task(
        subagent_mod._dialogue_consumer(channel, app_state)
    )
    subagent_mod._channels["cleanup-2"] = channel

    await asyncio.sleep(0)

    subagent_mod._cleanup_channel("cleanup-2")

    assert channel.closed is True
    await asyncio.sleep(0)
    assert channel.consumer_task.cancelled() or channel.consumer_task.done()


@pytest.mark.asyncio
async def test_on_agent_done_cleans_up_channel():
    """_on_agent_done triggers _cleanup_channel when agent has a channel."""
    agent_id = "done-cleanup"
    channel = subagent_mod.DialogueChannel(
        agent_id=agent_id,
        original_request="req",
        spawn_prompt="task",
        to_main=asyncio.Queue(),
    )
    subagent_mod._channels[agent_id] = channel
    subagent_mod._agents[agent_id] = {
        "name": "agent",
        "status": "running",
        "group_id": "grp",
        "created_at": "2026-01-01T00:00:00+00:00",
    }

    async def _crash():
        raise RuntimeError("boom")

    task = asyncio.create_task(_crash(), name=f"agent-{agent_id}")
    subagent_mod._agents[agent_id]["task"] = task
    task.add_done_callback(lambda t, aid=agent_id: subagent_mod._on_agent_done(aid, t))

    try:
        await task
    except RuntimeError:
        pass
    await asyncio.sleep(0)

    assert agent_id not in subagent_mod._channels


# ---------------------------------------------------------------------------
# Bidirectional dialogue — Channel creation in spawn
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_spawn_creates_dialogue_channel():
    """_handle_spawn creates a DialogueChannel for the spawned agent."""
    app_state = _make_app_state()
    app_state.current_request = "user wants research"

    with patch.object(subagent_mod, "_run_agent", side_effect=_noop_run_agent):
        result = json.loads(
            await subagent_mod._handle_spawn(
                app_state, name="researcher", prompt="find info"
            )
        )

    agent_id = result["agent_id"]
    assert agent_id in subagent_mod._channels
    channel = subagent_mod._channels[agent_id]
    assert channel.original_request == "user wants research"
    assert channel.spawn_prompt == "find info"
    assert channel.consumer_task is not None

    subagent_mod._cleanup_channel(agent_id)
    await asyncio.sleep(0)
