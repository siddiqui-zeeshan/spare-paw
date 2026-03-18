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
    yield
    subagent_mod._agents.clear()
    subagent_mod._last_spawn_time = 0


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
