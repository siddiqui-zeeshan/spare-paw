"""Subagent spawn tool — run background agents with their own tool loops.

A spawned agent gets its own message context, runs independently, and
delivers results back to the main agent via the message queue callback
pattern. Results are synthesized by the main LLM before being sent to
the user.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

# Active and completed agents
_agents: dict[str, dict[str, Any]] = {}
_MAX_CONCURRENT = 3
_MAX_PER_GROUP = 3  # max agents in a single group/batch

# Reference to the message queue — set by handler.py at startup
_message_queue: asyncio.Queue | None = None

# ---------------------------------------------------------------------------
# Agent types / archetypes
# ---------------------------------------------------------------------------

AGENT_TYPES: dict[str, dict[str, Any]] = {
    "researcher": {
        "system_suffix": (
            "You are a research agent. Search thoroughly, use multiple sources, "
            "and cite URLs when possible. Focus on finding accurate, up-to-date information."
        ),
        "tools": ["tavily_search", "web_scrape", "shell", "files"],
    },
    "coder": {
        "system_suffix": (
            "You are a coding agent. Write, test, and debug code. "
            "Use the shell to run commands and verify your work."
        ),
        "tools": ["shell", "files", "code"],
    },
    "analyst": {
        "system_suffix": (
            "You are an analysis agent. Analyze data, produce summaries, "
            "and extract key insights. Be thorough but concise."
        ),
        "tools": ["files", "shell", "tavily_search"],
    },
}


# ---------------------------------------------------------------------------
# Group completion check
# ---------------------------------------------------------------------------

def _check_group_complete(group_id: str) -> bool:
    """Return True if all agents in the group have finished (completed or failed)."""
    group = [a for a in _agents.values() if a.get("group_id") == group_id]
    if not group:
        return False
    return all(a["status"] in ("completed", "failed") for a in group)


async def _notify_main_agent(group_id: str) -> None:
    """Push a synthetic message onto the queue with all group results bundled."""
    group = [
        (aid, info)
        for aid, info in _agents.items()
        if info.get("group_id") == group_id
    ]

    parts = []
    for _aid, agent in group:
        name = agent.get("name", "unnamed")
        status = agent["status"]
        if status == "completed":
            result = agent.get("result", "(no result)")
            parts.append(f"## {name}\n{result}")
        else:
            error = agent.get("error", "unknown error")
            parts.append(f"## {name}\nFAILED: {error}")

    synthetic_text = (
        "[AGENT_RESULTS]\n"
        f"{len(group)} background agent(s) finished.\n\n"
        + "\n\n".join(parts)
    )

    if _message_queue is not None:
        await _message_queue.put(("agent_callback", synthetic_text))
        logger.info("Group %s: pushed callback with %d agent results", group_id, len(group))
    else:
        logger.warning("Group %s completed but no message queue available", group_id)


# ---------------------------------------------------------------------------
# Agent execution
# ---------------------------------------------------------------------------

async def _run_agent(
    agent_id: str,
    prompt: str,
    app_state: Any,
    model: str | None,
    tools_filter: list[str] | None,
    max_iterations: int,
    system_suffix: str | None = None,
) -> None:
    """Background task that runs a self-contained tool loop and notifies on completion."""
    _agents[agent_id]["status"] = "running"
    _agents[agent_id]["started_at"] = datetime.now(timezone.utc).isoformat()

    try:
        from spare_paw.bot.handler import _build_system_prompt
        from spare_paw.router.tool_loop import run_tool_loop

        # Build system prompt
        system_prompt = await _build_system_prompt(app_state.config)
        if system_suffix:
            system_prompt = f"{system_prompt}\n\n{system_suffix}"

        # Resolve model
        resolved_model = (
            model
            or app_state.config.get("models.default", "google/gemini-2.0-flash")
        )

        # Get tool schemas (filtered if requested)
        all_schemas = app_state.tool_registry.get_schemas()
        if tools_filter:
            allowed = set(tools_filter)
            tool_schemas = [
                s for s in all_schemas
                if s.get("function", {}).get("name") in allowed
            ]
        else:
            tool_schemas = all_schemas

        # Agents cannot spawn other agents
        _agent_tools = {"spawn_agent", "list_agents"}
        tool_schemas = [
            s for s in tool_schemas
            if s.get("function", {}).get("name") not in _agent_tools
        ]

        # Build messages
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

        # Run tool loop with usage tracking
        result_text, usage = await run_tool_loop(
            client=app_state.router_client,
            messages=messages,
            model=resolved_model,
            tools=tool_schemas,
            tool_registry=app_state.tool_registry,
            max_iterations=max_iterations,
            executor=app_state.executor,
            track_usage=True,
        )

        _agents[agent_id]["status"] = "completed"
        _agents[agent_id]["result"] = result_text
        _agents[agent_id]["result_preview"] = result_text[:200]
        _agents[agent_id]["usage"] = usage
        logger.info("Agent %s completed", agent_id[:8])

    except Exception as exc:
        error_msg = f"{type(exc).__name__}: {exc}"
        _agents[agent_id]["status"] = "failed"
        _agents[agent_id]["error"] = error_msg
        logger.error("Agent %s failed: %s", agent_id[:8], error_msg, exc_info=True)

    finally:
        _agents[agent_id]["finished_at"] = datetime.now(timezone.utc).isoformat()

        # Check if the group is complete and notify
        group_id = _agents[agent_id].get("group_id")
        if group_id and _check_group_complete(group_id):
            try:
                await _notify_main_agent(group_id)
            except Exception:
                logger.exception("Failed to notify main agent for group %s", group_id)


_last_spawn_time: float = 0  # monotonic time of last successful spawn
_last_group_id: str | None = None  # group_id from the most recent spawn


async def _handle_spawn(
    app_state: Any,
    name: str,
    prompt: str,
    model: str | None = None,
    tools: list[str] | None = None,
    max_iterations: int = 15,
    agent_type: str | None = None,
    group_id: str | None = None,
) -> str:
    """Spawn a background agent."""
    global _last_spawn_time, _last_group_id
    import time

    now = time.monotonic()
    elapsed = now - _last_spawn_time

    # Auto-group: spawns within 5 seconds of each other share a group
    if elapsed < 5 and _last_group_id is not None and group_id is None:
        group_id = _last_group_id

    # Rate limit: max 1 spawn per 30 seconds (skip if part of a group batch)
    if elapsed < 30 and group_id is None:
        return json.dumps({
            "status": "rate_limited",
            "message": "An agent was just spawned. Do NOT spawn another. Reply to the user now.",
        })

    # Check concurrency limit
    running = sum(1 for a in _agents.values() if a["status"] == "running")
    if running >= _MAX_CONCURRENT:
        return json.dumps({
            "error": f"Max concurrent agents ({_MAX_CONCURRENT}) reached. Wait for one to finish.",
            "running": running,
        })

    # Check per-group cap
    if group_id is not None:
        group_count = sum(1 for a in _agents.values() if a.get("group_id") == group_id)
        if group_count >= _MAX_PER_GROUP:
            return json.dumps({
                "status": "group_full",
                "message": f"Max {_MAX_PER_GROUP} agents per group. Do NOT spawn more — reply to the user now.",
            })

    # Resolve agent type
    tools_filter = tools
    system_suffix: str | None = None
    if agent_type and agent_type in AGENT_TYPES:
        archetype = AGENT_TYPES[agent_type]
        if tools_filter is None:
            tools_filter = archetype["tools"]
        system_suffix = archetype["system_suffix"]

    # Assign group_id (create new one if not provided)
    resolved_group_id = group_id or str(uuid.uuid4())[:8]

    agent_id = str(uuid.uuid4())[:8]
    _agents[agent_id] = {
        "name": name,
        "prompt": prompt[:100],
        "status": "starting",
        "group_id": resolved_group_id,
        "agent_type": agent_type,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    # Mark spawn time and group for auto-grouping
    _last_spawn_time = now
    _last_group_id = resolved_group_id

    # Launch as background task
    asyncio.create_task(
        _run_agent(
            agent_id, prompt, app_state, model, tools_filter,
            max_iterations, system_suffix,
        ),
        name=f"agent-{agent_id}",
    )

    logger.info("Spawned agent %s: %s (group=%s)", agent_id, name, resolved_group_id)
    return json.dumps({
        "__stop_turn__": True,
        "reply": f"Got it — spawned background agent '{name}'. I'll send you the results when it's done.",
        "agent_id": agent_id,
        "group_id": resolved_group_id,
    })


async def _handle_list_agents() -> str:
    """List all agents and their status."""
    agents = []
    for aid, info in _agents.items():
        entry: dict[str, Any] = {
            "id": aid,
            "name": info.get("name"),
            "status": info.get("status"),
            "created_at": info.get("created_at"),
            "finished_at": info.get("finished_at"),
        }
        if "usage" in info:
            entry["usage"] = info["usage"]
        agents.append(entry)
    # Most recent first
    agents.sort(key=lambda a: a.get("created_at", ""), reverse=True)
    return json.dumps({"agents": agents[:20], "count": len(agents)})


# -- Registration ----------------------------------------------------------

SPAWN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "name": {
            "type": "string",
            "description": "Short name for the agent task",
        },
        "prompt": {
            "type": "string",
            "description": "The full task prompt for the background agent",
        },
        "model": {
            "type": "string",
            "description": "Model to use (optional, defaults to current model)",
        },
        "agent_type": {
            "type": "string",
            "enum": list(AGENT_TYPES.keys()),
            "description": (
                "Predefined agent archetype: 'researcher' (web search + scraping), "
                "'coder' (shell + files), 'analyst' (data analysis). "
                "Sets appropriate tools and system prompt automatically."
            ),
        },
        "tools": {
            "type": "array",
            "items": {"type": "string"},
            "description": "List of tool names the agent can use (optional, overrides agent_type tools)",
        },
        "max_iterations": {
            "type": "integer",
            "description": "Max tool loop iterations (default 15)",
            "default": 15,
        },
    },
    "required": ["name", "prompt"],
}

LIST_AGENTS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {},
}


def register(registry: Any, config: dict[str, Any], app_state: Any) -> None:
    """Register spawn_agent and list_agents tools."""

    async def _spawn_handler(
        name: str,
        prompt: str,
        model: str | None = None,
        tools: list[str] | None = None,
        max_iterations: int = 15,
        agent_type: str | None = None,
    ) -> str:
        return await _handle_spawn(
            app_state, name=name, prompt=prompt,
            model=model, tools=tools, max_iterations=max_iterations,
            agent_type=agent_type,
        )

    registry.register(
        name="spawn_agent",
        description=(
            "Spawn a background agent that works independently and reports results back to you. "
            "For multi-part requests, spawn MULTIPLE agents in parallel (one per subtask, max 3) "
            "in a SINGLE tool-call batch — they auto-group and results are delivered together. "
            "Use agent_type for specialization: 'researcher' (web search), 'coder' (shell/files), "
            "'analyst' (data analysis). Give each agent a focused, self-contained prompt. "
            "Do NOT spawn for simple questions or single tool calls — handle those directly."
        ),
        parameters_schema=SPAWN_SCHEMA,
        handler=_spawn_handler,
        run_in_executor=False,
    )

    async def _list_handler() -> str:
        return await _handle_list_agents()

    registry.register(
        name="list_agents",
        description="List all background agents, their status, and token usage.",
        parameters_schema=LIST_AGENTS_SCHEMA,
        handler=_list_handler,
        run_in_executor=False,
    )
