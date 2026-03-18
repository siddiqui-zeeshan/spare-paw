"""Subagent spawn tool — run background agents with their own tool loops.

A spawned agent gets its own message context, runs independently, and
delivers results via send_message. Does not enter conversation memory.
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


async def _run_agent(
    agent_id: str,
    prompt: str,
    app_state: Any,
    model: str | None,
    tools_filter: list[str] | None,
    max_iterations: int,
) -> None:
    """Background task that runs a self-contained tool loop and sends the result."""
    _agents[agent_id]["status"] = "running"
    _agents[agent_id]["started_at"] = datetime.now(timezone.utc).isoformat()

    try:
        from spare_paw.bot.handler import _build_system_prompt
        from spare_paw.router.tool_loop import run_tool_loop

        # Build system prompt
        system_prompt = await _build_system_prompt(app_state.config)

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

        # Run tool loop
        result = await run_tool_loop(
            client=app_state.router_client,
            messages=messages,
            model=resolved_model,
            tools=tool_schemas,
            tool_registry=app_state.tool_registry,
            max_iterations=max_iterations,
            executor=app_state.executor,
        )

        # Send result to owner
        owner_id = app_state.config.get("telegram.owner_id")
        if owner_id and app_state.application:
            bot = app_state.application.bot
            text = result
            while text:
                chunk = text[:4096]
                text = text[4096:]
                await bot.send_message(chat_id=owner_id, text=chunk)

        _agents[agent_id]["status"] = "completed"
        _agents[agent_id]["result_preview"] = result[:200]
        logger.info("Agent %s completed", agent_id[:8])

    except Exception as exc:
        error_msg = f"{type(exc).__name__}: {exc}"
        _agents[agent_id]["status"] = "failed"
        _agents[agent_id]["error"] = error_msg
        logger.error("Agent %s failed: %s", agent_id[:8], error_msg, exc_info=True)

        # Notify owner of failure
        try:
            owner_id = app_state.config.get("telegram.owner_id")
            if owner_id and app_state.application:
                bot = app_state.application.bot
                await bot.send_message(
                    chat_id=owner_id,
                    text=f"Agent {agent_id[:8]} failed: {error_msg}",
                )
        except Exception:
            logger.exception("Failed to send agent error notification")

    finally:
        _agents[agent_id]["finished_at"] = datetime.now(timezone.utc).isoformat()


_last_spawn_time: float = 0  # monotonic time of last successful spawn


async def _handle_spawn(
    app_state: Any,
    name: str,
    prompt: str,
    model: str | None = None,
    tools: list[str] | None = None,
    max_iterations: int = 15,
) -> str:
    """Spawn a background agent."""
    global _last_spawn_time
    import time

    # Rate limit: max 1 spawn per 30 seconds
    now = time.monotonic()
    if now - _last_spawn_time < 30:
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

    agent_id = str(uuid.uuid4())[:8]
    _agents[agent_id] = {
        "name": name,
        "prompt": prompt[:100],
        "status": "starting",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    # Mark spawn time before launching
    _last_spawn_time = now

    # Launch as background task
    asyncio.create_task(
        _run_agent(agent_id, prompt, app_state, model, tools, max_iterations),
        name=f"agent-{agent_id}",
    )

    logger.info("Spawned agent %s: %s", agent_id, name)
    return json.dumps({
        "__stop_turn__": True,
        "reply": f"Got it — spawned background agent '{name}'. I'll send you the results when it's done.",
    })


async def _handle_list_agents() -> str:
    """List all agents and their status."""
    agents = []
    for aid, info in _agents.items():
        agents.append({
            "id": aid,
            "name": info.get("name"),
            "status": info.get("status"),
            "created_at": info.get("created_at"),
            "finished_at": info.get("finished_at"),
        })
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
        "tools": {
            "type": "array",
            "items": {"type": "string"},
            "description": "List of tool names the agent can use (optional, defaults to all)",
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
    ) -> str:
        return await _handle_spawn(
            app_state, name=name, prompt=prompt,
            model=model, tools=tools, max_iterations=max_iterations,
        )

    registry.register(
        name="spawn_agent",
        description=(
            "Spawn a background agent for tasks requiring extensive research (3+ web searches), "
            "comparisons, or long analysis. The agent works independently and sends results via "
            "Telegram when done. Do NOT spawn for simple questions, quick lookups, or single "
            "tool calls — handle those directly."
        ),
        parameters_schema=SPAWN_SCHEMA,
        handler=_spawn_handler,
        run_in_executor=False,
    )

    async def _list_handler() -> str:
        return await _handle_list_agents()

    registry.register(
        name="list_agents",
        description="List all background agents and their status.",
        parameters_schema=LIST_AGENTS_SCHEMA,
        handler=_list_handler,
        run_in_executor=False,
    )
