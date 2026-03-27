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

from dataclasses import dataclass, field

from spare_paw.config import resolve_model

logger = logging.getLogger(__name__)

# Active and completed agents
_agents: dict[str, dict[str, Any]] = {}
_MAX_CONCURRENT = 10
_MAX_PER_GROUP = 5  # max agents in a single group/batch

@dataclass
class DialogueChannel:
    agent_id: str
    original_request: str
    spawn_prompt: str
    to_main: asyncio.Queue
    max_rounds: int = 5
    round_count: int = 0
    history: list[dict] = field(default_factory=list)
    consumer_task: asyncio.Task | None = None
    closed: bool = False


# Active dialogue channels keyed by agent_id
_channels: dict[str, DialogueChannel] = {}

CONSULT_SYSTEM_PROMPT = (
    "You are the main agent coordinating background agents. A subagent is consulting "
    "you for guidance. You know the original user request and what this agent was "
    "tasked with. Answer concisely and directly. If the agent is on the wrong track, "
    "redirect it. If it needs information you don't have, say so."
)


async def _update_progress(channel: DialogueChannel, app_state: Any) -> None:
    """Edit the agent's progress message to show consult status. Stub until Task 7."""
    pass


async def _dialogue_consumer(channel: DialogueChannel, app_state: Any) -> None:
    """Consumer coroutine: receives questions from subagent, calls main-agent LLM, resolves Futures."""
    try:
        while not channel.closed:
            question, future = await channel.to_main.get()

            messages: list[dict[str, Any]] = [
                {"role": "system", "content": CONSULT_SYSTEM_PROMPT},
                {"role": "user", "content": f"Original request: {channel.original_request}"},
                {"role": "user", "content": f"Agent task: {channel.spawn_prompt}"},
            ]
            for entry in channel.history:
                messages.append(entry)
            messages.append({"role": "user", "content": question})

            model = resolve_model(app_state.config, "main_agent")
            response = await app_state.router_client.chat(messages, model)
            answer = response["choices"][0]["message"].get("content", "")

            channel.history.append({"role": "user", "content": question})
            channel.history.append({"role": "assistant", "content": answer})
            channel.round_count += 1

            if not future.done():
                future.set_result(answer)

            await _update_progress(channel, app_state)

    except asyncio.CancelledError:
        logger.info("Dialogue consumer for agent %s cancelled", channel.agent_id[:8])


def _cleanup_channel(agent_id: str) -> None:
    """Tear down a dialogue channel: cancel consumer, resolve pending futures."""
    channel = _channels.get(agent_id)
    if channel is None:
        return
    channel.closed = True
    if channel.consumer_task and not channel.consumer_task.done():
        channel.consumer_task.cancel()
    while not channel.to_main.empty():
        try:
            _, future = channel.to_main.get_nowait()
            if not future.done():
                future.set_result("Error: agent terminated, consult cancelled")
        except asyncio.QueueEmpty:
            break
    del _channels[agent_id]


_CONSULT_HEARTBEAT_INTERVAL = 15  # seconds


async def _consult_heartbeat(agent_id: str, future: asyncio.Future) -> None:
    """Tick last_activity while a consult Future is pending."""
    try:
        while not future.done():
            _agents[agent_id]["last_activity"] = datetime.now(timezone.utc)
            await asyncio.sleep(_CONSULT_HEARTBEAT_INTERVAL)
    except asyncio.CancelledError:
        pass


async def _handle_consult(agent_id: str, question: str) -> str:
    """Handle a consult_main tool call from a subagent."""
    channel = _channels.get(agent_id)
    if channel is None:
        return json.dumps({"error": "No dialogue channel for this agent"})

    if channel.round_count >= channel.max_rounds:
        return json.dumps({
            "error": f"Consultation limit reached ({channel.max_rounds}/{channel.max_rounds}). "
            "Continue with the information you have.",
        })

    if len(question) > 2000:
        return json.dumps({
            "error": f"Question too long ({len(question)} chars, max 2000). "
            "Summarize before consulting.",
        })

    if agent_id in _agents:
        _agents[agent_id]["last_activity"] = datetime.now(timezone.utc)

    future: asyncio.Future = asyncio.get_running_loop().create_future()
    await channel.to_main.put((question, future))

    hb_task = asyncio.create_task(
        _consult_heartbeat(agent_id, future),
        name=f"consult-hb-{agent_id}",
    )

    try:
        answer = await future
    finally:
        hb_task.cancel()
        try:
            await hb_task
        except asyncio.CancelledError:
            pass

    return answer


# Reference to the message queue — set by engine.py at startup
_message_queue: asyncio.Queue | None = None

# Reference to app_state — set during register()
_app_state: Any | None = None

# Watchdog settings
_WATCHDOG_INTERVAL = 30  # seconds between scans
_WATCHDOG_TIMEOUT = 180  # cancel after 3 minutes of inactivity
_watchdog_task: asyncio.Task | None = None

# ---------------------------------------------------------------------------
# Agent types / archetypes
# ---------------------------------------------------------------------------

_DEFAULT_AGENT_LIMITS: dict[str, int] = {"shell": 15, "web_search": 5}

AGENT_TYPES: dict[str, dict[str, Any]] = {
    "researcher": {
        "system_suffix": (
            "You are a research agent. Search thoroughly, use multiple sources, "
            "and cite URLs when possible. Focus on finding accurate, up-to-date information."
        ),
        "tools": ["tavily_search", "web_scrape", "shell", "files"],
        "tool_limits": {"web_search": 10, "tavily_search": 10, "shell": 10},
    },
    "coder": {
        "system_suffix": (
            "You are a coding agent. Write, test, and debug code. "
            "Use the shell to run commands and verify your work."
        ),
        "tools": ["shell", "files", "code"],
        "tool_limits": {"shell": 30, "web_search": 3},
    },
    "analyst": {
        "system_suffix": (
            "You are an analysis agent. Analyze data, produce summaries, "
            "and extract key insights. Be thorough but concise."
        ),
        "tools": ["files", "shell", "tavily_search"],
        "tool_limits": {"shell": 15, "web_search": 5, "tavily_search": 5},
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
    return all(a["status"] in ("completed", "failed", "timed_out") for a in group)


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

    # Delete ephemeral progress messages
    if _app_state is not None:
        backend = getattr(_app_state, "backend", None)
        if backend is not None and hasattr(type(backend), "delete_progress"):
            for _aid2, agent in group:
                msg_id = agent.get("progress_message_id")
                if msg_id is not None:
                    await backend.delete_progress(msg_id)

    if _message_queue is not None:
        await _message_queue.put(("agent_callback", synthetic_text))
        logger.info("Group %s: pushed callback with %d agent results", group_id, len(group))
    else:
        logger.error("Group %s completed but no message queue — results DROPPED", group_id)


def _on_agent_done(agent_id: str, task: asyncio.Task) -> None:
    """Done-callback: detect crashes that escaped _run_agent's try/except."""
    if task.cancelled():
        return
    exc = task.exception()
    if exc is None:
        return  # Normal completion — _run_agent already updated status
    error_msg = f"{type(exc).__name__}: {exc}"
    _agents[agent_id]["status"] = "failed"
    _agents[agent_id]["error"] = error_msg
    _agents[agent_id].setdefault(
        "finished_at", datetime.now(timezone.utc).isoformat()
    )
    logger.error("Agent %s crashed (done-callback): %s", agent_id[:8], error_msg)

    _cleanup_channel(agent_id)

    group_id = _agents[agent_id].get("group_id")
    if group_id and _check_group_complete(group_id):
        asyncio.create_task(
            _notify_main_agent(group_id),
            name=f"agent-notify-{group_id}",
        )


# ---------------------------------------------------------------------------
# Heartbeat watchdog
# ---------------------------------------------------------------------------

async def _watchdog_tick() -> None:
    """Single pass: cancel any stuck agents."""
    now = datetime.now(timezone.utc)
    for agent_id, info in list(_agents.items()):
        if info["status"] != "running":
            continue
        last = info.get("last_activity")
        if last is None:
            continue
        elapsed = (now - last).total_seconds()
        if elapsed > _WATCHDOG_TIMEOUT:
            task = info.get("task")
            if task is not None and not task.done():
                logger.warning(
                    "Watchdog: cancelling agent %s (no activity for %.0fs)",
                    agent_id[:8], elapsed,
                )
                task.cancel()


async def _watchdog_loop() -> None:
    """Background loop that runs _watchdog_tick periodically."""
    while True:
        await asyncio.sleep(_WATCHDOG_INTERVAL)
        try:
            await _watchdog_tick()
        except Exception:
            logger.exception("Watchdog tick failed")


def start_watchdog() -> None:
    """Start the watchdog background task."""
    global _watchdog_task
    if _watchdog_task is None or _watchdog_task.done():
        _watchdog_task = asyncio.create_task(_watchdog_loop(), name="agent-watchdog")


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
    tool_limits: dict[str, int] | None = None,
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

        # Resolve model (explicit → role-specific → main_agent)
        agent_type = _agents[agent_id].get("agent_type")
        resolved_model = model or resolve_model(
            app_state.config, agent_type or "main_agent"
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

        # Agents cannot spawn other agents or message the user directly
        _agent_tools = {"spawn_agent", "list_agents", "send_message", "send_file"}
        tool_schemas = [
            s for s in tool_schemas
            if s.get("function", {}).get("name") not in _agent_tools
        ]

        # Build messages
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

        # Heartbeat callback — updates last_activity on every tool event
        def _heartbeat(event: Any) -> None:
            _agents[agent_id]["last_activity"] = datetime.now(timezone.utc)

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
            tool_limits=tool_limits,
            on_event=_heartbeat,
        )

        _agents[agent_id]["status"] = "completed"
        _agents[agent_id]["result"] = result_text
        _agents[agent_id]["result_preview"] = result_text[:200]
        _agents[agent_id]["usage"] = usage
        logger.info("Agent %s completed", agent_id[:8])

    except asyncio.CancelledError:
        _agents[agent_id]["status"] = "timed_out"
        _agents[agent_id]["error"] = "Agent timed out: no activity for too long"
        logger.warning("Agent %s timed out (cancelled by watchdog)", agent_id[:8])

    except Exception as exc:
        error_msg = f"{type(exc).__name__}: {exc}"
        _agents[agent_id]["status"] = "failed"
        _agents[agent_id]["error"] = error_msg
        logger.error("Agent %s failed: %s", agent_id[:8], error_msg, exc_info=True)

    finally:
        _agents[agent_id]["finished_at"] = datetime.now(timezone.utc).isoformat()

        _cleanup_channel(agent_id)

        # Edit progress message with per-agent status
        progress_msg_id = _agents[agent_id].get("progress_message_id")
        if progress_msg_id is not None:
            backend = getattr(app_state, "backend", None)
            if backend is not None and hasattr(type(backend), "edit_progress"):
                group_id = _agents[agent_id].get("group_id")
                if group_id:
                    group = [a for a in _agents.values() if a.get("group_id") == group_id]
                    done = sum(1 for a in group if a["status"] in ("completed", "failed", "timed_out"))
                    total = len(group)
                    status_emoji = "\u2705" if _agents[agent_id]["status"] == "completed" else "\u274c"
                    try:
                        await backend.edit_progress(
                            progress_msg_id,
                            f"{status_emoji} {_agents[agent_id].get('name', 'agent')} done ({done}/{total})",
                        )
                    except Exception:
                        pass

        # Check if the group is complete and notify
        group_id = _agents[agent_id].get("group_id")
        if group_id and _check_group_complete(group_id):
            try:
                await _notify_main_agent(group_id)
            except Exception:
                logger.exception("Failed to notify main agent for group %s", group_id)


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
    agent_tool_limits: dict[str, int] | None = _DEFAULT_AGENT_LIMITS
    if agent_type and agent_type in AGENT_TYPES:
        archetype = AGENT_TYPES[agent_type]
        if tools_filter is None:
            tools_filter = archetype["tools"]
        system_suffix = archetype["system_suffix"]
        agent_tool_limits = archetype.get("tool_limits", _DEFAULT_AGENT_LIMITS)

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
        "last_activity": datetime.now(timezone.utc),
    }

    # Create dialogue channel
    original_request = getattr(app_state, "current_request", prompt)
    channel = DialogueChannel(
        agent_id=agent_id,
        original_request=original_request,
        spawn_prompt=prompt,
        to_main=asyncio.Queue(),
    )
    channel.consumer_task = asyncio.create_task(
        _dialogue_consumer(channel, app_state),
        name=f"dialogue-{agent_id}",
    )
    _channels[agent_id] = channel

    # Launch as background task with done-callback for crash detection
    task = asyncio.create_task(
        _run_agent(
            agent_id, prompt, app_state, model, tools_filter,
            max_iterations, system_suffix,
            tool_limits=agent_tool_limits,
        ),
        name=f"agent-{agent_id}",
    )
    _agents[agent_id]["task"] = task
    task.add_done_callback(lambda t, aid=agent_id: _on_agent_done(aid, t))

    # Send ephemeral progress message (Telegram-specific)
    backend = getattr(app_state, "backend", None)
    if backend is not None and hasattr(type(backend), "send_progress"):
        msg_id = await backend.send_progress(f"\u23f3 Working on: {name}...")
        _agents[agent_id]["progress_message_id"] = msg_id

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
        task = info.get("task")
        entry: dict[str, Any] = {
            "id": aid,
            "name": info.get("name"),
            "status": info.get("status"),
            "agent_type": info.get("agent_type"),
            "group_id": info.get("group_id"),
            "created_at": info.get("created_at"),
            "finished_at": info.get("finished_at"),
        }
        if info.get("error"):
            entry["error"] = info["error"]
        if info.get("result_preview"):
            entry["result_preview"] = info["result_preview"]
        if "usage" in info:
            entry["usage"] = info["usage"]
        if task is not None:
            entry["is_alive"] = not task.done()
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
    global _app_state
    _app_state = app_state

    async def _spawn_handler(
        name: str,
        prompt: str,
        model: str | None = None,
        tools: list[str] | None = None,
        max_iterations: int = 15,
        agent_type: str | None = None,
        group_id: str | None = None,
    ) -> str:
        return await _handle_spawn(
            app_state, name=name, prompt=prompt,
            model=model, tools=tools, max_iterations=max_iterations,
            agent_type=agent_type, group_id=group_id,
        )

    registry.register(
        name="spawn_agent",
        description=(
            "Spawn a background agent that works independently and reports results back to you. "
            "For multi-part requests, spawn MULTIPLE agents in parallel (one per subtask, max 3) "
            "in a SINGLE tool-call batch — batch-based grouping ensures results are delivered together. "
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
