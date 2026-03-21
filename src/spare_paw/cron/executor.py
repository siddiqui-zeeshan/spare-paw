"""Cron job execution — runs a prompt through the model router and delivers results via Telegram.

Execution is semaphore-gated (via the OpenRouter client) so cron runs
don't race with user messages.  Results are NOT stored in conversation
memory — they are fire-and-forget messages to the owner.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from spare_paw.config import resolve_model
from spare_paw.core.prompt import build_system_prompt
from spare_paw.db import get_db
from spare_paw.router.tool_loop import run_tool_loop

if TYPE_CHECKING:
    from spare_paw.gateway import AppState

logger = logging.getLogger(__name__)


async def execute_cron(
    app_state: AppState,
    cron_id: str,
    prompt: str,
    model: str | None,
    tools_allowed: list[str] | None,
) -> None:
    """Execute a cron job: run the prompt through the tool loop and send the result to the owner.

    Steps:
        1. Resolve model (cron-specific → cron_default → default).
        2. Build messages with system prompt + user prompt.
        3. Get tool schemas, filtered by tools_allowed if set.
        4. Run the tool loop.
        5. Send result to owner via Telegram (chunked if needed).
        6. Update cron_jobs row with last_run_at and last_result/last_error.
    """
    now = datetime.now(timezone.utc).isoformat()

    try:
        # 1. Resolve model (per-job override → cron role → main_agent)
        resolved_model = model or resolve_model(app_state.config, "cron")

        # 2. Build system prompt (includes IDENTITY.md, USER.md, SYSTEM.md)
        system_prompt = await build_system_prompt(app_state.config)

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

        # 3. Get tool schemas
        tool_registry = app_state.config.get("_tool_registry")
        # The tool registry is stored on app_state directly in real usage.
        # Try the canonical attribute first, then fall back to config stash.
        registry = getattr(app_state, "tool_registry", None) or tool_registry

        tools: list[dict[str, Any]] = []
        if registry is not None:
            all_schemas = registry.get_schemas()
            if tools_allowed is not None:
                allowed_set = set(tools_allowed)
                tools = [
                    s
                    for s in all_schemas
                    if s.get("function", {}).get("name") in allowed_set
                ]
            else:
                tools = all_schemas

        # Crons cannot create other crons or spawn agents
        _blocked = {"cron_create", "cron_edit", "cron_delete", "spawn_agent", "list_agents"}
        tools = [s for s in tools if s.get("function", {}).get("name") not in _blocked]

        # 4. Get the OpenRouter client
        router_client = getattr(app_state, "router_client", None)
        if router_client is None:
            raise RuntimeError("OpenRouter client not available on app_state")

        # 5. Run the tool loop
        max_iterations = app_state.config.get("agent.max_tool_iterations", 20)
        result = await run_tool_loop(
            client=router_client,
            messages=messages,
            model=resolved_model,
            tools=tools,
            tool_registry=registry,
            max_iterations=max_iterations,
            executor=app_state.executor,
        )

        # 6. Send result to owner via backend
        backend = getattr(app_state, "backend", None)
        if backend is not None:
            await backend.send_text(result)

        # 7. Update DB — success
        await _update_cron_result(cron_id, now, result=result)
        logger.info("Cron %s executed successfully", cron_id)

    except Exception as exc:
        error_msg = f"{type(exc).__name__}: {exc}"
        logger.error("Cron %s failed: %s", cron_id, error_msg, exc_info=True)

        # Send error notification to owner
        try:
            backend = getattr(app_state, "backend", None)
            if backend is not None:
                await backend.send_text(
                    f"\u26a0\ufe0f Cron {cron_id} failed:\n{error_msg}",
                )
        except Exception:
            logger.exception("Failed to send cron error notification")

        # Update DB — error
        await _update_cron_result(cron_id, now, error=error_msg)

    finally:
        # Auto-delete one-shot crons regardless of success/failure
        await _maybe_delete_once(app_state, cron_id)


async def _update_cron_result(
    cron_id: str,
    run_time: str,
    result: str | None = None,
    error: str | None = None,
) -> None:
    """Update the cron_jobs row with execution results."""
    try:
        db = await get_db()
        if result is not None:
            await db.execute(
                "UPDATE cron_jobs SET last_run_at = ?, last_result = ?, last_error = NULL WHERE id = ?",
                (run_time, result[:10000], cron_id),
            )
        elif error is not None:
            await db.execute(
                "UPDATE cron_jobs SET last_run_at = ?, last_error = ? WHERE id = ?",
                (run_time, error[:10000], cron_id),
            )
        await db.commit()
    except Exception:
        logger.exception("Failed to update cron_jobs for %s", cron_id)


async def _maybe_delete_once(app_state: Any, cron_id: str) -> None:
    """Delete a cron job if it has once=true in its metadata."""
    try:
        db = await get_db()
        cursor = await db.execute(
            "SELECT metadata FROM cron_jobs WHERE id = ?", (cron_id,)
        )
        row = await cursor.fetchone()
        if row and row["metadata"]:
            import json
            meta = json.loads(row["metadata"])
            if meta.get("once"):
                await db.execute("DELETE FROM cron_jobs WHERE id = ?", (cron_id,))
                await db.commit()
                if app_state.scheduler:
                    await app_state.scheduler.remove_job(cron_id)
                logger.info("One-shot cron %s auto-deleted", cron_id)
    except Exception:
        logger.exception("Failed to auto-delete one-shot cron %s", cron_id)
