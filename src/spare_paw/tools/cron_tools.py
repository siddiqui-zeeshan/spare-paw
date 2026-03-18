"""Cron management tools — cron_create, cron_delete, cron_list.

These tools interact with the application's CronScheduler and SQLite database
to manage scheduled jobs.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from spare_paw.db import get_db

if TYPE_CHECKING:
    from spare_paw.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

# -- Schemas ---------------------------------------------------------------

CRON_CREATE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "name": {
            "type": "string",
            "description": "Human-readable name for the cron job",
        },
        "schedule": {
            "type": "string",
            "description": "Cron expression (e.g. '*/5 * * * *' for every 5 minutes)",
        },
        "prompt": {
            "type": "string",
            "description": "Prompt to send to the AI model when this cron fires",
        },
        "model": {
            "type": "string",
            "description": "Model to use (optional, defaults to cron_default)",
        },
        "tools_allowed": {
            "type": "array",
            "items": {"type": "string"},
            "description": "List of tool names this cron may use (optional, defaults to all)",
        },
        "once": {
            "type": "boolean",
            "description": "If true, the job fires once then auto-deletes. Use for reminders.",
            "default": False,
        },
    },
    "required": ["name", "schedule", "prompt"],
}

CRON_DELETE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "cron_id": {
            "type": "string",
            "description": "ID of the cron job to delete",
        },
    },
    "required": ["cron_id"],
}

CRON_EDIT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "cron_id": {
            "type": "string",
            "description": "ID of the cron job to edit",
        },
        "name": {
            "type": "string",
            "description": "New name (optional)",
        },
        "schedule": {
            "type": "string",
            "description": "New cron expression (optional)",
        },
        "prompt": {
            "type": "string",
            "description": "New prompt (optional)",
        },
        "model": {
            "type": "string",
            "description": "New model (optional)",
        },
    },
    "required": ["cron_id"],
}

CRON_LIST_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {},
}

# -- Handlers --------------------------------------------------------------


async def _handle_cron_create(
    app_state: Any,
    name: str,
    schedule: str,
    prompt: str,
    model: str | None = None,
    tools_allowed: list[str] | None = None,
    once: bool = False,
) -> str:
    """Create a new cron job, persist to DB, and schedule via CronScheduler."""
    # Resolve model slot names to actual model IDs
    if model:
        slot_names = {"default", "smart", "cron_default"}
        if model in slot_names:
            model = app_state.config.get(f"models.{model}") or None
        elif "/" not in model:
            # Doesn't look like a valid model ID — treat as null
            model = None

    cron_id = str(uuid.uuid4())[:8]
    now = datetime.now(timezone.utc).isoformat()
    tools_json = json.dumps(tools_allowed) if tools_allowed else None
    metadata_json = json.dumps({"once": True}) if once else None

    try:
        db = await get_db()
        await db.execute(
            """
            INSERT INTO cron_jobs (id, name, schedule, prompt, model, tools_allowed, enabled, created_at, metadata)
            VALUES (?, ?, ?, ?, ?, ?, 1, ?, ?)
            """,
            (cron_id, name, schedule, prompt, model, tools_json, now, metadata_json),
        )
        await db.commit()
    except Exception as exc:  # noqa: BLE001
        return json.dumps({"error": f"Database error: {exc}"})

    # Schedule via CronScheduler.
    try:
        await app_state.scheduler.add_job(
            cron_id, schedule, prompt, model, tools_allowed
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to add job to scheduler")
        return json.dumps(
            {"error": f"Job saved to DB but scheduler failed: {exc}", "id": cron_id}
        )

    next_run_dt = app_state.scheduler.get_next_run(cron_id)
    next_run = next_run_dt.isoformat() if next_run_dt else None

    logger.info("cron_create: id=%s name=%s schedule=%s", cron_id, name, schedule)
    return json.dumps(
        {
            "id": cron_id,
            "name": name,
            "schedule": schedule,
            "next_run": next_run,
            "model": model,
        }
    )


async def _handle_cron_delete(app_state: Any, cron_id: str) -> str:
    """Delete a cron job from the DB and CronScheduler."""
    db = await get_db()

    # Check existence.
    cursor = await db.execute(
        "SELECT id, name FROM cron_jobs WHERE id = ?", (cron_id,)
    )
    row = await cursor.fetchone()
    if row is None:
        return json.dumps({"error": f"Cron job not found: {cron_id}"})

    # Remove from DB.
    await db.execute("DELETE FROM cron_jobs WHERE id = ?", (cron_id,))
    await db.commit()

    # Remove from scheduler.
    await app_state.scheduler.remove_job(cron_id)

    logger.info("cron_delete: id=%s name=%s", cron_id, row[1])
    return json.dumps({"success": True, "id": cron_id, "name": row[1]})


async def _handle_cron_edit(
    app_state: Any,
    cron_id: str,
    name: str | None = None,
    schedule: str | None = None,
    prompt: str | None = None,
    model: str | None = None,
) -> str:
    """Edit an existing cron job. Only provided fields are updated."""
    db = await get_db()

    cursor = await db.execute(
        "SELECT id, name, schedule, prompt, model, tools_allowed FROM cron_jobs WHERE id = ?",
        (cron_id,),
    )
    row = await cursor.fetchone()
    if row is None:
        return json.dumps({"error": f"Cron job not found: {cron_id}"})

    # Build update fields
    updates: dict[str, Any] = {}
    if name is not None:
        updates["name"] = name
    if schedule is not None:
        updates["schedule"] = schedule
    if prompt is not None:
        updates["prompt"] = prompt
    if model is not None:
        updates["model"] = model

    if not updates:
        return json.dumps({"error": "No fields to update"})

    # Update DB
    set_clause = ", ".join(f"{k} = ?" for k in updates)
    values = list(updates.values()) + [cron_id]
    await db.execute(f"UPDATE cron_jobs SET {set_clause} WHERE id = ?", values)
    await db.commit()

    # If schedule or prompt changed, reschedule the job
    if schedule is not None or prompt is not None:
        new_schedule = schedule or row[2]
        new_prompt = prompt or row[3]
        new_model = model if model is not None else row[4]
        tools_allowed = None
        if row[5]:
            try:
                tools_allowed = json.loads(row[5])
            except (json.JSONDecodeError, TypeError):
                pass

        await app_state.scheduler.remove_job(cron_id)
        await app_state.scheduler.add_job(
            cron_id, new_schedule, new_prompt, new_model, tools_allowed
        )

    logger.info("cron_edit: id=%s updated=%s", cron_id, list(updates.keys()))
    return json.dumps({"success": True, "id": cron_id, "updated": list(updates.keys())})


async def _handle_cron_list(app_state: Any) -> str:
    """List all cron jobs with their next run times."""
    db = await get_db()

    cursor = await db.execute(
        """
        SELECT id, name, schedule, prompt, model, tools_allowed,
               enabled, created_at, last_run_at, last_result, last_error
        FROM cron_jobs
        ORDER BY created_at
        """
    )
    rows = await cursor.fetchall()

    jobs: list[dict[str, Any]] = []
    for row in rows:
        cron_id = row[0]

        # Get next run time from CronScheduler.
        next_run: str | None = None
        next_run_dt = app_state.scheduler.get_next_run(cron_id)
        if next_run_dt is not None:
            next_run = next_run_dt.isoformat()

        jobs.append(
            {
                "id": cron_id,
                "name": row[1],
                "schedule": row[2],
                "prompt": row[3][:100] + ("..." if len(row[3]) > 100 else ""),
                "model": row[4],
                "tools_allowed": json.loads(row[5]) if row[5] else None,
                "enabled": bool(row[6]),
                "created_at": row[7],
                "last_run_at": row[8],
                "last_result": (
                    row[9][:200] + ("..." if row[9] and len(row[9]) > 200 else "")
                    if row[9]
                    else None
                ),
                "last_error": row[10],
                "next_run": next_run,
            }
        )

    return json.dumps({"cron_jobs": jobs, "count": len(jobs)})


# -- Registration ----------------------------------------------------------


def register(
    registry: ToolRegistry,
    config: dict[str, Any],
    app_state: Any,
) -> None:
    """Register ``cron_create``, ``cron_edit``, ``cron_delete``, and ``cron_list``."""

    # -- cron_create -------------------------------------------------------
    async def _create_handler(
        name: str,
        schedule: str,
        prompt: str,
        model: str | None = None,
        tools_allowed: list[str] | None = None,
        once: bool = False,
    ) -> str:
        return await _handle_cron_create(
            app_state,
            name=name,
            schedule=schedule,
            prompt=prompt,
            model=model,
            tools_allowed=tools_allowed,
            once=once,
        )

    registry.register(
        name="cron_create",
        description=(
            "Create a scheduled task. Schedule uses cron expressions: "
            "'*/5 * * * *' (every 5 min), '0 10 * * *' (daily 10am), '0 */2 * * *' (every 2 hours). "
            "The prompt runs WITHOUT conversation context — make it self-contained with clear instructions. "
            "Set once=true for one-shot tasks that auto-delete after firing. "
            "Crons cannot create other crons or spawn agents."
        ),
        parameters_schema=CRON_CREATE_SCHEMA,
        handler=_create_handler,
        run_in_executor=False,
    )

    # -- cron_edit ---------------------------------------------------------
    async def _edit_handler(
        cron_id: str,
        name: str | None = None,
        schedule: str | None = None,
        prompt: str | None = None,
        model: str | None = None,
    ) -> str:
        return await _handle_cron_edit(
            app_state,
            cron_id=cron_id,
            name=name,
            schedule=schedule,
            prompt=prompt,
            model=model,
        )

    registry.register(
        name="cron_edit",
        description=(
            "Edit an existing scheduled task (cron job). "
            "Only the fields you provide will be updated."
        ),
        parameters_schema=CRON_EDIT_SCHEMA,
        handler=_edit_handler,
        run_in_executor=False,
    )

    # -- cron_delete -------------------------------------------------------
    async def _delete_handler(cron_id: str) -> str:
        return await _handle_cron_delete(app_state, cron_id=cron_id)

    registry.register(
        name="cron_delete",
        description="Delete a scheduled task (cron job) by its ID.",
        parameters_schema=CRON_DELETE_SCHEMA,
        handler=_delete_handler,
        run_in_executor=False,
    )

    # -- cron_list ---------------------------------------------------------
    async def _list_handler() -> str:
        return await _handle_cron_list(app_state)

    registry.register(
        name="cron_list",
        description="List all scheduled tasks (cron jobs) with their status and next run time.",
        parameters_schema=CRON_LIST_SCHEMA,
        handler=_list_handler,
        run_in_executor=False,
    )
