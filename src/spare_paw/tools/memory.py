"""Persistent memory tools — remember, recall, forget_memory, list_memories.

Memories survive /forget and persist across conversations. They are stored
in SQLite with FTS5 for search. All active memories are also injected into
the system prompt on every turn.
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

REMEMBER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "key": {
            "type": "string",
            "description": "Short label for the memory (e.g. 'wifi_password', 'mom_birthday')",
        },
        "value": {
            "type": "string",
            "description": "The information to remember",
        },
    },
    "required": ["key", "value"],
}

RECALL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": "Search query to find relevant memories",
        },
    },
    "required": ["query"],
}

FORGET_MEMORY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "key": {
            "type": "string",
            "description": "Key of the memory to delete",
        },
    },
    "required": ["key"],
}

LIST_MEMORIES_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {},
}

# -- Handlers --------------------------------------------------------------


async def _handle_remember(key: str, value: str) -> str:
    """Store or update a memory."""
    db = await get_db()
    now = datetime.now(timezone.utc).isoformat()

    # Check if key already exists — update if so
    cursor = await db.execute("SELECT id FROM memories WHERE key = ?", (key,))
    row = await cursor.fetchone()

    if row:
        await db.execute(
            "UPDATE memories SET value = ?, updated_at = ? WHERE key = ?",
            (value, now, key),
        )
        await db.commit()
        logger.info("memory updated: %s", key)
        return json.dumps({"status": "updated", "key": key})
    else:
        mem_id = str(uuid.uuid4())[:8]
        await db.execute(
            "INSERT INTO memories (id, key, value, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
            (mem_id, key, value, now, now),
        )
        await db.commit()
        logger.info("memory created: %s", key)
        return json.dumps({"status": "created", "key": key, "id": mem_id})


async def _handle_recall(query: str) -> str:
    """Search memories via FTS5."""
    db = await get_db()
    cursor = await db.execute(
        """
        SELECT m.key, m.value, m.updated_at
        FROM memories_fts f
        JOIN memories m ON m.rowid = f.rowid
        WHERE memories_fts MATCH ?
        ORDER BY rank
        LIMIT 10
        """,
        (query,),
    )
    rows = await cursor.fetchall()

    if not rows:
        return json.dumps({"results": [], "query": query})

    results = [
        {"key": r["key"], "value": r["value"], "updated_at": r["updated_at"]}
        for r in rows
    ]
    return json.dumps({"results": results, "query": query})


async def _handle_forget_memory(key: str) -> str:
    """Delete a memory by key."""
    db = await get_db()
    cursor = await db.execute("SELECT id FROM memories WHERE key = ?", (key,))
    row = await cursor.fetchone()

    if not row:
        return json.dumps({"error": f"No memory found with key: {key}"})

    await db.execute("DELETE FROM memories WHERE key = ?", (key,))
    await db.commit()
    logger.info("memory deleted: %s", key)
    return json.dumps({"status": "deleted", "key": key})


async def _handle_list_memories() -> str:
    """List all memories."""
    db = await get_db()
    cursor = await db.execute(
        "SELECT key, value, updated_at FROM memories ORDER BY updated_at DESC"
    )
    rows = await cursor.fetchall()
    memories = [
        {"key": r["key"], "value": r["value"], "updated_at": r["updated_at"]}
        for r in rows
    ]
    return json.dumps({"memories": memories, "count": len(memories)})


async def get_all_memories() -> list[dict[str, str]]:
    """Return all memories as a list of dicts. Used for system prompt injection."""
    db = await get_db()
    cursor = await db.execute(
        "SELECT key, value FROM memories ORDER BY key"
    )
    rows = await cursor.fetchall()
    return [{"key": r["key"], "value": r["value"]} for r in rows]


# -- Registration ----------------------------------------------------------


def register(registry: ToolRegistry, config: dict[str, Any]) -> None:
    """Register memory tools."""
    registry.register(
        name="remember",
        description=(
            "Save important information to persistent memory that survives /forget. "
            "Use for: names, preferences, passwords, birthdays, addresses, account details, "
            "recurring preferences. Do NOT save: transient conversation details, things easily "
            "searchable, or one-time instructions."
        ),
        parameters_schema=REMEMBER_SCHEMA,
        handler=_handle_remember,
        run_in_executor=False,
    )

    registry.register(
        name="recall",
        description=(
            "Search persistent memory. All memories are already in your system prompt — "
            "only use this tool if you need to search for something specific that might "
            "not be in the current prompt window."
        ),
        parameters_schema=RECALL_SCHEMA,
        handler=_handle_recall,
        run_in_executor=False,
    )

    registry.register(
        name="forget_memory",
        description="Delete a specific memory by its key.",
        parameters_schema=FORGET_MEMORY_SCHEMA,
        handler=_handle_forget_memory,
        run_in_executor=False,
    )

    registry.register(
        name="list_memories",
        description="List all saved memories.",
        parameters_schema=LIST_MEMORIES_SCHEMA,
        handler=_handle_list_memories,
        run_in_executor=False,
    )
