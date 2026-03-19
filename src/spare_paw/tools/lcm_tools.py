"""LCM (Lossless Context Management) tools — lcm_grep, lcm_expand, lcm_describe.

Provide search, expansion, and description capabilities over the DAG-based
summary_nodes and raw messages tables.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from spare_paw.context import count_tokens
from spare_paw.db import get_db

logger = logging.getLogger(__name__)

# -- Schemas ---------------------------------------------------------------

LCM_GREP_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": "FTS5 search query to find in messages and summaries",
        },
        "limit": {
            "type": "integer",
            "description": "Maximum number of results to return (default 10)",
        },
    },
    "required": ["query"],
}

LCM_EXPAND_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "summary_id": {
            "type": "string",
            "description": "The ID of the summary node to expand",
        },
        "max_tokens": {
            "type": "integer",
            "description": "Maximum tokens to return (default 4000)",
        },
    },
    "required": ["summary_id"],
}

LCM_DESCRIBE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "conversation_id": {
            "type": "string",
            "description": "Filter by conversation ID (optional)",
        },
        "start_time": {
            "type": "string",
            "description": "ISO 8601 start time for the range (optional)",
        },
        "end_time": {
            "type": "string",
            "description": "ISO 8601 end time for the range (optional)",
        },
    },
}

# -- Handlers --------------------------------------------------------------


async def _handle_lcm_grep(query: str, limit: int = 10) -> str:
    """Search both messages and summary_nodes via FTS5."""
    db = await get_db()
    results: list[dict[str, Any]] = []
    seen: set[str] = set()

    # Search messages_fts
    async with db.execute(
        """SELECT m.id, m.role, m.content, m.created_at
           FROM messages_fts fts
           JOIN messages m ON m.rowid = fts.rowid
           WHERE messages_fts MATCH ?
           ORDER BY rank
           LIMIT ?""",
        (query, limit),
    ) as cursor:
        rows = await cursor.fetchall()

    for row in rows:
        key = row["id"]
        if key not in seen:
            seen.add(key)
            results.append({
                "source": "message",
                "content": row["content"],
                "role": row["role"],
                "created_at": row["created_at"],
            })

    # Search summary_nodes_fts
    async with db.execute(
        """SELECT s.id, s.content, s.created_at
           FROM summary_nodes_fts fts
           JOIN summary_nodes s ON s.rowid = fts.rowid
           WHERE summary_nodes_fts MATCH ?
           ORDER BY rank
           LIMIT ?""",
        (query, limit),
    ) as cursor:
        rows = await cursor.fetchall()

    for row in rows:
        key = row["id"]
        if key not in seen:
            seen.add(key)
            results.append({
                "source": "summary",
                "content": row["content"],
                "role": "summary",
                "created_at": row["created_at"],
            })

    return json.dumps({"results": results})


async def _handle_lcm_expand(summary_id: str, max_tokens: int = 4000) -> str:
    """Return the original messages that a summary was built from, capped at max_tokens."""
    db = await get_db()

    # Fetch the summary node
    async with db.execute(
        "SELECT * FROM summary_nodes WHERE id = ?",
        (summary_id,),
    ) as cursor:
        node = await cursor.fetchone()

    if node is None:
        return json.dumps({"error": f"Summary node not found: {summary_id}"})

    source_msg_ids = json.loads(node["source_msg_ids"])
    total_source_messages = len(source_msg_ids)

    if not source_msg_ids:
        return json.dumps({
            "summary_id": summary_id,
            "messages": [],
            "truncated": False,
            "total_source_messages": 0,
        })

    # Fetch messages by IDs, ordered by created_at
    placeholders = ",".join("?" for _ in source_msg_ids)
    async with db.execute(
        f"""SELECT role, content, created_at
            FROM messages
            WHERE id IN ({placeholders})
            ORDER BY created_at""",
        source_msg_ids,
    ) as cursor:
        rows = await cursor.fetchall()

    # Walk through messages, accumulating tokens
    messages: list[dict[str, str]] = []
    token_total = 0
    truncated = False

    for row in rows:
        tokens = count_tokens(row["content"])
        if token_total + tokens > max_tokens:
            truncated = True
            break
        token_total += tokens
        messages.append({
            "role": row["role"],
            "content": row["content"],
            "created_at": row["created_at"],
        })

    return json.dumps({
        "summary_id": summary_id,
        "messages": messages,
        "truncated": truncated,
        "total_source_messages": total_source_messages,
    })


async def _handle_lcm_describe(
    conversation_id: str | None = None,
    start_time: str | None = None,
    end_time: str | None = None,
) -> str:
    """Return summaries covering a time range."""
    db = await get_db()

    conditions: list[str] = []
    params: list[str] = []

    if conversation_id is not None:
        conditions.append("conversation_id = ?")
        params.append(conversation_id)
    if start_time is not None:
        conditions.append("created_at >= ?")
        params.append(start_time)
    if end_time is not None:
        conditions.append("created_at <= ?")
        params.append(end_time)

    where_clause = " AND ".join(conditions) if conditions else "1=1"

    async with db.execute(
        f"""SELECT id, conversation_id, parent_id, depth, content,
                   token_count, source_msg_ids, created_at
            FROM summary_nodes
            WHERE {where_clause}
            ORDER BY created_at""",
        params,
    ) as cursor:
        rows = await cursor.fetchall()

    summaries = [
        {
            "id": row["id"],
            "conversation_id": row["conversation_id"],
            "parent_id": row["parent_id"],
            "depth": row["depth"],
            "content": row["content"],
            "token_count": row["token_count"],
            "source_msg_ids": json.loads(row["source_msg_ids"]),
            "created_at": row["created_at"],
        }
        for row in rows
    ]

    return json.dumps({"summaries": summaries, "count": len(summaries)})


# -- Registration ----------------------------------------------------------


def register(registry: Any, config: dict[str, Any]) -> None:
    """Register LCM tools."""
    registry.register(
        name="lcm_grep",
        description=(
            "Search conversation history and compressed summaries for a query. "
            "Returns results from both raw messages and summary nodes."
        ),
        parameters_schema=LCM_GREP_SCHEMA,
        handler=_handle_lcm_grep,
        run_in_executor=False,
    )

    registry.register(
        name="lcm_expand",
        description=(
            "Expand a compressed history summary to see the original messages "
            "it was built from. Use when you need more detail than a summary provides."
        ),
        parameters_schema=LCM_EXPAND_SCHEMA,
        handler=_handle_lcm_expand,
        run_in_executor=False,
    )

    registry.register(
        name="lcm_describe",
        description="Get summaries of conversation history for a given time range.",
        parameters_schema=LCM_DESCRIBE_SCHEMA,
        handler=_handle_lcm_describe,
        run_in_executor=False,
    )
