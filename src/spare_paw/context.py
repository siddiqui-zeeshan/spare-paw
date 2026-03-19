"""Sliding window context manager with FTS5 search.

Stores every message in SQLite and assembles a token-budgeted sliding window
for each model call. Designed as a clean interface so LCM can replace this
module later without touching other components.

Interface:
    ingest(conversation_id, role, content, metadata) -> message_id
    assemble(conversation_id, system_prompt) -> list[dict]
    search(query, limit) -> list[dict]
    get_or_create_conversation() -> conversation_id
    new_conversation() -> conversation_id
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any

import tiktoken

from spare_paw.config import config
from spare_paw.db import get_db

logger = logging.getLogger(__name__)

# Cache the tiktoken encoder at module level
_encoder: tiktoken.Encoding | None = None


def _get_encoder() -> tiktoken.Encoding:
    """Return the cached cl100k_base encoder."""
    global _encoder
    if _encoder is None:
        _encoder = tiktoken.get_encoding("cl100k_base")
    return _encoder


def count_tokens(text: str) -> int:
    """Count tokens in text using cl100k_base."""
    return len(_get_encoder().encode(text))


async def ingest(
    conversation_id: str,
    role: str,
    content: str,
    metadata: dict[str, Any] | None = None,
) -> str:
    """Store a message in the database and return its ID.

    Token count is computed via tiktoken cl100k_base.
    """
    db = await get_db()
    message_id = uuid.uuid4().hex
    token_count = count_tokens(content)
    now = datetime.now(timezone.utc).isoformat()
    metadata_json = json.dumps(metadata) if metadata else None

    await db.execute(
        """INSERT INTO messages (id, conversation_id, role, content, token_count, created_at, metadata)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (message_id, conversation_id, role, content, token_count, now, metadata_json),
    )
    # Update conversation last_message_at
    await db.execute(
        "UPDATE conversations SET last_message_at = ? WHERE id = ?",
        (now, conversation_id),
    )
    await db.commit()

    logger.debug(
        "Ingested message %s (role=%s, tokens=%d) into conversation %s",
        message_id,
        role,
        token_count,
        conversation_id,
    )
    return message_id


async def assemble(
    conversation_id: str,
    system_prompt: str,
) -> list[dict[str, str]]:
    """Assemble a token-budgeted context window for the model.

    1. Fetch the last max_messages for this conversation, ordered by created_at.
    2. Walk backwards from newest, accumulating token counts.
    3. Stop when token_budget * safety_margin is exceeded.
    4. Prepend the system prompt.
    5. Return as OpenAI-format message list.
    """
    db = await get_db()
    max_messages = config.get("context.max_messages", 64)
    token_budget = config.get("context.token_budget", 120000)
    safety_margin = config.get("context.safety_margin", 0.85)
    effective_budget = int(token_budget * safety_margin)

    # Fetch last N messages ordered by time ascending
    async with db.execute(
        """SELECT id, role, content, token_count, metadata
           FROM messages
           WHERE conversation_id = ?
           ORDER BY created_at DESC
           LIMIT ?""",
        (conversation_id, max_messages),
    ) as cursor:
        rows = await cursor.fetchall()

    # rows are newest-first; walk backwards accumulating tokens
    system_tokens = count_tokens(system_prompt)
    remaining_budget = effective_budget - system_tokens
    selected: list[dict[str, str]] = []

    for row in rows:
        token_count = row["token_count"]
        if remaining_budget - token_count < 0:
            break
        remaining_budget -= token_count

        msg: dict[str, str] = {
            "role": row["role"],
            "content": row["content"],
        }
        # Restore tool_call metadata if present
        if row["metadata"]:
            meta = json.loads(row["metadata"])
            if "tool_call_id" in meta:
                msg["tool_call_id"] = meta["tool_call_id"]
            if "tool_calls" in meta:
                msg["tool_calls"] = meta["tool_calls"]

        selected.append(msg)

    # Reverse to chronological order
    selected.reverse()

    # Prepend system prompt
    messages = [{"role": "system", "content": system_prompt}]

    # Inject summary nodes (compressed history) between system prompt and fresh messages
    try:
        async with db.execute(
            """SELECT content FROM summary_nodes
               WHERE conversation_id = ? AND parent_id IS NULL
               ORDER BY created_at ASC""",
            (conversation_id,),
        ) as cursor:
            summary_rows = await cursor.fetchall()

        for srow in summary_rows:
            messages.append({
                "role": "system",
                "content": f"[Compressed History]\n{srow['content']}",
            })
    except Exception:
        # summary_nodes table may not exist in older schemas
        pass

    messages.extend(selected)

    logger.debug(
        "Assembled %d messages (%d tokens used of %d budget) for conversation %s",
        len(selected),
        effective_budget - remaining_budget,
        effective_budget,
        conversation_id,
    )
    return messages


async def search(query: str, limit: int = 10) -> list[dict[str, Any]]:
    """Full-text search over message content via FTS5.

    Returns list of dicts with id, conversation_id, role, content, created_at.
    """
    db = await get_db()
    async with db.execute(
        """SELECT m.id, m.conversation_id, m.role, m.content, m.created_at
           FROM messages_fts fts
           JOIN messages m ON m.rowid = fts.rowid
           WHERE messages_fts MATCH ?
           ORDER BY rank
           LIMIT ?""",
        (query, limit),
    ) as cursor:
        rows = await cursor.fetchall()

    return [
        {
            "id": row["id"],
            "conversation_id": row["conversation_id"],
            "role": row["role"],
            "content": row["content"],
            "created_at": row["created_at"],
        }
        for row in rows
    ]


async def get_or_create_conversation() -> str:
    """Return the most recent conversation ID, or create one if none exists."""
    db = await get_db()
    async with db.execute(
        "SELECT id FROM conversations ORDER BY created_at DESC LIMIT 1"
    ) as cursor:
        row = await cursor.fetchone()

    if row:
        return row["id"]
    return await new_conversation()


async def new_conversation() -> str:
    """Create a new conversation and return its ID."""
    db = await get_db()
    conversation_id = uuid.uuid4().hex
    now = datetime.now(timezone.utc).isoformat()

    await db.execute(
        "INSERT INTO conversations (id, created_at) VALUES (?, ?)",
        (conversation_id, now),
    )
    await db.commit()
    logger.info("Created new conversation %s", conversation_id)
    return conversation_id


# ---------------------------------------------------------------------------
# LCM compaction engine
# ---------------------------------------------------------------------------


async def compact(
    conversation_id: str,
    router_client: Any,
    model: str,
) -> None:
    """Run LCM compaction on a conversation.

    1. Count total messages.
    2. If total <= fresh_tail_count, return early.
    3. Find messages not in the fresh tail and not already covered by summaries.
    4. Chunk them and create leaf summaries.
    5. If enough uncondensed leaves exist, condense them.
    """
    db = await get_db()
    fresh_tail_count = config.get("context.fresh_tail_count", 32)
    leaf_chunk_size = config.get("context.leaf_chunk_size", 8)

    # 1. Count total messages
    async with db.execute(
        "SELECT COUNT(*) FROM messages WHERE conversation_id = ?",
        (conversation_id,),
    ) as cur:
        total = (await cur.fetchone())[0]

    # 2. If not enough messages, nothing to compact
    if total <= fresh_tail_count:
        return

    # 3. Find messages NOT in the fresh tail
    async with db.execute(
        """SELECT id, role, content FROM messages
           WHERE conversation_id = ?
           ORDER BY created_at ASC""",
        (conversation_id,),
    ) as cur:
        all_msgs = await cur.fetchall()

    # Messages outside the fresh tail (oldest ones)
    stale_msgs = all_msgs[: len(all_msgs) - fresh_tail_count]

    # Find message IDs already covered by existing summary nodes
    async with db.execute(
        "SELECT source_msg_ids FROM summary_nodes WHERE conversation_id = ? AND depth = 0",
        (conversation_id,),
    ) as cur:
        rows = await cur.fetchall()

    covered_ids: set[str] = set()
    for row in rows:
        covered_ids.update(json.loads(row["source_msg_ids"]))

    # Filter to uncovered messages only
    uncovered = [
        {"id": m["id"], "role": m["role"], "content": m["content"]}
        for m in stale_msgs
        if m["id"] not in covered_ids
    ]

    if not uncovered:
        # Still check if condensation is needed
        await _condense_summaries(conversation_id, router_client, model)
        return

    # 4. Chunk and create leaf summaries
    for i in range(0, len(uncovered), leaf_chunk_size):
        chunk = uncovered[i : i + leaf_chunk_size]
        await _create_leaf_summary(conversation_id, chunk, router_client, model)

    # 5. Condense if enough uncondensed leaves
    await _condense_summaries(conversation_id, router_client, model)


async def _create_leaf_summary(
    conversation_id: str,
    messages: list[dict],
    router_client: Any,
    model: str,
) -> str:
    """Summarize a chunk of messages into a leaf summary node (depth=0).

    Returns the new node_id.
    """
    # Build the summarization prompt
    formatted = "\n".join(
        f"{m['role']}: {m['content']}" for m in messages
    )
    prompt = (
        "Summarize the following conversation messages concisely, "
        "preserving key facts, decisions, and context:\n\n" + formatted
    )

    # Call the LLM
    response = await router_client.chat(
        messages=[{"role": "user", "content": prompt}],
        model=model,
    )
    summary_text = response["choices"][0]["message"]["content"]

    # Store in DB
    db = await get_db()
    node_id = uuid.uuid4().hex
    token_count = count_tokens(summary_text)
    now = datetime.now(timezone.utc).isoformat()
    source_ids = json.dumps([m["id"] for m in messages])

    await db.execute(
        """INSERT INTO summary_nodes
           (id, conversation_id, parent_id, depth, content, token_count, source_msg_ids, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (node_id, conversation_id, None, 0, summary_text, token_count, source_ids, now),
    )
    await db.commit()

    logger.debug(
        "Created leaf summary %s (%d tokens) for conversation %s",
        node_id, token_count, conversation_id,
    )
    return node_id


async def _condense_summaries(
    conversation_id: str,
    router_client: Any,
    model: str,
) -> None:
    """Condense uncondensed leaf nodes into a higher-level summary (depth=1).

    Only runs when there are >= condensed_min_fanout orphan leaf nodes.
    """
    db = await get_db()
    condensed_min_fanout = config.get("context.condensed_min_fanout", 4)

    # Fetch orphan leaf nodes (depth=0, no parent)
    async with db.execute(
        """SELECT id, content FROM summary_nodes
           WHERE conversation_id = ? AND depth = 0 AND parent_id IS NULL
           ORDER BY created_at ASC""",
        (conversation_id,),
    ) as cur:
        leaves = await cur.fetchall()

    if len(leaves) < condensed_min_fanout:
        return

    # Build condensation prompt
    formatted = "\n\n".join(
        f"Summary {i + 1}: {leaf['content']}" for i, leaf in enumerate(leaves)
    )
    prompt = (
        "Condense the following conversation summaries into a single, "
        "higher-level summary preserving all key facts and context:\n\n" + formatted
    )

    # Call the LLM
    response = await router_client.chat(
        messages=[{"role": "user", "content": prompt}],
        model=model,
    )
    condensed_text = response["choices"][0]["message"]["content"]

    # Store condensed node
    node_id = uuid.uuid4().hex
    token_count = count_tokens(condensed_text)
    now = datetime.now(timezone.utc).isoformat()
    leaf_ids = [leaf["id"] for leaf in leaves]
    source_ids = json.dumps(leaf_ids)

    await db.execute(
        """INSERT INTO summary_nodes
           (id, conversation_id, parent_id, depth, content, token_count, source_msg_ids, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (node_id, conversation_id, None, 1, condensed_text, token_count, source_ids, now),
    )

    # Update leaf nodes to point to the new parent
    for leaf_id in leaf_ids:
        await db.execute(
            "UPDATE summary_nodes SET parent_id = ? WHERE id = ?",
            (node_id, leaf_id),
        )

    await db.commit()

    logger.debug(
        "Condensed %d leaves into node %s (depth=1) for conversation %s",
        len(leaves), node_id, conversation_id,
    )
