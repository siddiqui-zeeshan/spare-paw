"""Tests for LCM (Lossless Context Management) DAG-based memory system.

Covers schema v3 (summary_nodes table + FTS), compaction engine, modified
assemble() with summary injection, and LCM tools (lcm_grep, lcm_expand).
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio
import aiosqlite

import spare_paw.db as db_mod
from spare_paw.context import ingest, assemble, new_conversation


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture()
async def _init_db(tmp_path):
    """Create a temp DB with all schemas (v1 + v2 + v3) applied."""
    db_path = tmp_path / "test.db"

    conn = await aiosqlite.connect(str(db_path))
    conn.row_factory = aiosqlite.Row
    await conn.execute("PRAGMA journal_mode = WAL")
    await conn.execute("PRAGMA foreign_keys = ON")

    # Patch the singleton
    db_mod._connection = conn

    # Apply all schemas
    await conn.executescript(db_mod.SCHEMA_V1)
    await conn.executescript(db_mod.SCHEMA_V2)
    await conn.executescript(db_mod.SCHEMA_V3)
    await conn.execute(f"PRAGMA user_version = {db_mod.CURRENT_SCHEMA_VERSION}")
    await conn.commit()

    yield conn

    await conn.close()
    db_mod._connection = None


async def _insert_summary_node(
    conn,
    conversation_id: str,
    content: str,
    *,
    node_id: str | None = None,
    parent_id: str | None = None,
    depth: int = 0,
    token_count: int = 10,
    source_msg_ids: list[str] | None = None,
) -> str:
    """Helper to insert a summary_node directly."""
    node_id = node_id or uuid.uuid4().hex
    now = datetime.now(timezone.utc).isoformat()
    source_ids_json = json.dumps(source_msg_ids or [])

    await conn.execute(
        """INSERT INTO summary_nodes
           (id, conversation_id, parent_id, depth, content, token_count, source_msg_ids, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (node_id, conversation_id, parent_id, depth, content, token_count, source_ids_json, now),
    )
    await conn.commit()
    return node_id


# ---------------------------------------------------------------------------
# Schema v3 tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_schema_v3_creates_summary_nodes_table(_init_db):
    """After init_db with v3, summary_nodes table should exist."""
    conn = _init_db

    async with conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    ) as cur:
        tables = {row[0] for row in await cur.fetchall()}

    assert "summary_nodes" in tables


@pytest.mark.asyncio
async def test_summary_nodes_fts(_init_db):
    """Inserting into summary_nodes should populate the FTS index via triggers."""
    conn = _init_db
    conv_id = await new_conversation()

    await _insert_summary_node(
        conn, conv_id, "The user discussed quantum computing and entanglement"
    )

    async with conn.execute(
        "SELECT content FROM summary_nodes_fts WHERE summary_nodes_fts MATCH 'quantum'"
    ) as cur:
        rows = await cur.fetchall()

    assert len(rows) == 1
    assert "quantum" in rows[0][0]


# ---------------------------------------------------------------------------
# Compaction tests
# ---------------------------------------------------------------------------


def _make_mock_router_client(summary_text: str = "This is a summary.") -> MagicMock:
    """Create a mock router client that returns a canned summary response (dict format)."""
    client = MagicMock()
    mock_response = {
        "choices": [{"message": {"content": summary_text}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }
    client.chat = AsyncMock(return_value=mock_response)
    return client


@pytest.mark.asyncio
async def test_compact_creates_leaf_summaries(_init_db):
    """When enough messages exist beyond the fresh tail, compact() creates leaf nodes."""
    from spare_paw.config import config as config_mod
    from spare_paw.context import compact

    # Use a small fresh tail so 30 messages trigger compaction
    config_mod.set_override("context.fresh_tail_count", 10)

    conn = _init_db
    conv_id = await new_conversation()

    # Ingest enough messages to trigger compaction (well beyond fresh tail of 10)
    for i in range(30):
        await ingest(conv_id, "user", f"Message number {i} about various topics")

    mock_client = _make_mock_router_client("Summary of old messages.")

    await compact(conv_id, mock_client, "test-model")

    # Verify leaf summary nodes were created
    async with conn.execute(
        "SELECT * FROM summary_nodes WHERE conversation_id = ? AND depth = 0",
        (conv_id,),
    ) as cur:
        leaves = await cur.fetchall()

    assert len(leaves) > 0
    assert leaves[0]["content"] == "Summary of old messages."
    # The LLM should have been called at least once
    assert mock_client.chat.call_count >= 1

    # Clean up override
    config_mod.set_override("context.fresh_tail_count", 32)


@pytest.mark.asyncio
async def test_compact_does_nothing_below_threshold(_init_db):
    """When message count is below threshold, compact() does nothing."""
    from spare_paw.context import compact

    conn = _init_db
    conv_id = await new_conversation()

    # Only a few messages — not enough to trigger compaction
    for i in range(3):
        await ingest(conv_id, "user", f"Short message {i}")

    mock_client = _make_mock_router_client()

    await compact(conv_id, mock_client, "test-model")

    # No summary nodes should be created
    async with conn.execute(
        "SELECT COUNT(*) FROM summary_nodes WHERE conversation_id = ?",
        (conv_id,),
    ) as cur:
        count = (await cur.fetchone())[0]

    assert count == 0
    # LLM should not have been called
    assert mock_client.chat.call_count == 0


@pytest.mark.asyncio
async def test_condense_creates_higher_level_node(_init_db):
    """When 4+ leaf nodes exist, they get condensed into a depth=1 node."""
    from spare_paw.config import config as config_mod
    from spare_paw.context import compact

    config_mod.set_override("context.fresh_tail_count", 5)

    conn = _init_db
    conv_id = await new_conversation()

    # Pre-create 4 leaf summary nodes to trigger condensation
    leaf_ids = []
    for i in range(4):
        lid = await _insert_summary_node(
            conn, conv_id, f"Leaf summary {i} about topic {i}",
            depth=0, source_msg_ids=[f"msg_{i}"],
        )
        leaf_ids.append(lid)

    # Need enough messages so compact runs past the fresh tail check
    for i in range(20):
        await ingest(conv_id, "user", f"Message {i}")

    mock_client = _make_mock_router_client("Condensed summary of all leaves.")

    await compact(conv_id, mock_client, "test-model")

    # Check for higher-level node (depth >= 1)
    async with conn.execute(
        "SELECT * FROM summary_nodes WHERE conversation_id = ? AND depth >= 1",
        (conv_id,),
    ) as cur:
        condensed = await cur.fetchall()

    assert len(condensed) >= 1

    # Clean up override
    config_mod.set_override("context.fresh_tail_count", 32)


@pytest.mark.asyncio
async def test_compact_preserves_fresh_tail(_init_db):
    """The last N messages (fresh tail) should never be compacted."""
    from spare_paw.config import config as config_mod
    from spare_paw.context import compact

    config_mod.set_override("context.fresh_tail_count", 10)

    conn = _init_db
    conv_id = await new_conversation()

    # Ingest messages — the most recent ones should remain untouched
    msg_ids = []
    for i in range(30):
        mid = await ingest(conv_id, "user", f"Message {i} with content")
        msg_ids.append(mid)

    mock_client = _make_mock_router_client("Summary of older messages.")

    await compact(conv_id, mock_client, "test-model")

    # Fetch all source_msg_ids from summary nodes
    async with conn.execute(
        "SELECT source_msg_ids FROM summary_nodes WHERE conversation_id = ?",
        (conv_id,),
    ) as cur:
        rows = await cur.fetchall()

    compacted_ids: set[str] = set()
    for row in rows:
        compacted_ids.update(json.loads(row["source_msg_ids"]))

    # The newest messages should NOT appear in any summary source_msg_ids
    # (they are the "fresh tail" that stays unconsolidated)
    newest_ids = set(msg_ids[-10:])  # last 10 = fresh tail, should not be compacted
    assert compacted_ids.isdisjoint(newest_ids), (
        "Fresh tail messages should not be compacted"
    )

    # Clean up override
    config_mod.set_override("context.fresh_tail_count", 32)


# ---------------------------------------------------------------------------
# Assembly tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_assemble_includes_summaries(_init_db):
    """When summary nodes exist, assemble() includes them between system prompt and fresh messages."""
    conn = _init_db
    conv_id = await new_conversation()

    # Insert summary nodes
    await _insert_summary_node(
        conn, conv_id, "Previously, the user discussed Python and databases.",
        depth=1, token_count=10,
    )

    # Add some fresh messages
    await ingest(conv_id, "user", "What were we talking about?")
    await ingest(conv_id, "assistant", "Let me check our history.")

    messages = await assemble(conv_id, "You are a helpful assistant.")

    # System prompt first
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == "You are a helpful assistant."

    # Should contain a compressed history entry somewhere before fresh messages
    contents = [m["content"] for m in messages]
    has_summary = any("[Compressed History]" in c for c in contents)
    assert has_summary, "assemble() should include summary nodes as compressed history"

    # Fresh messages should still be present
    assert any("What were we talking about?" in c for c in contents)
    assert any("Let me check our history." in c for c in contents)


@pytest.mark.asyncio
async def test_assemble_with_no_summaries_unchanged(_init_db):
    """When no summary nodes exist, assemble() works exactly as before."""
    conv_id = await new_conversation()
    await ingest(conv_id, "user", "Hello")
    await ingest(conv_id, "assistant", "Hi there!")

    messages = await assemble(conv_id, "You are a helpful assistant.")

    assert messages[0] == {"role": "system", "content": "You are a helpful assistant."}
    assert messages[1] == {"role": "user", "content": "Hello"}
    assert messages[2] == {"role": "assistant", "content": "Hi there!"}
    assert len(messages) == 3

    # No compressed history markers
    assert not any("[Compressed History]" in m["content"] for m in messages)


# ---------------------------------------------------------------------------
# LCM tool tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_lcm_grep_searches_summaries(_init_db):
    """lcm_grep should find content in summary nodes via FTS5."""
    from spare_paw.tools.lcm_tools import _handle_lcm_grep

    conn = _init_db
    conv_id = await new_conversation()

    # Insert a summary node with specific content
    await _insert_summary_node(
        conn, conv_id, "Discussed machine learning and neural networks extensively"
    )

    # Also insert a regular message with different content
    await ingest(conv_id, "user", "Tell me about neural networks")

    result_json = await _handle_lcm_grep(query="neural networks")
    result = json.loads(result_json)

    # Should find results from both messages and summary nodes
    assert len(result.get("results", [])) >= 1


@pytest.mark.asyncio
async def test_lcm_expand_returns_source_messages(_init_db):
    """lcm_expand should return the original messages that a summary was built from."""
    from spare_paw.tools.lcm_tools import _handle_lcm_expand

    conn = _init_db
    conv_id = await new_conversation()

    # Create source messages
    msg_id1 = await ingest(conv_id, "user", "First message about Python")
    msg_id2 = await ingest(conv_id, "assistant", "Python is a great language")
    msg_id3 = await ingest(conv_id, "user", "Tell me about decorators")

    # Create a summary referencing those messages
    summary_id = await _insert_summary_node(
        conn, conv_id, "Discussion about Python and decorators",
        source_msg_ids=[msg_id1, msg_id2, msg_id3],
    )

    result_json = await _handle_lcm_expand(summary_id=summary_id)
    result = json.loads(result_json)

    # Should return the source messages
    messages = result.get("messages", [])
    assert len(messages) == 3
    contents = [m["content"] for m in messages]
    assert "First message about Python" in contents
    assert "Python is a great language" in contents
    assert "Tell me about decorators" in contents


@pytest.mark.asyncio
async def test_lcm_expand_respects_token_limit(_init_db):
    """lcm_expand should cap output at max_tokens."""
    from spare_paw.tools.lcm_tools import _handle_lcm_expand

    conn = _init_db
    conv_id = await new_conversation()

    # Create several long messages
    msg_ids = []
    for i in range(10):
        mid = await ingest(
            conv_id, "user",
            f"This is a fairly long message number {i} with lots of words to consume tokens. " * 20,
        )
        msg_ids.append(mid)

    summary_id = await _insert_summary_node(
        conn, conv_id, "Summary of many long messages",
        source_msg_ids=msg_ids,
    )

    # Request with a very small token limit
    result_json = await _handle_lcm_expand(summary_id=summary_id, max_tokens=100)
    result = json.loads(result_json)

    # Should return fewer messages than the total (capped by tokens)
    messages = result.get("messages", [])
    assert len(messages) < 10, "lcm_expand should cap output when max_tokens is small"
    assert result.get("truncated", False) is True
