"""Tests for spare_paw.context — sliding window context manager."""

from __future__ import annotations

import pytest
import pytest_asyncio
import aiosqlite

import spare_paw.db as db_mod
from spare_paw.context import ingest, assemble, search, get_or_create_conversation, new_conversation


@pytest_asyncio.fixture()
async def _init_db(tmp_path):
    """Create an in-memory-like temp DB and wire it into the db module singleton."""
    db_path = tmp_path / "test.db"

    # Replace the module-level singleton so get_db() returns our temp connection
    conn = await aiosqlite.connect(str(db_path))
    conn.row_factory = aiosqlite.Row
    await conn.execute("PRAGMA journal_mode = WAL")
    await conn.execute("PRAGMA foreign_keys = ON")

    # Patch the singleton
    db_mod._connection = conn

    # Apply schema
    await conn.executescript(db_mod.SCHEMA_V1)
    await conn.execute(f"PRAGMA user_version = {db_mod.CURRENT_SCHEMA_VERSION}")
    await conn.commit()

    yield conn

    await conn.close()
    db_mod._connection = None


# ---------------------------------------------------------------------------
# ingest
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ingest_stores_message_and_returns_id(_init_db):
    conv_id = await new_conversation()
    msg_id = await ingest(conv_id, "user", "Hello, world!")

    assert isinstance(msg_id, str)
    assert len(msg_id) == 32  # uuid4 hex

    # Verify row exists
    db = _init_db
    async with db.execute("SELECT * FROM messages WHERE id = ?", (msg_id,)) as cur:
        row = await cur.fetchone()

    assert row is not None
    assert row["role"] == "user"
    assert row["content"] == "Hello, world!"
    assert row["conversation_id"] == conv_id
    assert row["token_count"] > 0


# ---------------------------------------------------------------------------
# assemble
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_assemble_returns_openai_format_with_system_prompt(_init_db):
    conv_id = await new_conversation()
    await ingest(conv_id, "user", "Hi")
    await ingest(conv_id, "assistant", "Hello!")

    messages = await assemble(conv_id, "You are a helpful assistant.")

    assert messages[0] == {"role": "system", "content": "You are a helpful assistant."}
    assert messages[1] == {"role": "user", "content": "Hi"}
    assert messages[2] == {"role": "assistant", "content": "Hello!"}
    assert len(messages) == 3


@pytest.mark.asyncio
async def test_assemble_respects_token_budget(_init_db, monkeypatch):
    """When the token budget is tiny, the oldest messages should be dropped."""
    from spare_paw import config as config_mod

    # Set a very small budget so only 1-2 messages fit
    config_mod.config.set_override("context.token_budget", 50)
    config_mod.config.set_override("context.safety_margin", 1.0)

    conv_id = await new_conversation()

    # Ingest several messages — each has some token cost
    for i in range(10):
        await ingest(conv_id, "user", f"Message number {i} with some extra words to consume tokens")

    messages = await assemble(conv_id, "sys")

    # System prompt is always first
    assert messages[0]["role"] == "system"
    # Fewer than 10 user messages should survive (budget is only 50 tokens)
    user_messages = [m for m in messages if m["role"] == "user"]
    assert len(user_messages) < 10

    # The surviving messages should be the *newest* ones (sliding window)
    # because assemble walks newest-first and stops when budget exceeded
    if user_messages:
        assert "Message number 9" in user_messages[-1]["content"]

    # Clean up override
    config_mod.config.set_override("context.token_budget", 120000)
    config_mod.config.set_override("context.safety_margin", 0.85)


# ---------------------------------------------------------------------------
# search (FTS5)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_search_finds_messages_via_fts(_init_db):
    conv_id = await new_conversation()
    await ingest(conv_id, "user", "The quick brown fox jumps over the lazy dog")
    await ingest(conv_id, "user", "Python is a great programming language")
    await ingest(conv_id, "assistant", "I agree, Python is wonderful")

    results = await search("Python")
    assert len(results) >= 2
    contents = [r["content"] for r in results]
    assert any("Python" in c for c in contents)


@pytest.mark.asyncio
async def test_search_returns_empty_for_no_match(_init_db):
    conv_id = await new_conversation()
    await ingest(conv_id, "user", "Hello there")

    results = await search("xyznonexistent")
    assert results == []


# ---------------------------------------------------------------------------
# get_or_create_conversation / new_conversation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_or_create_conversation_creates_new_if_none(_init_db):
    conv_id = await get_or_create_conversation()
    assert isinstance(conv_id, str)
    assert len(conv_id) == 32

    # Calling again should return the same conversation (it's the most recent)
    conv_id2 = await get_or_create_conversation()
    assert conv_id2 == conv_id


@pytest.mark.asyncio
async def test_new_conversation_creates_fresh(_init_db):
    conv_id1 = await new_conversation()
    conv_id2 = await new_conversation()

    assert conv_id1 != conv_id2

    # get_or_create should return the newest one
    latest = await get_or_create_conversation()
    assert latest == conv_id2
