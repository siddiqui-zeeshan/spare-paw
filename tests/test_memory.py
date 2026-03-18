"""Tests for spare_paw.tools.memory — remember, recall, forget, list."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest
import pytest_asyncio

import spare_paw.db as db_mod


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest_asyncio.fixture(autouse=True)
async def _isolate_db(tmp_path):
    """Redirect DB to a temp directory and reset the singleton."""
    tmp_db_dir = tmp_path / ".spare-paw"
    tmp_db_path = tmp_db_dir / "spare-paw.db"
    tmp_db_dir.mkdir()

    with (
        patch.object(db_mod, "DB_DIR", tmp_db_dir),
        patch.object(db_mod, "DB_PATH", tmp_db_path),
    ):
        db_mod._connection = None
        await db_mod.init_db()
        yield
        await db_mod.close_db()


# ---------------------------------------------------------------------------
# remember
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_remember_creates_new_memory():
    from spare_paw.tools.memory import _handle_remember

    result = json.loads(await _handle_remember("wifi_password", "hunter2"))
    assert result["status"] == "created"
    assert result["key"] == "wifi_password"
    assert "id" in result


@pytest.mark.asyncio
async def test_remember_updates_existing_memory():
    from spare_paw.tools.memory import _handle_remember

    await _handle_remember("color", "blue")
    result = json.loads(await _handle_remember("color", "red"))
    assert result["status"] == "updated"
    assert result["key"] == "color"


# ---------------------------------------------------------------------------
# recall (FTS5)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_recall_finds_by_fts():
    from spare_paw.tools.memory import _handle_recall, _handle_remember

    await _handle_remember("birthday", "January 15th")
    result = json.loads(await _handle_recall("birthday"))
    assert len(result["results"]) >= 1
    assert result["results"][0]["key"] == "birthday"
    assert result["results"][0]["value"] == "January 15th"


@pytest.mark.asyncio
async def test_recall_returns_empty_for_no_match():
    from spare_paw.tools.memory import _handle_recall

    result = json.loads(await _handle_recall("xyznonexistent999"))
    assert result["results"] == []
    assert result["query"] == "xyznonexistent999"


# ---------------------------------------------------------------------------
# forget_memory
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_forget_memory_deletes():
    from spare_paw.tools.memory import _handle_forget_memory, _handle_remember

    await _handle_remember("temp_note", "delete me")
    result = json.loads(await _handle_forget_memory("temp_note"))
    assert result["status"] == "deleted"
    assert result["key"] == "temp_note"


@pytest.mark.asyncio
async def test_forget_memory_nonexistent_returns_error():
    from spare_paw.tools.memory import _handle_forget_memory

    result = json.loads(await _handle_forget_memory("no_such_key"))
    assert "error" in result
    assert "no_such_key" in result["error"]


# ---------------------------------------------------------------------------
# list_memories
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_list_memories_returns_all():
    from spare_paw.tools.memory import _handle_list_memories, _handle_remember

    await _handle_remember("k1", "v1")
    await _handle_remember("k2", "v2")
    result = json.loads(await _handle_list_memories())
    assert result["count"] == 2
    keys = {m["key"] for m in result["memories"]}
    assert keys == {"k1", "k2"}


# ---------------------------------------------------------------------------
# get_all_memories (system prompt injection)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_all_memories_for_prompt_injection():
    from spare_paw.tools.memory import _handle_remember, get_all_memories

    await _handle_remember("name", "Zeeshan")
    await _handle_remember("city", "Karachi")
    memories = await get_all_memories()

    assert isinstance(memories, list)
    assert len(memories) == 2
    keys = {m["key"] for m in memories}
    assert keys == {"name", "city"}
    # Should be sorted by key
    assert memories[0]["key"] == "city"
    assert memories[1]["key"] == "name"
