"""Tests for dream consolidation engine.

Covers knowledge directory creation, dream execution with mocked LLM,
knowledge retrieval, selective knowledge loading, and token limits.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
import aiosqlite

import spare_paw.db as db_mod
from spare_paw.tools.dream import (
    ensure_knowledge_dir,
    get_knowledge_for_context,
    get_selective_knowledge,
    run_dream,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture()
async def _init_db(tmp_path):
    """Create a temp DB with all schemas applied."""
    db_path = tmp_path / "test.db"

    conn = await aiosqlite.connect(str(db_path))
    conn.row_factory = aiosqlite.Row
    await conn.execute("PRAGMA journal_mode = WAL")
    await conn.execute("PRAGMA foreign_keys = ON")

    db_mod._connection = conn

    await conn.executescript(db_mod.SCHEMA_V1)
    await conn.executescript(db_mod.SCHEMA_V2)
    await conn.executescript(db_mod.SCHEMA_V3)
    await conn.execute(f"PRAGMA user_version = {db_mod.CURRENT_SCHEMA_VERSION}")
    await conn.commit()

    yield conn

    await conn.close()
    db_mod._connection = None


@pytest.fixture()
def knowledge_dir(tmp_path):
    """Provide a temporary knowledge directory and patch KNOWLEDGE_DIR."""
    kdir = tmp_path / "knowledge"
    with patch("spare_paw.tools.dream.KNOWLEDGE_DIR", kdir):
        yield kdir


def _make_mock_app_state(llm_response: str) -> MagicMock:
    """Create a mock app_state with a router_client that returns the given text."""
    app_state = MagicMock()
    mock_response = {
        "choices": [{"message": {"content": llm_response}}],
    }
    app_state.router_client = MagicMock()
    app_state.router_client.chat = AsyncMock(return_value=mock_response)
    app_state.config = MagicMock()
    app_state.config.get = MagicMock(return_value="test-model")
    return app_state


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ensure_knowledge_dir_creates_structure(knowledge_dir):
    """ensure_knowledge_dir creates the directory and INDEX.md if missing."""
    assert not knowledge_dir.exists()

    result = ensure_knowledge_dir()

    assert result == knowledge_dir
    assert knowledge_dir.is_dir()
    index = knowledge_dir / "INDEX.md"
    assert index.is_file()
    assert "Knowledge Index" in index.read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_ensure_knowledge_dir_idempotent(knowledge_dir):
    """Calling ensure_knowledge_dir twice doesn't overwrite existing files."""
    ensure_knowledge_dir()

    # Write custom content to INDEX.md
    index = knowledge_dir / "INDEX.md"
    index.write_text("# Custom Index\n- my-file.md — custom stuff\n", encoding="utf-8")

    # Create a knowledge file
    (knowledge_dir / "my-file.md").write_text("Custom content\n", encoding="utf-8")

    # Call again — should NOT overwrite
    ensure_knowledge_dir()

    assert "Custom Index" in index.read_text(encoding="utf-8")
    assert (knowledge_dir / "my-file.md").read_text(encoding="utf-8") == "Custom content\n"


@pytest.mark.asyncio
async def test_run_dream_creates_knowledge_files(_init_db, knowledge_dir):
    """run_dream parses LLM output and writes knowledge files."""
    conn = _init_db

    # Insert a conversation and recent messages
    now = datetime.now(timezone.utc).isoformat()
    await conn.execute(
        "INSERT INTO conversations (id, created_at) VALUES (?, ?)",
        ("conv1", now),
    )
    await conn.execute(
        "INSERT INTO messages (id, conversation_id, role, content, token_count, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        ("msg1", "conv1", "user", "I prefer terse responses", 5, now),
    )
    await conn.execute(
        "INSERT INTO messages (id, conversation_id, role, content, token_count, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        ("msg2", "conv1", "assistant", "Noted, I will keep it brief", 6, now),
    )
    await conn.commit()

    llm_response = (
        "### FILE: user-preferences.md\n"
        "- Prefers terse responses\n"
        "- Uses Python and TypeScript\n"
        "\n"
        "### FILE: INDEX.md\n"
        "- user-preferences.md — Communication style and language preferences\n"
    )

    app_state = _make_mock_app_state(llm_response)

    result = await run_dream(app_state)

    assert "Updated 2 file(s)" in result
    assert (knowledge_dir / "user-preferences.md").is_file()
    content = (knowledge_dir / "user-preferences.md").read_text(encoding="utf-8")
    assert "Prefers terse responses" in content
    assert "Python and TypeScript" in content

    index_content = (knowledge_dir / "INDEX.md").read_text(encoding="utf-8")
    assert "user-preferences.md" in index_content


@pytest.mark.asyncio
async def test_run_dream_with_no_recent_messages(_init_db, knowledge_dir):
    """If no messages in last 24h, dream returns early."""
    # Insert a message from 48 hours ago
    conn = _init_db
    old_time = (datetime.now(timezone.utc) - timedelta(hours=48)).isoformat()
    await conn.execute(
        "INSERT INTO conversations (id, created_at) VALUES (?, ?)",
        ("conv1", old_time),
    )
    await conn.execute(
        "INSERT INTO messages (id, conversation_id, role, content, token_count, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        ("msg1", "conv1", "user", "Old message", 3, old_time),
    )
    await conn.commit()

    app_state = _make_mock_app_state("should not be called")

    result = await run_dream(app_state)

    assert "No recent messages" in result
    # LLM should not have been called
    app_state.router_client.chat.assert_not_called()


@pytest.mark.asyncio
async def test_get_knowledge_for_context(knowledge_dir):
    """Writes some files, verifies they're read back and formatted."""
    ensure_knowledge_dir()

    (knowledge_dir / "user-preferences.md").write_text(
        "- Likes Python\n- Prefers dark mode\n", encoding="utf-8"
    )
    (knowledge_dir / "tech-stack.md").write_text(
        "- Python 3.12\n- PostgreSQL\n", encoding="utf-8"
    )

    result = get_knowledge_for_context(max_tokens=5000)

    assert "# Knowledge" in result
    assert "Likes Python" in result
    assert "Python 3.12" in result
    assert "tech-stack" in result
    assert "user-preferences" in result


@pytest.mark.asyncio
async def test_get_selective_knowledge(knowledge_dir):
    """Writes files with different topics, verifies keyword matching picks the right ones."""
    ensure_knowledge_dir()

    (knowledge_dir / "INDEX.md").write_text(
        "# Knowledge Index\n"
        "- user-preferences.md — Communication style and language preferences\n"
        "- tech-stack.md — Programming tools and frameworks\n"
        "- people.md — People mentioned in conversations\n",
        encoding="utf-8",
    )
    (knowledge_dir / "user-preferences.md").write_text(
        "- Prefers terse responses\n", encoding="utf-8"
    )
    (knowledge_dir / "tech-stack.md").write_text(
        "- Python 3.12\n- FastAPI\n", encoding="utf-8"
    )
    (knowledge_dir / "people.md").write_text(
        "- Alice: coworker\n", encoding="utf-8"
    )

    # Query about programming should match tech-stack
    result = get_selective_knowledge("programming tools Python")

    assert "Python 3.12" in result
    assert "FastAPI" in result


@pytest.mark.asyncio
async def test_get_knowledge_respects_token_limit(knowledge_dir):
    """With a small max_tokens, output is truncated."""
    ensure_knowledge_dir()

    # Write a large file
    large_content = "Important fact number X. " * 500
    (knowledge_dir / "big-file.md").write_text(large_content, encoding="utf-8")

    # Very small budget — only ~40 chars
    result = get_knowledge_for_context(max_tokens=10)

    # Should be significantly shorter than the full content
    assert len(result) < len(large_content)
    # Should contain truncation marker or be cut short
    assert "[truncated]" in result or len(result) < 200
