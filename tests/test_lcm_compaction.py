"""Tests for LCM compaction error handling and retry logic (Issue #3)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import aiosqlite
import pytest
import pytest_asyncio

import spare_paw.db as db_mod
import spare_paw.context as ctx_mod
from spare_paw.context import compact_with_retry, ingest, new_conversation


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


@pytest.fixture(autouse=True)
def _reset_failure_counter():
    """Reset the module-level failure counter before each test."""
    ctx_mod._compact_consecutive_failures = 0
    yield
    ctx_mod._compact_consecutive_failures = 0


def _make_mock_router_client(summary_text: str = "Summary.") -> MagicMock:
    client = MagicMock()
    mock_response = {
        "choices": [{"message": {"content": summary_text}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }
    client.chat = AsyncMock(return_value=mock_response)
    return client


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_compact_retry_on_single_failure(_init_db):
    """A single compaction failure should trigger one retry and succeed."""
    mock_client = _make_mock_router_client()

    # Patch compact() itself: fail on first call, succeed on second
    call_count = 0

    async def _fake_compact(conv_id, client, model):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise Exception("transient API error")

    conv_id = await new_conversation()

    with patch("spare_paw.context.compact", side_effect=_fake_compact), \
         patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
        await compact_with_retry(conv_id, mock_client, "test-model")
        mock_sleep.assert_awaited_once_with(2)

    assert call_count == 2, "compact() should be called exactly twice (attempt + retry)"
    assert ctx_mod._compact_consecutive_failures == 0


@pytest.mark.asyncio
async def test_source_messages_preserved_on_failure(_init_db):
    """Source messages must not be deleted or modified when compaction fails."""
    from spare_paw.config import config as config_mod

    config_mod.set_override("context.fresh_tail_count", 5)

    conn = _init_db
    conv_id = await new_conversation()
    msg_ids = []
    for i in range(15):
        mid = await ingest(conv_id, "user", f"Important message {i}")
        msg_ids.append(mid)

    mock_client = _make_mock_router_client()
    mock_client.chat.side_effect = Exception("permanent API failure")

    with patch("asyncio.sleep", new_callable=AsyncMock):
        await compact_with_retry(conv_id, mock_client, "test-model")

    # All original messages must still be present
    async with conn.execute(
        "SELECT COUNT(*) FROM messages WHERE conversation_id = ?",
        (conv_id,),
    ) as cur:
        count = (await cur.fetchone())[0]

    assert count == 15, "All source messages must be preserved after compaction failure"

    config_mod.set_override("context.fresh_tail_count", 32)


@pytest.mark.asyncio
async def test_consecutive_failure_counter_increments(_init_db):
    """Each double-failure cycle should increment the consecutive failure counter."""
    from spare_paw.config import config as config_mod

    config_mod.set_override("context.fresh_tail_count", 5)

    conv_id = await new_conversation()
    for i in range(15):
        await ingest(conv_id, "user", f"Message {i}")

    mock_client = _make_mock_router_client()
    mock_client.chat.side_effect = Exception("always fails")

    with patch("asyncio.sleep", new_callable=AsyncMock):
        await compact_with_retry(conv_id, mock_client, "test-model")

    assert ctx_mod._compact_consecutive_failures == 1

    with patch("asyncio.sleep", new_callable=AsyncMock):
        await compact_with_retry(conv_id, mock_client, "test-model")

    assert ctx_mod._compact_consecutive_failures == 2

    config_mod.set_override("context.fresh_tail_count", 32)


@pytest.mark.asyncio
async def test_consecutive_failure_counter_resets_on_success(_init_db):
    """Successful compaction must reset the consecutive failure counter to 0."""
    from spare_paw.config import config as config_mod

    config_mod.set_override("context.fresh_tail_count", 5)

    ctx_mod._compact_consecutive_failures = 2

    conv_id = await new_conversation()
    for i in range(15):
        await ingest(conv_id, "user", f"Message {i}")

    mock_client = _make_mock_router_client("Good summary.")

    await compact_with_retry(conv_id, mock_client, "test-model")

    assert ctx_mod._compact_consecutive_failures == 0

    config_mod.set_override("context.fresh_tail_count", 32)


@pytest.mark.asyncio
async def test_warning_fires_after_three_consecutive_failures(_init_db):
    """A distinct WARNING should be logged after 3+ consecutive failures."""
    import logging
    from spare_paw.config import config as config_mod

    config_mod.set_override("context.fresh_tail_count", 5)

    conv_id = await new_conversation()
    for i in range(15):
        await ingest(conv_id, "user", f"Message {i}")

    mock_client = _make_mock_router_client()
    mock_client.chat.side_effect = Exception("always fails")

    warning_messages: list[str] = []

    class CapturingHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            if record.levelno == logging.WARNING:
                warning_messages.append(record.getMessage())

    handler = CapturingHandler()
    ctx_logger = logging.getLogger("spare_paw.context")
    ctx_logger.addHandler(handler)

    try:
        for _ in range(3):
            with patch("asyncio.sleep", new_callable=AsyncMock):
                await compact_with_retry(conv_id, mock_client, "test-model")
    finally:
        ctx_logger.removeHandler(handler)

    threshold_warnings = [
        m for m in warning_messages if "3+ times consecutively" in m
    ]
    assert len(threshold_warnings) >= 1, (
        "Expected a WARNING about 3+ consecutive failures, got: " + str(warning_messages)
    )

    assert ctx_mod._compact_consecutive_failures == 3

    config_mod.set_override("context.fresh_tail_count", 32)
