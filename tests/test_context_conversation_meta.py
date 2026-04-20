"""Tests for per-conversation metadata helpers."""
from __future__ import annotations

import pytest
import pytest_asyncio

from spare_paw import context as ctx
from spare_paw.db import close_db, init_db


@pytest_asyncio.fixture(autouse=True)
async def _fresh_db(tmp_path, monkeypatch):
    """Redirect the DB to a per-test temp file and initialize schema."""
    db_file = tmp_path / "spare-paw.db"
    monkeypatch.setattr("spare_paw.db.DB_PATH", db_file)
    monkeypatch.setattr("spare_paw.db.DB_DIR", tmp_path)
    monkeypatch.setattr("spare_paw.db._connection", None)
    await init_db()
    yield
    await close_db()


@pytest.mark.asyncio
async def test_get_conversation_meta_empty_when_unset():
    convo_id = await ctx.new_conversation()
    meta = await ctx.get_conversation_meta(convo_id)
    assert meta == {}


@pytest.mark.asyncio
async def test_set_conversation_meta_single_key():
    convo_id = await ctx.new_conversation()
    await ctx.set_conversation_meta(convo_id, "talk_mode", True)
    meta = await ctx.get_conversation_meta(convo_id)
    assert meta == {"talk_mode": True}


@pytest.mark.asyncio
async def test_set_conversation_meta_merges_not_overwrites():
    convo_id = await ctx.new_conversation()
    await ctx.set_conversation_meta(convo_id, "talk_mode", True)
    await ctx.set_conversation_meta(convo_id, "voice", "shimmer")
    meta = await ctx.get_conversation_meta(convo_id)
    assert meta == {"talk_mode": True, "voice": "shimmer"}


@pytest.mark.asyncio
async def test_set_conversation_meta_overwrite_same_key():
    convo_id = await ctx.new_conversation()
    await ctx.set_conversation_meta(convo_id, "talk_mode", True)
    await ctx.set_conversation_meta(convo_id, "talk_mode", False)
    meta = await ctx.get_conversation_meta(convo_id)
    assert meta == {"talk_mode": False}


@pytest.mark.asyncio
async def test_get_conversation_meta_unknown_convo_returns_empty():
    meta = await ctx.get_conversation_meta("does-not-exist")
    assert meta == {}


@pytest.mark.asyncio
async def test_get_conversation_meta_tolerates_invalid_json():
    from spare_paw.db import get_db
    convo_id = await ctx.new_conversation()
    db = await get_db()
    await db.execute(
        "UPDATE conversations SET metadata = ? WHERE id = ?",
        ("not valid json{", convo_id),
    )
    await db.commit()
    meta = await ctx.get_conversation_meta(convo_id)
    assert meta == {}
