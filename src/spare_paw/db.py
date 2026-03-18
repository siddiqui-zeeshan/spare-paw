"""SQLite database management with aiosqlite.

Provides async access to ~/.spare-paw/spare-paw.db with WAL mode, FTS5, and schema
versioning via PRAGMA user_version.
"""

from __future__ import annotations

import logging
from pathlib import Path

import aiosqlite

logger = logging.getLogger(__name__)

DB_DIR = Path.home() / ".spare-paw"
DB_PATH = DB_DIR / "spare-paw.db"

CURRENT_SCHEMA_VERSION = 2

SCHEMA_V1 = """\
CREATE TABLE IF NOT EXISTS messages (
    id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    token_count INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    metadata TEXT
);

CREATE INDEX IF NOT EXISTS idx_messages_conversation
    ON messages(conversation_id, created_at);

CREATE TABLE IF NOT EXISTS conversations (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    last_message_at TEXT,
    metadata TEXT
);

CREATE TABLE IF NOT EXISTS cron_jobs (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    schedule TEXT NOT NULL,
    prompt TEXT NOT NULL,
    model TEXT,
    tools_allowed TEXT,
    enabled INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL,
    last_run_at TEXT,
    last_result TEXT,
    last_error TEXT,
    metadata TEXT
);

CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts
    USING fts5(content, content=messages, content_rowid=rowid);

CREATE TRIGGER IF NOT EXISTS messages_ai AFTER INSERT ON messages BEGIN
    INSERT INTO messages_fts(rowid, content) VALUES (new.rowid, new.content);
END;

CREATE TRIGGER IF NOT EXISTS messages_ad AFTER DELETE ON messages BEGIN
    INSERT INTO messages_fts(messages_fts, rowid, content)
        VALUES('delete', old.rowid, old.content);
END;

CREATE TRIGGER IF NOT EXISTS messages_au AFTER UPDATE ON messages BEGIN
    INSERT INTO messages_fts(messages_fts, rowid, content)
        VALUES('delete', old.rowid, old.content);
    INSERT INTO messages_fts(rowid, content) VALUES (new.rowid, new.content);
END;
"""

SCHEMA_V2 = """\
CREATE TABLE IF NOT EXISTS memories (
    id TEXT PRIMARY KEY,
    key TEXT NOT NULL,
    value TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_memories_key ON memories(key);

CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts
    USING fts5(key, value, content=memories, content_rowid=rowid);

CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
    INSERT INTO memories_fts(rowid, key, value) VALUES (new.rowid, new.key, new.value);
END;

CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, key, value)
        VALUES('delete', old.rowid, old.key, old.value);
END;

CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, key, value)
        VALUES('delete', old.rowid, old.key, old.value);
    INSERT INTO memories_fts(rowid, key, value) VALUES (new.rowid, new.key, new.value);
END;
"""

# ---- Singleton connection ----

_connection: aiosqlite.Connection | None = None


async def get_db() -> aiosqlite.Connection:
    """Return the shared aiosqlite connection, creating it if needed."""
    global _connection
    if _connection is None:
        _connection = await _open_connection()
    return _connection


async def _open_connection() -> aiosqlite.Connection:
    """Open a new connection with pragmas set."""
    DB_DIR.mkdir(parents=True, exist_ok=True)
    conn = await aiosqlite.connect(str(DB_PATH))
    conn.row_factory = aiosqlite.Row
    await conn.execute("PRAGMA journal_mode = WAL")
    await conn.execute("PRAGMA foreign_keys = ON")
    await conn.execute("PRAGMA busy_timeout = 5000")
    return conn


async def init_db() -> None:
    """Create tables if they don't exist and apply any pending migrations."""
    conn = await get_db()
    version = await _get_user_version(conn)

    if version < 1:
        logger.info("Initializing database schema v1")
        await conn.executescript(SCHEMA_V1)
        await _set_user_version(conn, 1)
        await conn.commit()
        logger.info("Database schema v1 applied")
    else:
        logger.debug("Database schema is up to date (v%d)", version)

    if version < 2:
        logger.info("Migrating database to schema v2 (memories)")
        await conn.executescript(SCHEMA_V2)
        await _set_user_version(conn, 2)
        await conn.commit()
        logger.info("Database schema v2 applied")


async def close_db() -> None:
    """Close the shared database connection."""
    global _connection
    if _connection is not None:
        await _connection.close()
        _connection = None
        logger.info("Database connection closed")


async def _get_user_version(conn: aiosqlite.Connection) -> int:
    """Read PRAGMA user_version."""
    async with conn.execute("PRAGMA user_version") as cursor:
        row = await cursor.fetchone()
        return row[0] if row else 0


async def _set_user_version(conn: aiosqlite.Connection, version: int) -> None:
    """Set PRAGMA user_version (cannot use parameter binding for PRAGMAs)."""
    await conn.execute(f"PRAGMA user_version = {version}")
