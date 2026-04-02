"""Async SQLite database helper — connection management and table auto-creation.

This module owns the database schema definition and provides two public
async helpers:

* ``get_db()``  — opens a new aiosqlite connection with WAL mode and
                  foreign-key enforcement enabled.
* ``init_db()`` — creates every table if it does not already exist.

Every caller that obtains a connection via ``get_db()`` is responsible for
closing it (typically inside a ``try / finally`` block).
"""

import os

import aiosqlite

from repo_insight.config import settings

# ---------------------------------------------------------------------------
# Schema DDL
# ---------------------------------------------------------------------------
# SQL statements used to bootstrap the schema on first run.
#
# Four tables are created:
#
# 1. ``sessions``        — top-level chat sessions (one per conversation).
# 2. ``messages``        — individual chat messages belonging to a session.
# 3. ``reports``         — generated HTML / Markdown reports tied to a session.
# 4. ``session_context`` — persisted LangGraph state (search results,
#                          analysis results, report data) so that follow-up
#                          user requests inside the same session can reuse
#                          earlier tool outputs without re-fetching them.
#
# Foreign-key constraints with ``ON DELETE CASCADE`` ensure that deleting a
# session automatically removes all its child messages, reports, and context.
# ---------------------------------------------------------------------------
_CREATE_TABLES = """
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL DEFAULT 'New Chat',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    metadata TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS reports (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    title TEXT NOT NULL DEFAULT '',
    format TEXT NOT NULL DEFAULT 'markdown',
    content TEXT NOT NULL DEFAULT '',
    file_path TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS session_context (
    session_id TEXT PRIMARY KEY,
    search_results TEXT NOT NULL DEFAULT '',
    analysis_results TEXT NOT NULL DEFAULT '',
    report_data TEXT NOT NULL DEFAULT '',
    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
);
"""


async def get_db() -> aiosqlite.Connection:
    """Open a connection to the SQLite database, creating parent dirs if needed.

    Configuration details:
    * The database file path comes from ``settings.db_path``.
    * ``row_factory = aiosqlite.Row`` so that query results behave like
      dictionaries (column-name access) rather than plain tuples.
    * **WAL mode** (Write-Ahead Logging) is enabled via PRAGMA so that
      readers do not block writers — this is essential for an async web
      server where multiple requests may hit the database concurrently.
    * **Foreign keys** are enabled explicitly because SQLite turns them
      off by default for backwards-compatibility reasons.  Without this
      PRAGMA the ON DELETE CASCADE clauses in the schema would be ignored.
    """
    db_path = settings.db_path
    # Ensure the directory that will hold the .db file exists.
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
    db = await aiosqlite.connect(db_path)
    # Use Row factory so rows can be accessed by column name (e.g. row["id"]).
    db.row_factory = aiosqlite.Row
    # Enable WAL journal mode for better concurrent read/write performance.
    await db.execute("PRAGMA journal_mode=WAL")
    # Turn on foreign-key constraint enforcement (off by default in SQLite).
    await db.execute("PRAGMA foreign_keys=ON")
    return db


async def init_db() -> None:
    """Create all tables if they do not exist yet.

    This is called once at application startup (typically in an ``on_startup``
    FastAPI lifecycle hook) to guarantee the schema is present before any
    request is served.
    """
    db = await get_db()
    try:
        await db.executescript(_CREATE_TABLES)
        await db.commit()
    finally:
        await db.close()
