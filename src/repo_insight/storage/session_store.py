"""Async CRUD operations for sessions, messages, and reports.

This module is the single data-access layer between the API / agent code and
the SQLite database.  Every public function opens its own connection, performs
the query, and closes the connection in a ``try / finally`` block to avoid
leaking database handles.

The functions are grouped into four sections:
  1. Sessions        — create, get, list, update, delete.
  2. Messages        — add and list (messages are append-only).
  3. Reports         — add, list, and get-by-id.
  4. Session Context — get and upsert persisted LangGraph state.
"""

from __future__ import annotations

from repo_insight.storage.database import get_db
from repo_insight.storage.models import Message, Report, Session


# --------------- Sessions ---------------

async def create_session(title: str = "New Chat") -> Session:
    """Insert a new session row and return the populated model.

    A fresh ``Session`` object is created first (which auto-generates the
    UUID and timestamps), then its fields are written to the database.
    """
    session = Session(title=title)
    db = await get_db()
    try:
        await db.execute(
            "INSERT INTO sessions (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
            (session.id, session.title, session.created_at, session.updated_at),
        )
        await db.commit()
    finally:
        await db.close()
    return session


async def get_session(session_id: str) -> Session | None:
    """Look up a single session by its primary key or a unique ID prefix.

    If ``session_id`` is shorter than a full UUID (36 chars), the lookup
    uses a LIKE prefix match.  If exactly one session matches, it is
    returned; otherwise ``None`` is returned (ambiguous or not found).
    """
    db = await get_db()
    try:
        if len(session_id) < 36:
            cursor = await db.execute(
                "SELECT * FROM sessions WHERE id LIKE ?",
                (session_id + "%",),
            )
            rows = await cursor.fetchall()
            if len(rows) == 1:
                return Session(**dict(rows[0]))
            return None
        cursor = await db.execute("SELECT * FROM sessions WHERE id = ?", (session_id,))
        row = await cursor.fetchone()
        if not row:
            return None
        return Session(**dict(row))
    finally:
        await db.close()


async def list_sessions() -> list[Session]:
    """Return every session, ordered by most-recently-updated first.

    The descending ``updated_at`` ordering ensures the UI can display
    the conversation list with the most active session at the top.
    """
    db = await get_db()
    try:
        cursor = await db.execute("SELECT * FROM sessions ORDER BY updated_at DESC")
        rows = await cursor.fetchall()
        return [Session(**dict(r)) for r in rows]
    finally:
        await db.close()


async def update_session(session_id: str, title: str) -> Session | None:
    """Change a session's title and refresh its ``updated_at`` timestamp.

    After the UPDATE query, the function re-fetches the session via
    ``get_session`` so the returned model reflects the actual database
    state (including the new ``updated_at`` value).
    """
    from datetime import datetime

    db = await get_db()
    try:
        now = datetime.utcnow().isoformat()
        await db.execute(
            "UPDATE sessions SET title = ?, updated_at = ? WHERE id = ?",
            (title, now, session_id),
        )
        await db.commit()
    finally:
        await db.close()
    return await get_session(session_id)


async def delete_session(session_id: str) -> bool:
    """Delete a session and all of its dependent data.

    Child rows in ``messages``, ``reports``, and ``session_context`` are
    removed explicitly before the session itself is deleted.  Although the
    schema defines ``ON DELETE CASCADE``, the explicit deletes serve as an
    extra safety net in case PRAGMA foreign_keys is not enabled for some
    reason.

    Returns ``True`` if a session was actually deleted, ``False`` otherwise.
    """
    db = await get_db()
    try:
        # Explicitly delete children first as a defensive measure.
        await db.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
        await db.execute("DELETE FROM reports WHERE session_id = ?", (session_id,))
        await db.execute("DELETE FROM session_context WHERE session_id = ?", (session_id,))
        # Delete the session itself and check if a row was affected.
        cursor = await db.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        await db.commit()
        return cursor.rowcount > 0
    finally:
        await db.close()


# --------------- Messages ---------------

async def add_message(session_id: str, role: str, content: str, metadata: str = "{}") -> Message:
    """Append a new message to a session and return the model.

    In addition to inserting the message row, this function also bumps
    the parent session's ``updated_at`` timestamp so that the session
    list stays sorted by recency.
    """
    msg = Message(session_id=session_id, role=role, content=content, metadata=metadata)
    db = await get_db()
    try:
        await db.execute(
            "INSERT INTO messages (id, session_id, role, content, metadata, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (msg.id, msg.session_id, msg.role, msg.content, msg.metadata, msg.created_at),
        )
        # Touch session updated_at
        from datetime import datetime

        await db.execute(
            "UPDATE sessions SET updated_at = ? WHERE id = ?",
            (datetime.utcnow().isoformat(), session_id),
        )
        await db.commit()
    finally:
        await db.close()
    return msg


async def get_messages(session_id: str) -> list[Message]:
    """Return all messages for a session in chronological order.

    Messages are ordered by ``created_at ASC`` so the conversation reads
    top-to-bottom from oldest to newest, which is the order expected by
    both the front-end chat view and the LangChain message history builder.
    """
    db = await get_db()
    try:
        cursor = await db.execute(
            "SELECT * FROM messages WHERE session_id = ? ORDER BY created_at ASC",
            (session_id,),
        )
        rows = await cursor.fetchall()
        return [Message(**dict(r)) for r in rows]
    finally:
        await db.close()


# --------------- Reports ---------------

async def add_report(
    session_id: str,
    title: str,
    fmt: str,
    content: str,
    file_path: str = "",
) -> Report:
    """Create a new report row linked to a session.

    The ``fmt`` parameter maps to the ``format`` column in the database
    (named differently here to avoid shadowing Python's built-in ``format``).
    """
    report = Report(
        session_id=session_id,
        title=title,
        format=fmt,
        content=content,
        file_path=file_path,
    )
    db = await get_db()
    try:
        await db.execute(
            "INSERT INTO reports (id, session_id, title, format, content, file_path, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (report.id, report.session_id, report.title, report.format, report.content, report.file_path, report.created_at),
        )
        await db.commit()
    finally:
        await db.close()
    return report


async def get_reports(session_id: str) -> list[Report]:
    """Return all reports for a session, newest first.

    The descending order ensures the most recently generated report
    appears at the top of the list in the UI.
    """
    db = await get_db()
    try:
        cursor = await db.execute(
            "SELECT * FROM reports WHERE session_id = ? ORDER BY created_at DESC",
            (session_id,),
        )
        rows = await cursor.fetchall()
        return [Report(**dict(r)) for r in rows]
    finally:
        await db.close()


async def get_report(report_id: str) -> Report | None:
    """Look up a single report by its primary key.

    Returns ``None`` when no matching report exists so the caller can
    respond with an appropriate 404 error.
    """
    db = await get_db()
    try:
        cursor = await db.execute("SELECT * FROM reports WHERE id = ?", (report_id,))
        row = await cursor.fetchone()
        if not row:
            return None
        return Report(**dict(row))
    finally:
        await db.close()


# --------------- Session Context (graph state persistence) ---------------

async def get_session_context(session_id: str) -> dict:
    """Return persisted graph state for a session (search_results, analysis_results, report_data).

    The session context stores the intermediate outputs of the LangGraph
    pipeline (e.g. search results from the GitHub search tool, analysis
    text from the deep-analysis tool) so that follow-up user requests
    within the same session can pick up where the previous invocation
    left off.  For example, the user can say "search for AI frameworks"
    and then "generate a report" in a later message, and the report
    generator will have access to the earlier search results.

    Returns a dict with three keys.  If no context row exists yet for the
    session, all values default to empty strings.
    """
    db = await get_db()
    try:
        cursor = await db.execute(
            "SELECT search_results, analysis_results, report_data FROM session_context WHERE session_id = ?",
            (session_id,),
        )
        row = await cursor.fetchone()
        if not row:
            return {"search_results": "", "analysis_results": "", "report_data": ""}
        return dict(row)
    finally:
        await db.close()


async def save_session_context(session_id: str, search_results: str, analysis_results: str, report_data: str) -> None:
    """Upsert graph state for a session so subsequent requests can use it.

    Uses SQLite's ``INSERT ... ON CONFLICT ... DO UPDATE`` (upsert) pattern:

    * If no row exists for this session_id, a new row is inserted with
      whatever values were supplied.
    * If a row already exists, each column is updated **only if the new
      value is non-empty**.  This conditional update (the ``CASE WHEN``
      clauses) prevents a later request that does not perform a search
      from accidentally blanking out the search_results that a previous
      request populated.

    The helper ``_clean()`` scrubs surrogate characters that LLM outputs
    sometimes contain, which would otherwise cause SQLite encoding errors.
    """
    # Clean surrogate characters that LLMs sometimes produce
    def _clean(s: str) -> str:
        return s.encode("utf-8", errors="replace").decode("utf-8") if s else ""

    db = await get_db()
    try:
        await db.execute(
            """INSERT INTO session_context (session_id, search_results, analysis_results, report_data)
               VALUES (?, ?, ?, ?)
               ON CONFLICT(session_id) DO UPDATE SET
                   search_results = CASE WHEN excluded.search_results != '' THEN excluded.search_results ELSE session_context.search_results END,
                   analysis_results = CASE WHEN excluded.analysis_results != '' THEN excluded.analysis_results ELSE session_context.analysis_results END,
                   report_data = CASE WHEN excluded.report_data != '' THEN excluded.report_data ELSE session_context.report_data END
            """,
            (session_id, _clean(search_results), _clean(analysis_results), _clean(report_data)),
        )
        await db.commit()
    finally:
        await db.close()
