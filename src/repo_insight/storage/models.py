"""Data models for SQLite tables — sessions, messages, and reports.

These Pydantic models serve as the canonical representation of the three core
database entities (Session, Message, Report).  They are used both when writing
rows to SQLite (via session_store) and when returning data from the API layer.
Each model auto-generates a UUID primary key and an ISO-8601 timestamp at
instantiation time so callers do not need to supply them manually.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, Field


def _uuid() -> str:
    """Generate a new random UUID4 string to use as a primary key.

    UUID4 is chosen because it does not depend on ordering or machine
    identity, which keeps the IDs globally unique across distributed
    or replicated setups.
    """
    return str(uuid.uuid4())


def _now() -> str:
    """Return the current UTC time as an ISO-8601 string.

    All timestamps in the database are stored as plain text in this
    format so they are human-readable and sortable as strings.
    """
    return datetime.utcnow().isoformat()


class Session(BaseModel):
    """A chat session grouping related messages.

    A session acts as a conversation container.  Every message and every
    generated report belongs to exactly one session.  The ``updated_at``
    field is bumped each time a new message is added so the UI can sort
    sessions by recency.

    Fields:
        id         -- Unique identifier (UUID4).  Auto-generated.
        title      -- Human-readable title shown in the sidebar / session
                      list.  Defaults to "New Chat" and is later replaced
                      by an auto-generated summary of the first user message.
        created_at -- ISO-8601 timestamp of when the session was created.
        updated_at -- ISO-8601 timestamp of the last activity (message or
                      title change) inside the session.
    """

    id: str = Field(default_factory=_uuid)
    title: str = "New Chat"
    created_at: str = Field(default_factory=_now)
    updated_at: str = Field(default_factory=_now)


class Message(BaseModel):
    """A single chat message within a session.

    Messages are the atomic units of conversation.  The ``role`` field
    distinguishes between user input, assistant (LLM) replies, and
    system-level messages.  An optional JSON ``metadata`` blob can store
    extra information such as tool-call results or token-usage stats.

    Fields:
        id         -- Unique identifier (UUID4).  Auto-generated.
        session_id -- Foreign key pointing to the owning Session.
        role       -- One of "user", "assistant", or "system".
        content    -- The text body of the message.
        metadata   -- Arbitrary JSON string for extra data (default "{}").
        created_at -- ISO-8601 timestamp of when the message was persisted.
    """

    id: str = Field(default_factory=_uuid)
    session_id: str
    role: str  # user / assistant / system
    content: str
    metadata: str = "{}"  # JSON string
    created_at: str = Field(default_factory=_now)


class Report(BaseModel):
    """A generated report (HTML or Markdown) linked to a session.

    Reports are the deliverable output of the analysis pipeline.  After
    the LLM produces analysis content, the report_generator tool converts
    it into either a self-contained HTML page or a Markdown file, writes
    it to disk, and stores a reference here for later retrieval or download.

    Fields:
        id         -- Unique identifier (UUID4).  Auto-generated.
        session_id -- Foreign key pointing to the session that triggered the
                      report generation.
        title      -- Human-readable report title (e.g. "Deep Analysis of
                      langchain-ai/langchain").
        format     -- Output format: "html" or "markdown".
        content    -- The full rendered content (HTML markup or Markdown text).
        file_path  -- Filesystem path where the report was saved.  Empty if
                      the file has not been persisted to disk yet.
        created_at -- ISO-8601 timestamp of when the report was generated.
    """

    id: str = Field(default_factory=_uuid)
    session_id: str
    title: str = ""
    format: str = "markdown"  # html / markdown
    content: str = ""
    file_path: str = ""
    created_at: str = Field(default_factory=_now)
