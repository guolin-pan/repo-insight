"""Pydantic request/response models for the API layer.

These models define the exact shape of every JSON body that the API accepts
(request models) and every JSON body it returns (response models).  FastAPI
uses them both for automatic request validation and for OpenAPI documentation
generation.
"""

from __future__ import annotations

from pydantic import BaseModel


# ---- Requests ----
# Request models describe the JSON bodies that clients send to the server.

class ChatRequest(BaseModel):
    """Incoming chat message from the client.

    Fields:
        session_id -- UUID of the existing session this message belongs to.
        message    -- The user's text input to be processed by the agent.
        model      -- Optional LLM model name override.  When provided, the
                      agent will use this model instead of the default one
                      configured in settings.  Useful for letting advanced
                      users choose between e.g. GPT-4 and GPT-3.5.
    """

    session_id: str
    message: str
    model: str | None = None  # optional runtime model override


class SessionCreate(BaseModel):
    """Create a new session.

    Fields:
        title -- Initial display title for the session.  Defaults to
                 "New Chat" and is typically replaced automatically after
                 the first user message.
    """

    title: str = "New Chat"


class SessionUpdate(BaseModel):
    """Update session title.

    Fields:
        title -- The new title to assign to the session.
    """

    title: str


# ---- Responses ----
# Response models describe the JSON bodies that the server sends back.

class SessionResponse(BaseModel):
    """Serialized representation of a session returned to the client.

    Fields:
        id         -- Session UUID.
        title      -- Human-readable session title.
        created_at -- ISO-8601 creation timestamp.
        updated_at -- ISO-8601 timestamp of the last activity.
    """

    id: str
    title: str
    created_at: str
    updated_at: str


class MessageResponse(BaseModel):
    """Serialized representation of a single message returned to the client.

    Fields:
        id         -- Message UUID.
        session_id -- UUID of the parent session.
        role       -- Message author role ("user", "assistant", or "system").
        content    -- Text body of the message.
        metadata   -- JSON string with optional extra information.
        created_at -- ISO-8601 timestamp of when the message was created.
    """

    id: str
    session_id: str
    role: str
    content: str
    metadata: str
    created_at: str


class SessionDetailResponse(BaseModel):
    """Full session view including the session metadata and its message history.

    This is the response model for GET /api/sessions/{session_id}.

    Fields:
        session  -- The session's own metadata (title, timestamps, etc.).
        messages -- Chronologically ordered list of all messages in the session.
    """

    session: SessionResponse
    messages: list[MessageResponse]


class ReportResponse(BaseModel):
    """Serialized representation of a generated report returned to the client.

    Fields:
        id         -- Report UUID.
        session_id -- UUID of the session that produced this report.
        title      -- Human-readable report title.
        format     -- Output format: "html" or "markdown".
        content    -- The full rendered content of the report.
        file_path  -- Filesystem path where the report was saved on disk.
        created_at -- ISO-8601 timestamp of when the report was generated.
    """

    id: str
    session_id: str
    title: str
    format: str
    content: str
    file_path: str
    created_at: str
