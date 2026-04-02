"""Session CRUD routes.

Provides RESTful endpoints for managing chat sessions:

* ``GET    /api/sessions``              — list all sessions.
* ``POST   /api/sessions``              — create a new session.
* ``GET    /api/sessions/{session_id}`` — get session details with messages.
* ``PUT    /api/sessions/{session_id}`` — rename a session.
* ``DELETE /api/sessions/{session_id}`` — delete a session and its data.

Each endpoint delegates persistence to ``session_store`` and returns Pydantic
response models so FastAPI can auto-generate OpenAPI docs and validate output.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from repo_insight.api.models import (
    MessageResponse,
    SessionCreate,
    SessionDetailResponse,
    SessionResponse,
    SessionUpdate,
)
from repo_insight.storage import session_store

router = APIRouter()


@router.get("/api/sessions", response_model=list[SessionResponse])
async def list_sessions():
    """List all chat sessions ordered by most recently updated.

    Returns a flat JSON array of session objects.  The ``updated_at``
    descending ordering is handled by the store layer so the UI can
    render the list directly without additional client-side sorting.
    """
    sessions = await session_store.list_sessions()
    return [SessionResponse(**s.model_dump()) for s in sessions]


@router.post("/api/sessions", response_model=SessionResponse, status_code=201)
async def create_session(body: SessionCreate):
    """Create a new chat session.

    Accepts an optional ``title`` in the request body (defaults to
    "New Chat").  Returns the newly created session with a 201 status
    code.  The session's UUID is generated server-side.
    """
    session = await session_store.create_session(title=body.title)
    return SessionResponse(**session.model_dump())


@router.get("/api/sessions/{session_id}", response_model=SessionDetailResponse)
async def get_session(session_id: str):
    """Get session details including full message history.

    Returns a composite object containing both the session metadata and
    its chronologically ordered messages.  This is the primary endpoint
    the front-end calls when the user opens an existing conversation.
    Raises 404 if the session ID does not exist.
    """
    session = await session_store.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    messages = await session_store.get_messages(session_id)
    return SessionDetailResponse(
        session=SessionResponse(**session.model_dump()),
        messages=[MessageResponse(**m.model_dump()) for m in messages],
    )


@router.put("/api/sessions/{session_id}", response_model=SessionResponse)
async def update_session(session_id: str, body: SessionUpdate):
    """Update session title.

    Replaces the session's title with the value supplied in the request
    body and refreshes the ``updated_at`` timestamp.  Returns the
    updated session object.  Raises 404 if no session with the given
    ID exists.
    """
    session = await session_store.update_session(session_id, body.title)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return SessionResponse(**session.model_dump())


@router.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a session and all associated messages and reports.

    Cascading deletes remove every message, report, and session-context
    row linked to this session.  Returns ``{"status": "deleted"}`` on
    success.  Raises 404 if the session was not found (or was already
    deleted).
    """
    deleted = await session_store.delete_session(session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"status": "deleted"}
