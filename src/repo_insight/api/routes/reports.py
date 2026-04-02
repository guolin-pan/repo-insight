"""Report routes — list, view, and download generated reports.

Provides three endpoints for working with reports that were created by the
``report_generator`` tool during a chat session:

* ``GET /api/sessions/{session_id}/reports``  — list all reports for a session.
* ``GET /api/reports/{report_id}``            — view a single report.
* ``GET /api/reports/{report_id}/download``   — download the report file.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, HTMLResponse

from repo_insight.api.models import ReportResponse
from repo_insight.storage import session_store

router = APIRouter()


@router.get("/api/sessions/{session_id}/reports", response_model=list[ReportResponse])
async def list_reports(session_id: str):
    """List all reports for a given session.

    Returns a JSON array of report metadata objects ordered by creation
    time (newest first).  The front-end uses this to populate the
    reports sidebar within a conversation view.
    """
    reports = await session_store.get_reports(session_id)
    return [ReportResponse(**r.model_dump()) for r in reports]


@router.get("/api/reports/{report_id}")
async def get_report(report_id: str):
    """Get report content. HTML reports rendered directly; Markdown returned as raw text.

    The response type depends on the report's format:
    * **HTML** reports are returned as an ``HTMLResponse`` so the browser
      renders them directly (useful for in-app preview iframes).
    * **Markdown** reports are returned as a JSON ``ReportResponse`` so
      the client can render them with its own Markdown renderer.

    Raises 404 if the report ID does not exist.
    """
    report = await session_store.get_report(report_id)
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")

    # HTML reports are served as rendered HTML for direct browser display.
    if report.format == "html":
        return HTMLResponse(content=report.content)

    # All other formats (primarily Markdown) are returned as structured JSON.
    return ReportResponse(**report.model_dump())


@router.get("/api/reports/{report_id}/download")
async def download_report(report_id: str):
    """Download the report file.

    Serves the report file from disk using FastAPI's ``FileResponse``,
    which triggers a browser download.  The correct MIME type is set
    based on the report format:
    * ``text/html`` for HTML reports.
    * ``text/markdown`` for Markdown reports.

    Returns 404 in three cases:
    1. The report ID does not exist in the database.
    2. No ``file_path`` was recorded for the report (file was never saved).
    3. The file no longer exists on disk (e.g. manually deleted).
    """
    report = await session_store.get_report(report_id)
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")

    # Guard: check that a file path was recorded when the report was created.
    if not report.file_path:
        raise HTTPException(status_code=404, detail="Report file not available")

    # Guard: verify the file still exists on disk before attempting to serve it.
    import os
    if not os.path.exists(report.file_path):
        raise HTTPException(status_code=404, detail="Report file not found on disk")

    # Choose the correct MIME type based on the report's declared format.
    media_type = "text/html" if report.format == "html" else "text/markdown"
    return FileResponse(
        path=report.file_path,
        media_type=media_type,
        filename=os.path.basename(report.file_path),
    )
