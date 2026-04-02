"""FastAPI application entrypoint — mounts routes, CORS, static files, and DB init.

This module wires together every component that the web server needs:
  - API route handlers (chat, sessions, reports)
  - CORS middleware for cross-origin browser requests
  - Static-file serving for the frontend SPA
  - Database initialisation on startup

It also exposes a ``main()`` function that is used as the console-script
entrypoint (defined in pyproject.toml) so that the server can be started
from the command line with ``repo-insight-server``.
"""

from __future__ import annotations

import os

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from repo_insight.api.routes import chat, reports, sessions
from repo_insight.config import settings
from repo_insight.storage.database import init_db

# --------------- FastAPI application instance ---------------
# Create the core FastAPI app with a human-readable title and version string.
# The title appears in the auto-generated OpenAPI docs at /docs.
app = FastAPI(title="Repo Insight", version="0.1.0")

# --------------- CORS middleware ---------------
# Cross-Origin Resource Sharing middleware is required so that the frontend
# (which may be served from a different port during development, e.g.
# localhost:5173 via Vite) can make fetch() calls to this API.
# ``allow_origins=["*"]`` is intentionally permissive for local development;
# in production this should be locked down to the actual frontend origin.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # Accept requests from any origin
    allow_credentials=True,       # Allow cookies / Authorization headers
    allow_methods=["*"],          # Permit all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],          # Permit all custom request headers
)

# --------------- API route registration ---------------
# Each router groups related endpoints under a common prefix (e.g. /chat,
# /sessions, /reports).  Splitting routes into separate modules keeps the
# codebase manageable as the API surface grows.
app.include_router(chat.router)       # Streaming chat / agent invocation
app.include_router(sessions.router)   # CRUD operations on chat sessions
app.include_router(reports.router)    # Report generation and retrieval

# --------------- Static file serving (frontend SPA) ---------------
# Attempt to locate the ``frontend/`` build directory in two places:
#   1. ``<CWD>/frontend`` — the typical location when running from the repo root.
#   2. Two levels up from this Python file — covers installed-package layouts.
# If found, mount it at "/" so that any path not matched by the API routers
# is served as a static file (with ``html=True`` enabling SPA-style fallback
# to index.html for client-side routing).
frontend_dir = os.path.join(os.getcwd(), "frontend")
if not os.path.isdir(frontend_dir):
    frontend_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "frontend"))
if os.path.isdir(frontend_dir):
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")


# --------------- Startup event ---------------
# FastAPI's ``on_event("startup")`` hook runs once when Uvicorn has finished
# binding to the port but before the first request is accepted. We use it to
# create the SQLite tables (if they don't already exist) so the application
# is ready to serve immediately.
@app.on_event("startup")
async def startup():
    """Initialize the database on application startup."""
    await init_db()


# --------------- CLI entrypoint ---------------
def main():
    """CLI entrypoint for running the server.

    Parses command-line arguments (host, port, reload toggle) and then
    delegates to Uvicorn to actually start the ASGI server.  The app is
    referenced as a dotted import string (``"repo_insight.main:app"``) so
    that Uvicorn can re-import it on each reload cycle when ``--reload``
    is active.
    """
    import argparse

    # Build an argument parser with sensible defaults pulled from the
    # application configuration (which itself comes from the .env file).
    parser = argparse.ArgumentParser(description="Repo Insight — AI-powered GitHub project analysis")
    parser.add_argument("--host", default=settings.server_host, help=f"Bind host (default: {settings.server_host})")
    parser.add_argument("--port", "-p", type=int, default=settings.server_port, help=f"Bind port (default: {settings.server_port})")
    parser.add_argument("--reload", action="store_true", default=True, help="Enable auto-reload (default: True)")
    parser.add_argument("--no-reload", action="store_true", help="Disable auto-reload")
    args = parser.parse_args()

    # Launch the Uvicorn ASGI server.  ``reload`` is True by default for a
    # convenient development experience; pass ``--no-reload`` in production.
    uvicorn.run(
        "repo_insight.main:app",
        host=args.host,
        port=args.port,
        reload=not args.no_reload and args.reload,
    )


# Allow running this module directly with ``python -m repo_insight.main``.
if __name__ == "__main__":
    main()
