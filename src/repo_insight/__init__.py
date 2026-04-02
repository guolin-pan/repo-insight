"""
Repo Insight — AI-powered GitHub trending project collector.

This is the top-level package for the Repo Insight application. It exposes
the public surface of the project so that sub-modules (agents, API routes,
storage, CLI, etc.) can be imported via ``from repo_insight import ...``.

The package provides two main entry-points:
  1. A FastAPI web server (see ``main.py``) that serves the API and frontend.
  2. An interactive Rich-based CLI client (see ``cli.py``) for terminal usage.

Configuration is loaded from a ``.env`` file at import time through the
``config`` module, which uses pydantic-settings for validation.
"""
