"""Application configuration via pydantic-settings, loaded from .env file.

All values MUST be defined in the .env file. Missing fields cause an
immediate validation error at startup.

This module is imported very early (by almost every other module in the
package), so it intentionally avoids heavy dependencies. It reads a ``.env``
file in the current working directory and maps each key to a typed Python
attribute on the ``Settings`` class.
"""

import sys

from pydantic import ValidationError
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Central configuration for Repo Insight application.

    Each attribute corresponds to an environment variable (case-insensitive).
    For example, ``github_token`` is populated from the ``GITHUB_TOKEN``
    variable in the ``.env`` file.  pydantic-settings performs type coercion
    automatically (e.g. ``server_port`` is cast to ``int``).
    """

    # --------------- GitHub Section ---------------
    # A GitHub personal access token.  Without it the GitHub API allows only
    # 60 unauthenticated requests per hour; with a token the limit rises to
    # 5,000 requests per hour, which is essential for trending-repo fetching.
    github_token: str

    # --------------- LLM Section ---------------
    # These three settings configure the Large Language Model backend.
    # They are compatible with any provider that exposes an OpenAI-compatible
    # HTTP API (OpenAI itself, Ollama, DeepSeek, Azure OpenAI, etc.).
    llm_base_url: str   # Base URL of the LLM API (e.g. "https://api.openai.com/v1")
    llm_model: str      # Model identifier to use (e.g. "gpt-4o", "deepseek-chat")
    llm_api_key: str    # Secret API key for authenticating with the LLM provider

    # --------------- Database Section ---------------
    # File-system path for the SQLite database that stores sessions, messages,
    # and cached analysis results.  A relative path is resolved from the CWD.
    db_path: str

    # --------------- Server Section ---------------
    # Host and port on which the FastAPI/Uvicorn server will listen.
    server_host: str    # Bind address (e.g. "0.0.0.0" or "127.0.0.1")
    server_port: int    # TCP port number (e.g. 8000)

    # pydantic-settings ``model_config`` — tells the library to read values
    # from a ``.env`` file in the working directory, encoded as UTF-8.
    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


# --------------- Module-level initialisation ---------------
# Attempt to instantiate the settings as soon as this module is imported.
# If any required field is missing from the ``.env`` file, pydantic raises a
# ``ValidationError``.  We catch it here to print a user-friendly message
# listing exactly which variables are absent, then exit immediately so the
# application never starts in an incomplete configuration state.
try:
    settings = Settings()  # type: ignore[call-arg]
except ValidationError as e:
    print("ERROR: Missing required configuration in .env file:\n")
    for err in e.errors():
        field = err["loc"][0]
        print(f"  - {str(field).upper()} is required")
    print(f"\nPlease copy .env.example to .env and fill in all values.")
    sys.exit(1)
