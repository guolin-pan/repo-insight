"""GitHub README & metadata fetch tool — retrieves README content and repo details.

This LangChain tool fetches a repository's README file and supplementary
metadata (language breakdown, contributor count) in a single call.  The
results are combined into a Markdown-formatted string that the LLM can
present to the user or use as input for deeper analysis.
"""

from __future__ import annotations

import asyncio
import base64

import httpx
from langchain_core.tools import tool

from repo_insight.config import settings


def _github_headers() -> dict[str, str]:
    """Build common HTTP headers for GitHub API requests.

    Returns a dict that always includes the JSON Accept header.  If a
    GitHub personal access token is available in settings, an
    Authorization header is added to increase the API rate limit.
    """
    headers = {"Accept": "application/vnd.github+json"}
    if settings.github_token:
        headers["Authorization"] = f"Bearer {settings.github_token}"
    return headers


async def _fetch_readme(client: httpx.AsyncClient, owner: str, repo: str) -> str:
    """Fetch and decode the repository README content.

    Uses the GitHub "get repository README" endpoint which returns the
    README file regardless of its exact filename (README.md, readme.rst,
    etc.).  The content is Base64-encoded in the API response, so it is
    decoded here before being returned.

    If the README exceeds 8 000 characters it is truncated to keep the
    LLM context window manageable.  A truncation marker is appended so
    the reader knows the text was shortened.

    Returns a descriptive string on error or if no README exists.
    """
    try:
        resp = await client.get(
            f"https://api.github.com/repos/{owner}/{repo}/readme",
            headers=_github_headers(),
        )
        if resp.status_code == 404:
            return "No README found."
        resp.raise_for_status()
        data = resp.json()
        # The README content arrives Base64-encoded; decode it to plain text.
        content = base64.b64decode(data.get("content", "")).decode("utf-8", errors="replace")
        # Truncate very long READMEs to keep context manageable
        if len(content) > 8000:
            content = content[:8000] + "\n\n... [README truncated]"
        return content
    except Exception as e:
        return f"Error fetching README: {e}"


async def _fetch_metadata(client: httpx.AsyncClient, owner: str, repo: str) -> dict:
    """Fetch extra metadata: languages, license, contributors count.

    Two pieces of supplementary information are gathered concurrently:

    1. **Languages** — a dict mapping language names to byte counts
       (e.g. ``{"Python": 123456, "JavaScript": 7890}``).
    2. **Contributor count** — extracted from the ``Link`` pagination
       header of the contributors endpoint.  By requesting only one
       result per page, the ``last`` page number in the Link header
       reveals the total number of contributors without fetching them all.
       If there is no Link header the count is derived from the response
       body length.
    """
    headers = _github_headers()
    results: dict = {}

    async def _languages():
        """Fetch the language breakdown for the repository."""
        try:
            r = await client.get(f"https://api.github.com/repos/{owner}/{repo}/languages", headers=headers)
            r.raise_for_status()
            results["languages"] = r.json()
        except Exception:
            results["languages"] = {}

    async def _contributors():
        """Fetch the approximate contributor count for the repository.

        Requests a single contributor (per_page=1) and inspects the Link
        pagination header to find the last page number, which equals the
        total contributor count.  This avoids downloading the full list.
        """
        try:
            r = await client.get(
                f"https://api.github.com/repos/{owner}/{repo}/contributors",
                params={"per_page": 1, "anon": "true"},
                headers=headers,
            )
            # GitHub returns contributor count in the Link header's last page
            results["contributors_count"] = "N/A"
            if "Link" in r.headers:
                import re
                m = re.search(r'page=(\d+)>; rel="last"', r.headers["Link"])
                if m:
                    results["contributors_count"] = m.group(1)
            else:
                # When there is no Link header the full list fits in one page,
                # so just count the items in the response body.
                results["contributors_count"] = str(len(r.json())) if r.status_code == 200 else "N/A"
        except Exception:
            results["contributors_count"] = "N/A"

    # Run both requests concurrently to reduce overall latency.
    await asyncio.gather(_languages(), _contributors())
    return results


@tool
async def github_readme(repo_full_name: str) -> str:
    """Fetch the README and metadata for a GitHub repository.

    Args:
        repo_full_name: Full repository name in "owner/repo" format (e.g. "langchain-ai/langchain").
    """
    # ── Validate the repo name format ────────────────────────────────
    parts = repo_full_name.split("/")
    if len(parts) != 2:
        return f"Invalid repo name: {repo_full_name}. Expected format: owner/repo"
    owner, repo = parts

    async with httpx.AsyncClient(timeout=30) as client:
        # Fetch README and metadata concurrently to minimize wall-clock time.
        readme_text, metadata = await asyncio.gather(
            _fetch_readme(client, owner, repo),
            _fetch_metadata(client, owner, repo),
        )

    # ── Format the language breakdown as a comma-separated string ────
    languages = metadata.get("languages", {})
    lang_str = ", ".join(f"{k}: {v}" for k, v in languages.items()) if languages else "N/A"

    # ── Assemble the final Markdown output ───────────────────────────
    # Combines language info, contributor count, and the full README into
    # a single Markdown document for the LLM to consume or display.
    return (
        f"## {repo_full_name}\n\n"
        f"**Languages**: {lang_str}\n"
        f"**Contributors**: {metadata.get('contributors_count', 'N/A')}\n\n"
        f"### README\n\n{readme_text}"
    )
