"""GitHub Search tool — searches repositories via GitHub REST API.

This LangChain tool allows the LLM agent to search GitHub for repositories
matching user-specified keywords.  It builds a query string compatible with
the GitHub Search API, executes the HTTP request (authenticating when a
token is configured), and formats the results into a human-readable string
that the LLM can present to the user or feed into downstream analysis tools.
"""

from __future__ import annotations

import httpx
from langchain_core.tools import tool

from repo_insight.config import settings


@tool
async def github_search(
    keywords: str,
    language: str = "",
    sort: str = "stars",
    limit: int = 10,
) -> str:
    """Search GitHub repositories by keywords.

    Args:
        keywords: Search query string (e.g. "AI Agent framework").
        language: Filter by programming language (e.g. "python"). Empty means any.
        sort: Sort field — "stars", "forks", or "updated".
        limit: Maximum number of results to return (max 30).
    """
    # ── Build the GitHub search query ────────────────────────────────
    # Start with the raw keywords and optionally append a language qualifier.
    # GitHub's search syntax uses "language:<name>" to filter by programming
    # language (e.g. "AI Agent framework language:python").
    query = keywords
    if language:
        query += f" language:{language}"

    # ── Assemble query parameters for the REST API ───────────────────
    # - ``q``        : the search query string built above.
    # - ``sort``     : field to sort results by (stars, forks, or updated).
    # - ``order``    : always descending so the most popular / recent come first.
    # - ``per_page`` : number of results, capped at 30 (GitHub's max per page).
    params = {
        "q": query,
        "sort": sort,
        "order": "desc",
        "per_page": min(limit, 30),
    }

    # ── Set up HTTP headers ──────────────────────────────────────────
    # The Accept header requests the v3 JSON format.  If a personal access
    # token is configured, it is attached as a Bearer token to increase the
    # rate limit from 10 requests/minute (unauthenticated) to 30 requests/minute.
    headers = {"Accept": "application/vnd.github+json"}
    if settings.github_token:
        headers["Authorization"] = f"Bearer {settings.github_token}"

    # ── Execute the search request ───────────────────────────────────
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(
            "https://api.github.com/search/repositories",
            params=params,
            headers=headers,
        )
        resp.raise_for_status()
        data = resp.json()

    items = data.get("items", [])
    if not items:
        return f"No repositories found for query: {keywords}"

    # ── Format results as a readable string for the LLM ──────────────
    # Each repository is displayed with its rank, full name, star count,
    # fork count, primary language, description, URL, topics, and last
    # update date.  This format gives the LLM (and the end user) enough
    # context to decide which repositories to investigate further.
    lines: list[str] = [f"Found {len(items)} repositories for \"{keywords}\":\n"]
    for i, repo in enumerate(items, 1):
        lines.append(
            f"{i}. **{repo['full_name']}** ⭐ {repo['stargazers_count']:,} | "
            f"Forks: {repo['forks_count']:,} | Language: {repo.get('language', 'N/A')}\n"
            f"   {repo.get('description', 'No description')}\n"
            f"   URL: {repo['html_url']}\n"
            f"   Topics: {', '.join(repo.get('topics', []))}\n"
            f"   Updated: {repo.get('updated_at', 'N/A')}"
        )
    return "\n".join(lines)
