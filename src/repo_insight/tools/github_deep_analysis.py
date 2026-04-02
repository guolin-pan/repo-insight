"""Deep analysis tool — fetches repo structure and dependency files, then uses LLM to analyze.

This is the most comprehensive of the GitHub tools.  It collects a wide range
of information about a repository — metadata, file tree, dependency/config
files, README, recent commits, and releases — then sends everything to the
LLM with a structured analysis prompt.  The LLM produces a detailed report
covering the project's purpose, tech stack, architecture, and more.
"""

from __future__ import annotations

import asyncio
import base64

import httpx
from langchain_core.tools import tool

from repo_insight.config import settings
from repo_insight.llm.provider import get_llm


def _github_headers() -> dict[str, str]:
    """Build common HTTP headers for GitHub API requests.

    Always includes the JSON Accept header.  Attaches the configured
    GitHub personal access token (if any) as a Bearer token for higher
    rate limits.
    """
    headers = {"Accept": "application/vnd.github+json"}
    if settings.github_token:
        headers["Authorization"] = f"Bearer {settings.github_token}"
    return headers


# ---------------------------------------------------------------------------
# Well-known dependency and configuration files to look for in a repository.
# These are fetched (when present) and included in the analysis prompt so the
# LLM can identify frameworks, build tools, CI configuration, etc.
# ---------------------------------------------------------------------------
_DEP_FILES = [
    "package.json",
    "requirements.txt",
    "pyproject.toml",
    "setup.py",
    "setup.cfg",
    "go.mod",
    "Cargo.toml",
    "pom.xml",
    "build.gradle",
    "Gemfile",
    "composer.json",
    "Makefile",
    "Dockerfile",
    "docker-compose.yml",
    ".github/workflows/ci.yml",
    ".github/workflows/ci.yaml",
]


async def _fetch_repo_info(client: httpx.AsyncClient, owner: str, repo: str) -> dict:
    """Fetch basic repo metadata (description, stars, forks, topics, license, etc.).

    Calls the "Get a repository" endpoint and extracts the most useful
    fields into a flat dictionary.  Returns an empty dict on any error
    so that downstream code can safely use ``.get()`` with defaults.
    """
    try:
        resp = await client.get(
            f"https://api.github.com/repos/{owner}/{repo}",
            headers=_github_headers(),
        )
        if resp.status_code != 200:
            return {}
        data = resp.json()
        return {
            "description": data.get("description", ""),
            "stars": data.get("stargazers_count", 0),
            "forks": data.get("forks_count", 0),
            "open_issues": data.get("open_issues_count", 0),
            "language": data.get("language", ""),
            "topics": data.get("topics", []),
            "license": (data.get("license") or {}).get("spdx_id", "N/A"),
            "created_at": data.get("created_at", ""),
            "updated_at": data.get("updated_at", ""),
            "default_branch": data.get("default_branch", "main"),
            "homepage": data.get("homepage", ""),
            "archived": data.get("archived", False),
            "size_kb": data.get("size", 0),
        }
    except Exception:
        return {}


async def _fetch_tree(client: httpx.AsyncClient, owner: str, repo: str, recursive: bool = False) -> str:
    """Fetch the file tree of the repository.

    Uses the Git Trees API (``GET /git/trees/HEAD``) to retrieve every
    file and directory path.  When ``recursive=True`` the tree is
    expanded fully, which can be very large for big repos.  To avoid
    overwhelming the LLM context, only the first 120 entries are kept
    and a count of omitted items is appended.

    Directory entries are prefixed with ``[dir]`` for visual clarity.
    """
    try:
        url = f"https://api.github.com/repos/{owner}/{repo}/git/trees/HEAD"
        params = {"recursive": "1"} if recursive else {}
        resp = await client.get(url, headers=_github_headers(), params=params)
        if resp.status_code != 200:
            return "Could not retrieve file tree."
        tree = resp.json().get("tree", [])
        lines = []
        # Limit output to the first 120 items to prevent token overload.
        for item in tree[:120]:
            prefix = "[dir] " if item["type"] == "tree" else ""
            lines.append(f"{prefix}{item['path']}")
        if len(tree) > 120:
            lines.append(f"... and {len(tree) - 120} more files")
        return "\n".join(lines)
    except Exception as e:
        return f"Error fetching tree: {e}"


async def _fetch_file(client: httpx.AsyncClient, owner: str, repo: str, path: str) -> str | None:
    """Fetch a single file's content from the repo (returns None if not found).

    Uses the Repository Contents API to download the file.  The content is
    returned Base64-encoded by GitHub, so it is decoded here.  Files longer
    than 6 000 characters are truncated to keep the analysis prompt within
    reasonable token limits.
    """
    try:
        resp = await client.get(
            f"https://api.github.com/repos/{owner}/{repo}/contents/{path}",
            headers=_github_headers(),
        )
        if resp.status_code != 200:
            return None
        data = resp.json()
        # Skip if the path points to a directory rather than a file.
        if data.get("type") != "file":
            return None
        content = base64.b64decode(data.get("content", "")).decode("utf-8", errors="replace")
        # Truncate large files
        if len(content) > 6000:
            content = content[:6000] + "\n... [truncated]"
        return content
    except Exception:
        return None


async def _fetch_languages(client: httpx.AsyncClient, owner: str, repo: str) -> dict:
    """Fetch language breakdown.

    Returns a dict mapping language names to byte counts (e.g.
    ``{"Python": 123456}``).  Returns an empty dict on any error.
    """
    try:
        resp = await client.get(
            f"https://api.github.com/repos/{owner}/{repo}/languages",
            headers=_github_headers(),
        )
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return {}


async def _fetch_recent_commits(client: httpx.AsyncClient, owner: str, repo: str, count: int = 10) -> str:
    """Fetch recent commit messages to understand development activity.

    Returns a multi-line string with one line per commit showing the date,
    abbreviated SHA, and the first 80 characters of the commit message.
    This gives the LLM a quick view of what has been worked on recently.
    """
    try:
        resp = await client.get(
            f"https://api.github.com/repos/{owner}/{repo}/commits",
            headers=_github_headers(),
            params={"per_page": count},
        )
        if resp.status_code != 200:
            return "Could not fetch commits."
        commits = resp.json()
        lines = []
        for c in commits:
            sha = c.get("sha", "")[:7]
            msg = c.get("commit", {}).get("message", "").split("\n")[0][:80]
            date = c.get("commit", {}).get("committer", {}).get("date", "")[:10]
            lines.append(f"  {date} {sha} {msg}")
        return "\n".join(lines)
    except Exception as e:
        return f"Error: {e}"


async def _fetch_releases(client: httpx.AsyncClient, owner: str, repo: str, count: int = 5) -> str:
    """Fetch recent releases / tags.

    Returns a multi-line string listing the most recent releases with
    their publication date, tag name, and release title.  This helps
    the LLM assess the project's release cadence and versioning scheme.
    """
    try:
        resp = await client.get(
            f"https://api.github.com/repos/{owner}/{repo}/releases",
            headers=_github_headers(),
            params={"per_page": count},
        )
        if resp.status_code != 200:
            return "No releases found."
        releases = resp.json()
        if not releases:
            return "No releases found."
        lines = []
        for r in releases:
            tag = r.get("tag_name", "")
            name = r.get("name", "")
            date = r.get("published_at", "")[:10]
            lines.append(f"  {date} {tag} — {name}")
        return "\n".join(lines)
    except Exception as e:
        return f"Error: {e}"


@tool
async def github_deep_analysis(repo_full_name: str) -> str:
    """Perform a deep analysis of a GitHub repository using LLM.

    Fetches the file tree, dependency files, README, recent commits, releases,
    and language breakdown, then asks the LLM to produce a comprehensive analysis.

    Args:
        repo_full_name: Full repository name in "owner/repo" format.
    """
    # ── Validate the repo name format ────────────────────────────────
    parts = repo_full_name.split("/")
    if len(parts) != 2:
        return f"Invalid repo name: {repo_full_name}. Expected format: owner/repo"
    owner, repo = parts

    async with httpx.AsyncClient(timeout=30) as client:
        # ── Phase 1: Fetch all primary data sources concurrently ─────
        # Running these requests in parallel via asyncio.gather greatly
        # reduces the total wall-clock time compared to sequential calls.
        repo_info, tree_text, languages, readme, commits, releases = await asyncio.gather(
            _fetch_repo_info(client, owner, repo),
            _fetch_tree(client, owner, repo, recursive=True),
            _fetch_languages(client, owner, repo),
            _fetch_file(client, owner, repo, "README.md"),
            _fetch_recent_commits(client, owner, repo),
            _fetch_releases(client, owner, repo),
        )

        # ── Phase 2: Try fetching known dependency / config files ────
        # For each well-known filename in _DEP_FILES, attempt to download
        # its content.  Missing files return None and are silently skipped.
        dep_contents: dict[str, str] = {}
        fetch_tasks = {f: _fetch_file(client, owner, repo, f) for f in _DEP_FILES}
        results = await asyncio.gather(*fetch_tasks.values(), return_exceptions=True)
        for fname, result in zip(fetch_tasks.keys(), results):
            if isinstance(result, str):
                dep_contents[fname] = result

    # ── Build metadata section ───────────────────────────────────────
    # Assemble the repository's key statistics into a bullet-pointed list
    # that the LLM can reference during its analysis.
    meta_lines = []
    if repo_info:
        meta_lines.append(f"- **Description**: {repo_info.get('description', 'N/A')}")
        meta_lines.append(f"- **Stars**: {repo_info.get('stars', 'N/A'):,}")
        meta_lines.append(f"- **Forks**: {repo_info.get('forks', 'N/A'):,}")
        meta_lines.append(f"- **Open Issues**: {repo_info.get('open_issues', 'N/A'):,}")
        meta_lines.append(f"- **Primary Language**: {repo_info.get('language', 'N/A')}")
        meta_lines.append(f"- **License**: {repo_info.get('license', 'N/A')}")
        meta_lines.append(f"- **Topics**: {', '.join(repo_info.get('topics', []))}")
        meta_lines.append(f"- **Created**: {repo_info.get('created_at', '')[:10]}")
        meta_lines.append(f"- **Last Updated**: {repo_info.get('updated_at', '')[:10]}")
        meta_lines.append(f"- **Homepage**: {repo_info.get('homepage', '') or 'N/A'}")
        meta_lines.append(f"- **Size**: {repo_info.get('size_kb', 0):,} KB")
    meta_section = "\n".join(meta_lines) if meta_lines else "N/A"

    # ── Language breakdown ───────────────────────────────────────────
    # Convert raw byte counts into percentages so the LLM (and reader)
    # can quickly understand the language mix of the project.
    lang_section = "N/A"
    if languages:
        total = sum(languages.values())
        lang_lines = []
        for lang, bytes_count in sorted(languages.items(), key=lambda x: -x[1]):
            pct = (bytes_count / total * 100) if total else 0
            lang_lines.append(f"  - {lang}: {pct:.1f}%")
        lang_section = "\n".join(lang_lines)

    # ── Dependency files section ─────────────────────────────────────
    # Include the content of each discovered config/dependency file so
    # the LLM can identify frameworks, libraries, and build tooling.
    dep_section = ""
    if dep_contents:
        dep_section = "\n\n### Dependency / Config Files\n\n"
        for fname, content in dep_contents.items():
            dep_section += f"**{fname}**:\n```\n{content}\n```\n\n"

    # ── README (truncated) ───────────────────────────────────────────
    # Limit the README to 4 000 characters inside the analysis prompt to
    # leave room for the other data sections and the LLM's own output.
    readme_section = readme or "README not available."
    if len(readme_section) > 4000:
        readme_section = readme_section[:4000] + "\n... [truncated]"

    # ── Construct the analysis prompt for the LLM ────────────────────
    # All of the collected data is assembled into a single prompt that
    # instructs the LLM to produce a structured 8-section analysis
    # covering project overview, tech stack, architecture, features,
    # development activity, dependencies, strengths/weaknesses, and
    # recommended use cases.
    prompt = (
        f"Perform a thorough deep analysis of the GitHub repository **{repo_full_name}**.\n\n"
        f"### Repository Metadata\n{meta_section}\n\n"
        f"### Language Breakdown\n{lang_section}\n\n"
        f"### File Tree (recursive)\n```\n{tree_text}\n```\n"
        f"{dep_section}\n"
        f"### README (excerpt)\n{readme_section}\n\n"
        f"### Recent Commits\n{commits}\n\n"
        f"### Recent Releases\n{releases}\n\n"
        "---\n\n"
        "Based on ALL the information above, provide a comprehensive deep analysis:\n"
        "1. **Project Overview** — Purpose and goals\n"
        "2. **Tech Stack** — Languages, frameworks, major libraries and their versions\n"
        "3. **Architecture** — Code organization, module structure, design patterns\n"
        "4. **Key Features** — Notable capabilities and unique selling points\n"
        "5. **Development Activity** — Commit frequency, release cadence, community health\n"
        "6. **Dependencies** — Core dependencies and their roles\n"
        "7. **Strengths & Weaknesses** — Objective assessment\n"
        "8. **Use Cases** — Who should use this and when\n"
    )

    # ── Invoke the LLM (non-streaming) and return its analysis ───────
    # Streaming is disabled because the entire analysis text will be
    # returned as the tool result, not streamed to the user directly.
    llm = get_llm(streaming=False)
    response = await llm.ainvoke(prompt)
    return response.content
