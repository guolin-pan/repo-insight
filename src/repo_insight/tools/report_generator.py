"""Report generator tool — creates HTML or Markdown reports and persists them.

This LangChain tool takes analysis content (typically produced by the deep
analysis tool or assembled by the agent) and converts it into a finished
report.  Two output formats are supported:

* **HTML** — a fully self-contained HTML page with embedded CSS, rendered
  server-side from Markdown using the ``markdown`` library.  No external
  JS or CDN links are required, so the file works offline.
* **Markdown** — a plain ``.md`` file with a title and generation timestamp.

In both cases the report is:
1. Written to the ``reports/`` directory on disk.
2. Stored in the ``reports`` database table via ``session_store``.
"""

from __future__ import annotations

import os
from datetime import datetime

import markdown as md
from langchain_core.tools import tool

from repo_insight.storage import session_store


@tool
async def report_generator(
    session_id: str,
    title: str,
    content: str,
    report_format: str = "markdown",
) -> str:
    """Generate an HTML or Markdown report from analysis content and save it.

    Args:
        session_id: The chat session this report belongs to.
        title: Report title.
        content: The main body content (analysis data) in Markdown format.
        report_format: Output format — "html" or "markdown".
    """
    # ── Build a date-stamped, filesystem-safe filename ────────────────
    date_str = datetime.utcnow().strftime("%Y%m%d")
    # Build safe filename: replace spaces and special chars with underscores.
    # Only alphanumeric characters, underscores, and hyphens are kept.
    # The title portion is capped at 50 characters to avoid overly long paths.
    safe_title = "".join(c if c.isalnum() or c in "_-" else "_" for c in title)[:50].strip("_")

    # ── Convert content to the requested output format ───────────────
    if report_format == "html":
        # Render the Markdown content into a full HTML page with inline CSS.
        html_content = _markdown_to_html(title, content)
        ext = "html"
        final_content = html_content
    else:
        # For Markdown output, prepend a title heading and a generation
        # timestamp, then include the raw analysis content as-is.
        final_content = f"# {title}\n\n*Generated: {datetime.utcnow().isoformat()}*\n\n{content}"
        ext = "md"

    # ── Save the report file to disk ─────────────────────────────────
    reports_dir = "reports"
    os.makedirs(reports_dir, exist_ok=True)
    file_name = f"{safe_title}_{date_str}.{ext}"
    file_path = os.path.join(reports_dir, file_name)
    # Encode-then-decode to replace any stray surrogate characters that
    # might have been produced by the LLM, which would cause write errors.
    clean_content = final_content.encode("utf-8", errors="replace").decode("utf-8")
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(clean_content)

    # ── Persist the report metadata and content in the database ──────
    # This allows the API to serve the report later without reading the
    # file from disk, and also keeps a record of all generated reports
    # associated with a session.
    report = await session_store.add_report(
        session_id=session_id,
        title=title,
        fmt=report_format,
        content=clean_content,
        file_path=file_path,
    )

    # ── Return a success summary for the LLM to relay to the user ────
    return (
        f"Report generated successfully!\n"
        f"- **Title**: {title}\n"
        f"- **Format**: {report_format}\n"
        f"- **Report ID**: {report.id}\n"
        f"- **File**: {file_path}\n"
    )


def _markdown_to_html(title: str, md_content: str) -> str:
    """Convert Markdown content to a fully self-contained HTML page.

    Uses the Python ``markdown`` library for server-side conversion so the
    resulting HTML file has NO external dependencies (no JS, no CDN links).

    The following Markdown extensions are enabled:
    * ``tables``      — GitHub-style pipe tables.
    * ``fenced_code`` — triple-backtick code blocks.
    * ``codehilite``  — syntax highlighting for code blocks.
    * ``toc``         — table-of-contents generation from headings.
    * ``nl2br``       — convert single newlines to ``<br>`` tags.

    The page includes a comprehensive inline ``<style>`` block that mimics
    GitHub's Markdown rendering aesthetics (fonts, spacing, table borders,
    code block backgrounds, etc.) and includes a ``@media print`` rule for
    clean printouts.
    """
    # Convert Markdown → HTML with common extensions
    body_html = md.markdown(
        md_content,
        extensions=["tables", "fenced_code", "codehilite", "toc", "nl2br"],
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    max-width: 960px;
    margin: 0 auto;
    padding: 2rem;
    line-height: 1.6;
    color: #24292e;
    background: #fff;
}}
h1 {{ border-bottom: 2px solid #eaecef; padding-bottom: 0.3em; margin-top: 1.5em; }}
h2 {{ border-bottom: 1px solid #eaecef; padding-bottom: 0.2em; margin-top: 1.5em; }}
h3 {{ margin-top: 1.2em; }}
pre {{
    background: #f6f8fa;
    padding: 16px;
    border-radius: 6px;
    overflow-x: auto;
    font-size: 0.9em;
    line-height: 1.45;
}}
code {{
    background: #f0f0f0;
    padding: 2px 6px;
    border-radius: 3px;
    font-size: 0.9em;
    font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, monospace;
}}
pre code {{
    background: none;
    padding: 0;
}}
table {{
    border-collapse: collapse;
    width: 100%;
    margin: 1em 0;
}}
th, td {{
    border: 1px solid #dfe2e5;
    padding: 8px 13px;
    text-align: left;
}}
th {{
    background: #f6f8fa;
    font-weight: 600;
}}
tr:nth-child(even) {{
    background: #f9f9f9;
}}
a {{ color: #0366d6; text-decoration: none; }}
a:hover {{ text-decoration: underline; }}
blockquote {{
    border-left: 4px solid #dfe2e5;
    padding: 0 1em;
    color: #6a737d;
    margin: 1em 0;
}}
hr {{
    border: none;
    border-top: 1px solid #eaecef;
    margin: 2em 0;
}}
ul, ol {{
    padding-left: 2em;
    margin: 0.5em 0;
}}
li {{
    margin: 0.25em 0;
}}
img {{
    max-width: 100%;
}}
.meta {{
    color: #6a737d;
    font-size: 0.9em;
    margin-bottom: 2em;
}}
@media print {{
    body {{ max-width: 100%; padding: 1em; }}
}}
</style>
</head>
<body>
<h1>{title}</h1>
<p class="meta">Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}</p>
{body_html}
</body>
</html>"""
