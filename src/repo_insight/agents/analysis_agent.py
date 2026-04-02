"""Analysis Agent -- fetches README and metadata for repos, optionally performs deep analysis.

The Analysis Agent is a specialized node in the LangGraph pipeline responsible
for examining specific GitHub repositories in detail. Its workflow is:

1. **Deep Request Detection**: Determines whether the user explicitly asked for
   a thorough/deep analysis (triggering the expensive `github_deep_analysis`
   tool) or just a basic overview (using the lighter `github_readme` tool).
   Detection uses a two-tier strategy: fast English keyword matching first,
   then an LLM-based semantic check for non-English inputs.

2. **Repository Name Extraction**: Identifies "owner/repo" patterns from the
   user's latest message. If the user does not mention specific repos, it
   falls back to extracting names from the full conversation history and
   prior search results.

3. **Concurrent Fetching**: Uses `asyncio.gather` to fetch data for up to 5
   repositories in parallel, significantly reducing wall-clock time when
   multiple repos need analysis.

4. **Summarization**: Passes the combined raw analysis data through a
   streaming LLM call (with non-streaming fallback) to produce a clear,
   user-friendly summary.

The agent writes its output to both `messages` (for the user) and
`analysis_results` (for the Report Agent to consume downstream).
"""

from __future__ import annotations

import asyncio
import logging
import re

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from repo_insight.agents.state import AgentState
from repo_insight.llm.provider import get_llm
from repo_insight.tools.github_deep_analysis import github_deep_analysis
from repo_insight.tools.github_readme import github_readme

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# English keywords that trigger deep analysis mode. These are checked via
# simple substring matching (case-insensitive) as a fast path before resorting
# to an LLM call. Non-English inputs (e.g., Chinese equivalents of "deep
# analysis") are handled by _is_deep_request() using semantic LLM detection.
# ---------------------------------------------------------------------------
_DEEP_KEYWORDS_EN = [
    "deep analysis", "deep dive", "detailed analysis", "in-depth", "thorough",
    "comprehensive analysis", "dig into", "examine closely",
]


def _extract_repo_names(text: str) -> list[str]:
    """Extract GitHub repository identifiers in "owner/repo" format from text.

    Uses a regex to find patterns like "langchain-ai/langchain" or
    "facebook/react". Results are deduplicated while preserving the order
    of first occurrence.

    Args:
        text: Any string that may contain repository references -- could be
              a user message, search results, or combined conversation history.

    Returns:
        A deduplicated list of "owner/repo" strings found in the text.
    """
    # Match sequences of alphanumeric characters, dots, hyphens, and
    # underscores separated by a single forward slash (the GitHub naming
    # convention for repositories).
    pattern = r'[a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+'
    matches = re.findall(pattern, text)

    # Filter out false positives such as URL fragments that happen to
    # contain a slash (e.g., "http://..." would match the pattern).
    filtered = [m for m in matches if not m.startswith("http") and "/" in m]

    # Deduplicate while preserving the order of first appearance.
    # dict.fromkeys() is a concise Python idiom for ordered deduplication.
    return list(dict.fromkeys(filtered))


async def _summarize_with_fallback(messages: list, tags: list[str]) -> str:
    """Call the LLM to summarize analysis results, with a streaming-to-non-streaming
    fallback strategy.

    This is the same pattern used by the Search Agent: try streaming first for
    a responsive UI experience, fall back to non-streaming if the provider errors.

    Args:
        messages: LangChain message list for the summarization prompt.
        tags:     Config tags; "stream_to_user" enables real-time token display.

    Returns:
        The LLM-generated summary text as a plain string.
    """
    try:
        llm = get_llm(streaming=True).with_config({"tags": tags})
        resp = await llm.ainvoke(messages)
        return resp.content
    except Exception as e:
        logger.warning("Streaming summary failed (%s), falling back to non-streaming", e)
        llm = get_llm(streaming=False)
        resp = await llm.ainvoke(messages)
        return resp.content


async def _is_deep_request(user_text: str) -> bool:
    """Determine whether the user explicitly requested a deep/thorough analysis.

    Uses a two-tier detection strategy:
      1. **Fast path (English keywords)**: Checks for known English phrases
         like "deep analysis", "in-depth", "thorough", etc. via simple
         substring matching. This avoids an LLM call for the common case.
      2. **Semantic path (non-English / ambiguous)**: Falls back to asking
         an LLM to determine intent. This handles languages like Chinese,
         Japanese, etc., where keyword matching would not work. The LLM is
         instructed to be strict -- only explicit requests for deep analysis
         count; casual mentions of a project name do NOT trigger it.

    Deep analysis is intentionally gated behind explicit user request because
    it is significantly more expensive (fetches repo structure, recent commits,
    issues, etc.) than a basic README fetch.

    Args:
        user_text: The full concatenated user text from the conversation.

    Returns:
        True if the user explicitly wants deep analysis, False otherwise.
    """
    lower = user_text.lower()

    # Tier 1: Fast English keyword match -- no LLM call needed.
    if any(kw in lower for kw in _DEEP_KEYWORDS_EN):
        return True

    # Tier 2: Semantic detection via LLM for non-English or ambiguous inputs.
    # Uses temperature=0.0 for deterministic yes/no classification.
    try:
        llm = get_llm(streaming=False, temperature=0.0)
        resp = await llm.ainvoke(
            "Does the following user message EXPLICITLY request a deep / thorough / detailed "
            "analysis of a repository? A simple mention of a project name or asking 'what is X' "
            "does NOT count. The user must clearly ask for in-depth examination. "
            "Answer ONLY 'yes' or 'no'.\n\n"
            f"Message: {user_text}"
        )
        return resp.content.strip().lower().startswith("yes")
    except Exception:
        # If the LLM call fails, default to NOT doing deep analysis (the
        # cheaper, safer option).
        return False


async def analysis_agent(state: AgentState) -> dict:
    """Analysis Agent node -- fetches README/metadata for repos, optionally deep-analyzes.

    This function is registered as a LangGraph node and called when the Lead
    Agent routes an "analyze" task here.

    The analysis flow:
      1. Concatenate all user text to detect deep-analysis intent.
      2. Extract "owner/repo" names from the latest user message (preferred)
         or fall back to the full conversation context.
      3. If a single repo is explicitly targeted, always do deep analysis.
      4. Concurrently fetch data for all identified repos (up to 5).
      5. Summarize the combined results via streaming LLM.

    Args:
        state: The current AgentState containing messages, search_results, etc.

    Returns:
        A partial state dict with:
        - messages: An AIMessage containing the analysis summary.
        - analysis_results: The raw combined analysis data for the Report Agent.
    """
    # -----------------------------------------------------------------
    # Step 1: Collect ALL user text from the conversation to determine
    # whether the user has expressed a desire for deep analysis at any
    # point. We check the entire history (not just the last message)
    # because the deep-analysis request might have been stated earlier.
    # -----------------------------------------------------------------
    user_text = ""
    for msg in state["messages"]:
        if hasattr(msg, "content"):
            user_text += " " + msg.content

    deep = await _is_deep_request(user_text)

    # -----------------------------------------------------------------
    # Step 2a: Extract the last user message. This is used both for
    # targeted repo extraction and for language detection when generating
    # the summary.
    # -----------------------------------------------------------------
    last_user_content = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage) and hasattr(msg, "content"):
            last_user_content = msg.content
            break

    # -----------------------------------------------------------------
    # Step 2b: Try to extract repo names from the LAST user message first.
    # This handles the common case where the user says something like
    # "analyze langchain-ai/langchain in detail" -- we only want to
    # analyze the repo(s) they just mentioned, not everything from prior
    # search results.
    # -----------------------------------------------------------------
    target_repos = _extract_repo_names(last_user_content)

    if target_repos:
        # User explicitly mentioned specific repos -- analyze only those.
        repo_names = target_repos
    else:
        # -----------------------------------------------------------------
        # Fallback: the user did not mention specific repos in their last
        # message (e.g., they said "analyze those repos"). Extract repo
        # names from the full conversation context including prior search
        # results and AI messages.
        # -----------------------------------------------------------------
        context = state.get("search_results", "")
        for msg in state["messages"]:
            if hasattr(msg, "content"):
                context += " " + msg.content
        repo_names = _extract_repo_names(context)

    # -----------------------------------------------------------------
    # Guard: if no repository names could be found anywhere, inform the
    # user and return early. This avoids making pointless API calls.
    # -----------------------------------------------------------------
    if not repo_names:
        return {
            "messages": [AIMessage(content="I couldn't find any specific repository names to analyze. "
                                           "Please mention repo names in owner/repo format (e.g., langchain-ai/langchain) "
                                           "or search for repos first.")],
            "analysis_results": "",
        }

    # -----------------------------------------------------------------
    # Special case: when the user explicitly targeted a single repo,
    # always perform deep analysis. The rationale is that if someone
    # specifically names one repo for analysis, they likely want the
    # full picture, not just a README summary.
    # -----------------------------------------------------------------
    if len(target_repos) == 1:
        deep = True

    # -----------------------------------------------------------------
    # Limit the number of repos to analyze. Deep analysis can be
    # expensive (multiple GitHub API calls per repo), so we cap at 5
    # to avoid excessive API usage and long wait times.
    # -----------------------------------------------------------------
    repo_names = repo_names[:5]

    # -----------------------------------------------------------------
    # Step 3: Concurrently fetch analysis data for all repos. We use
    # asyncio.gather to run all API calls in parallel, significantly
    # reducing total latency compared to sequential fetching.
    # - Deep mode uses `github_deep_analysis` (README + metadata +
    #   recent activity + repo structure).
    # - Normal mode uses `github_readme` (just the README content).
    # return_exceptions=True ensures that one failing repo does not
    # cancel the analysis of all other repos.
    # -----------------------------------------------------------------
    if deep:
        tasks = [github_deep_analysis.ainvoke({"repo_full_name": name}) for name in repo_names]
    else:
        tasks = [github_readme.ainvoke({"repo_full_name": name}) for name in repo_names]

    results = await asyncio.gather(*tasks, return_exceptions=True)

    # -----------------------------------------------------------------
    # Combine the results from all repos into a single string,
    # separated by horizontal rules for readability. Exceptions from
    # individual repos are formatted as error messages rather than
    # propagated, so the user still sees results for the repos that
    # succeeded.
    # -----------------------------------------------------------------
    analysis_parts: list[str] = []
    for name, result in zip(repo_names, results):
        if isinstance(result, Exception):
            analysis_parts.append(f"## {name}\nError: {result}")
        else:
            analysis_parts.append(str(result))

    combined = "\n\n---\n\n".join(analysis_parts)

    # -----------------------------------------------------------------
    # Step 4: Summarize the combined analysis data using a streaming
    # LLM call (with non-streaming fallback). The summary prompt
    # includes the user's original request so the LLM can match the
    # user's language and focus on what was asked.
    # -----------------------------------------------------------------
    summary_text = await _summarize_with_fallback(
        [
            SystemMessage(content="Summarize the following repository analysis results clearly. "
                                  "Highlight key findings for each repository. "
                                  "IMPORTANT: Respond in the SAME LANGUAGE as the user's message below."),
            HumanMessage(content=f"User's request: {last_user_content}\n\n"
                         f"Analysis data:\n\n{combined}"),
        ],
        tags=["stream_to_user"],
    )

    return {
        "messages": [AIMessage(content=summary_text)],
        "analysis_results": combined,
    }
