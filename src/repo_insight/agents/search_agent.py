"""Search Agent -- executes GitHub repository searches based on user queries.

The Search Agent is a specialized node in the LangGraph pipeline responsible
for discovering GitHub repositories. Its workflow is:

1. **Tool Binding**: The LLM is given access to the `github_search` tool via
   LangChain's `bind_tools` mechanism. This allows the LLM to decide *what*
   to search for by generating structured tool-call arguments from the user's
   natural-language request.

2. **Tool Execution**: If the LLM produces tool calls, the agent executes them
   against the GitHub API and collects the raw search results.

3. **Summarization with Fallback**: The raw results are fed to a *separate* LLM
   call that produces a user-friendly preliminary overview. Streaming is
   attempted first (so the frontend can display tokens incrementally), with an
   automatic fallback to non-streaming if the provider returns an error.

4. **Direct Response Fallback**: If the LLM chooses not to call any tool (e.g.,
   when the query is too vague), its direct textual response is used as-is.

The agent writes its output to both `messages` (for the user to see) and
`search_results` (for downstream agents like Analysis and Report to consume).
"""

from __future__ import annotations

import logging

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from repo_insight.agents.state import AgentState
from repo_insight.llm.provider import get_llm
from repo_insight.tools.github_search import github_search

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompt that instructs the LLM on how to behave as a Search Agent.
# It tells the LLM to extract keywords, respect user preferences (language,
# sort order), and produce a well-organized preliminary overview. The LLM
# also has access to the github_search tool via bind_tools.
# ---------------------------------------------------------------------------
_SYSTEM_PROMPT = SystemMessage(content="""You are the Search Agent. Your job is to search GitHub repositories based on the user's request.

You have access to the `github_search` tool. Use it to find repositories matching the user's query.

Instructions:
- Extract search keywords from the user's message.
- If the user mentions a specific project or library by name (e.g., "langchain", "fastapi"),
  search for it specifically to find the exact repository.
- If the user specifies a language, include it in the search.
- Default sort by stars unless user specifies otherwise.
- After getting results, provide a clear preliminary overview of the found repositories.
- For specific project inquiries, highlight the main (most-starred) matching repo
  and give useful details: what it does, stars, language, recent activity.
- Present results in a well-organized, informative format.
""")


async def _summarize_with_fallback(messages: list, tags: list[str]) -> str:
    """Call the LLM to summarize search results, with a streaming-to-non-streaming
    fallback strategy.

    Why the fallback? Some OpenAI-compatible endpoints (e.g., Ollama proxies)
    return HTTP 500 errors when streaming large payloads. By trying streaming
    first, we allow the frontend's `astream_events` handler to emit tokens in
    real time for a better user experience. If that fails, we silently retry
    with a non-streaming call which is more reliable.

    Args:
        messages: The list of LangChain message objects to send to the LLM.
                  Typically includes a SystemMessage with summarization
                  instructions and a HumanMessage with the search results.
        tags:     LangChain config tags. The tag "stream_to_user" signals the
                  frontend to display tokens incrementally.

    Returns:
        The LLM-generated summary text as a plain string.
    """
    try:
        # First attempt: streaming LLM for real-time token delivery to the UI.
        llm = get_llm(streaming=True).with_config({"tags": tags})
        resp = await llm.ainvoke(messages)
        return resp.content
    except Exception as e:
        # Second attempt: non-streaming LLM as a reliable fallback.
        logger.warning("Streaming summary failed (%s), falling back to non-streaming", e)
        llm = get_llm(streaming=False)
        resp = await llm.ainvoke(messages)
        return resp.content


async def search_agent(state: AgentState) -> dict:
    """Search Agent node -- invokes the github_search tool and summarizes results.

    This function is registered as a LangGraph node and called when the Lead
    Agent routes a "search" task here. It performs two LLM calls:

    1. A non-streaming call with the github_search tool bound, so the LLM can
       generate structured tool-call arguments (query, language, sort, etc.).
    2. A streaming summarization call that turns raw search results into a
       human-readable preliminary overview.

    Args:
        state: The current AgentState containing the conversation history.

    Returns:
        A partial state dict with:
        - messages: An AIMessage containing the search summary for the user.
        - search_results: The raw search output string for downstream agents.
    """
    # Use a non-streaming LLM for the tool-calling step because tool calls
    # require structured JSON output which does not benefit from streaming.
    llm = get_llm(streaming=False)

    # Bind the github_search tool to the LLM so it can generate tool calls.
    # LangChain's bind_tools mechanism injects the tool's JSON schema into
    # the LLM request, enabling the model to produce structured arguments.
    llm_with_tools = llm.bind_tools([github_search])

    # Only pass the last user message + system prompt for tool invocation.
    # Sending the entire conversation history would create an unnecessarily
    # large payload that can trigger API errors on some providers, and the
    # search tool only needs the current query to do its job.
    last_user = state["messages"][-1] if state["messages"] else HumanMessage(content="")
    messages = [_SYSTEM_PROMPT, last_user]
    response = await llm_with_tools.ainvoke(messages)

    # -----------------------------------------------------------------
    # Branch A: The LLM generated one or more tool calls.
    # Execute each tool call, collect results, then summarize them.
    # -----------------------------------------------------------------
    if response.tool_calls:
        # Execute every tool call the LLM requested. In practice there is
        # usually just one github_search call, but the loop handles the
        # general case of multiple searches (e.g., different keywords).
        tool_results = []
        for tool_call in response.tool_calls:
            result = await github_search.ainvoke(tool_call["args"])
            tool_results.append(result)

        # Merge all tool results into a single string, separated by blank
        # lines, so the summarization LLM sees everything at once.
        search_output = "\n\n".join(tool_results)

        # Build a focused summarization prompt. We deliberately do NOT
        # include the full conversation history here -- only the user's
        # query and the search results -- to keep the payload small and
        # avoid 500 errors on constrained LLM providers.
        summary_prompt = (
            f"The user asked: {last_user.content}\n\n"
            f"The search returned these results:\n\n{search_output}\n\n"
            "Provide a preliminary overview of the found repositories:\n"
            "- For each notable project, briefly explain what it does, its popularity, and key features.\n"
            "- If the user asked about a specific project, focus on that one and give a useful introduction.\n"
            "- Keep the summary informative but concise — this is a preliminary overview.\n"
            "- Let the user know they can ask for a deeper analysis of any specific repository."
        )

        # Summarize the search results using the streaming-with-fallback
        # helper. The "stream_to_user" tag enables real-time token display.
        summary_text = await _summarize_with_fallback(
            [
                SystemMessage(content="Summarize GitHub search results clearly and helpfully. "
                                      "IMPORTANT: Respond in the SAME LANGUAGE as the user's message."),
                HumanMessage(content=summary_prompt),
            ],
            tags=["stream_to_user"],
        )

        return {
            "messages": [AIMessage(content=summary_text)],
            "search_results": search_output,
        }
    else:
        # -----------------------------------------------------------------
        # Branch B: The LLM responded directly without calling any tool.
        # This can happen when the query is too vague for a GitHub search
        # or when the LLM decides it can answer from its own knowledge.
        # We use its direct response as both the user-facing message and
        # the stored search results.
        # -----------------------------------------------------------------
        return {
            "messages": [response],
            "search_results": response.content,
        }
