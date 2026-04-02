"""Report Agent -- generates HTML or Markdown reports from analysis results.

The Report Agent is a specialized node in the LangGraph pipeline responsible
for producing formatted, exportable reports. Its workflow is:

1. **Format Detection**: Scans the conversation history for the keyword "html"
   to decide the output format. Defaults to Markdown if "html" is not found.

2. **Data Source Selection**: Prefers `analysis_results` as the primary data
   source (richer, structured data). Falls back to `search_results` if no
   analysis has been performed yet. If neither is available, returns early
   with an informative message.

3. **Report Content Generation**: Uses a streaming LLM call (with non-streaming
   fallback) to transform the raw data into a well-structured report with
   clear sections, comparisons, and actionable insights.

4. **Content Cleaning**: Sanitizes the LLM output by encoding to UTF-8 with
   surrogate replacement to remove any invalid characters that could break
   downstream file I/O or rendering.

5. **Report Persistence**: Invokes the `report_generator` tool to save the
   report to disk (associated with the current session_id) and returns the
   result (typically a file path or confirmation message).

The agent writes its output to both `messages` (for the user) and
`report_data` (for potential downstream consumption).
"""

from __future__ import annotations

import logging

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from repo_insight.agents.state import AgentState
from repo_insight.llm.provider import get_llm
from repo_insight.tools.report_generator import report_generator

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompt for the Report Agent. Instructs the LLM on how to use the
# report_generator tool and what kind of report structure to produce.
# The LLM does not actually call the tool via bind_tools here -- instead,
# the agent generates report *content* via LLM and then calls the tool
# programmatically in the agent function.
# ---------------------------------------------------------------------------
_SYSTEM_PROMPT = SystemMessage(content="""You are the Report Agent. Your job is to generate structured reports from analysis data.

You have the `report_generator` tool. Use it to create HTML or Markdown reports.

Instructions:
- Use the analysis_results from the conversation state as report content.
- Default to Markdown unless the user asks for HTML.
- Generate a clear, well-structured report title.
- The report should include all repository analysis findings.
""")


async def _generate_with_fallback(messages: list, tags: list[str]) -> str:
    """Call the LLM to generate report content, with a streaming-to-non-streaming
    fallback strategy.

    This follows the same pattern as the Search and Analysis agents: try
    streaming first for a responsive UI experience, fall back to non-streaming
    if the provider returns an error (common with Ollama proxies for large
    payloads).

    Args:
        messages: LangChain message list containing the report generation prompt.
        tags:     Config tags; "stream_to_user" enables real-time token display
                  in the frontend.

    Returns:
        The LLM-generated report content as a plain string.
    """
    try:
        # First attempt: streaming LLM for real-time token delivery.
        llm = get_llm(streaming=True).with_config({"tags": tags})
        resp = await llm.ainvoke(messages)
        return resp.content
    except Exception as e:
        # Second attempt: non-streaming LLM as a reliable fallback.
        logger.warning("Streaming report generation failed (%s), falling back to non-streaming", e)
        llm = get_llm(streaming=False)
        resp = await llm.ainvoke(messages)
        return resp.content


async def report_agent(state: AgentState) -> dict:
    """Report Agent node -- generates a formatted report and saves it to disk.

    This function is registered as a LangGraph node and called when the Lead
    Agent routes a "report" task here. It orchestrates the full report
    generation pipeline: format detection, content generation, cleaning,
    and persistence.

    Args:
        state: The current AgentState containing messages, analysis_results,
               search_results, and session_id.

    Returns:
        A partial state dict with:
        - messages: An AIMessage telling the user the report was generated
                    (or explaining why it could not be).
        - report_data: The result from the report_generator tool (typically
                       a file path or confirmation string).
    """
    # -----------------------------------------------------------------
    # Read existing data from the state. The Report Agent consumes data
    # produced by upstream agents (Analysis and/or Search). The session_id
    # is needed to associate the report file with the correct session.
    # -----------------------------------------------------------------
    analysis = state.get("analysis_results", "")
    search = state.get("search_results", "")
    session_id = state.get("session_id", "")

    # -----------------------------------------------------------------
    # Format detection: check the LAST user message for the keyword
    # "html" (case-insensitive).  Only the most recent request matters;
    # scanning the full history would cause a previous "/report html"
    # to sticky-override all subsequent "/report markdown" requests.
    # -----------------------------------------------------------------
    last_user_msg = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage) and hasattr(msg, "content"):
            last_user_msg = msg.content
            break
    fmt = "html" if "html" in last_user_msg.lower() else "markdown"

    # -----------------------------------------------------------------
    # Data source selection: prefer analysis_results (richer, structured
    # data from deep analysis or README fetches). Fall back to
    # search_results if no analysis has been done. If neither exists,
    # inform the user and return early.
    # -----------------------------------------------------------------
    source_data = analysis or search
    if not source_data:
        return {
            "messages": [AIMessage(content="No analysis or search data available to generate a report. "
                                           "Please search for or analyze some repositories first.")],
            "report_data": "",
        }

    # -----------------------------------------------------------------
    # Language detection: reuse last_user_msg (already extracted above)
    # to determine the report language.
    # -----------------------------------------------------------------

    # Build the report generation prompt with the user's request and
    # all available source data. The LLM is asked to structure the
    # report with clear sections and actionable insights.
    report_prompt = (
        f"User's request: {last_user_msg}\n\n"
        f"Based on the following data, create a comprehensive report:\n\n{source_data}\n\n"
        "Structure the report with clear sections, comparisons if multiple repos, "
        "and actionable insights. Format as Markdown."
    )

    # -----------------------------------------------------------------
    # Generate the report body using streaming LLM (with fallback).
    # The "stream_to_user" tag allows the frontend to show the report
    # content as it is being generated.
    # -----------------------------------------------------------------
    report_text = await _generate_with_fallback(
        [
            SystemMessage(content="Generate a well-structured report from repository data. "
                                  "IMPORTANT: Write the report in the SAME LANGUAGE as the user's request."),
            HumanMessage(content=report_prompt),
        ],
        tags=["stream_to_user"],
    )

    # -----------------------------------------------------------------
    # Content cleaning: encode the LLM output to UTF-8 with "replace"
    # error handling to eliminate any surrogate characters or other
    # invalid byte sequences. Some LLM providers occasionally emit
    # surrogates (lone \uD800-\uDFFF code points) which would cause
    # errors when writing to files or rendering in browsers.
    # -----------------------------------------------------------------
    report_content = report_text.encode("utf-8", errors="replace").decode("utf-8")

    # -----------------------------------------------------------------
    # Persist the report using the report_generator tool. This tool
    # saves the report to disk under the session directory and returns
    # a result string (typically the file path or a success message).
    # If the tool raises an exception (e.g., disk full, permission
    # error), we catch it and return a user-friendly error message
    # instead of crashing the entire graph.
    # -----------------------------------------------------------------
    # -----------------------------------------------------------------
    # Derive a meaningful report title from the conversation context.
    # Use the first user message (topic) to create a descriptive title
    # rather than a generic hardcoded string.
    # -----------------------------------------------------------------
    first_user_content = ""
    for msg in state["messages"]:
        if isinstance(msg, HumanMessage) and hasattr(msg, "content"):
            first_user_content = msg.content
            break
    # Build title: truncate to 60 chars, strip trailing whitespace
    report_title = first_user_content.strip()[:60].strip()
    if not report_title:
        report_title = "Repository Report"

    try:
        result = await report_generator.ainvoke({
            "session_id": session_id,
            "title": report_title,
            "content": report_content,
            "report_format": fmt,
        })
    except Exception as e:
        result = f"Error generating report: {e}"

    return {
        "messages": [AIMessage(content=f"Report generated!\n\n{result}")],
        "report_data": result,
    }
