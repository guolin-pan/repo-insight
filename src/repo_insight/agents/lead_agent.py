"""Lead Agent -- understands user intent, decomposes tasks, and routes to specialized agents.

The Lead Agent is the orchestrator of the multi-agent pipeline. It is the first
node that runs on every graph invocation and has two primary responsibilities:

1. **Intent Classification**: On the first invocation for a user message, the
   Lead Agent uses an LLM (with a zero-temperature classification prompt) to
   determine what the user wants. The classified intent is one or more keywords
   from {"search", "analyze", "report", "chat"}.

2. **Task Queue Management**: When the LLM returns multiple keywords (e.g.,
   "search,report"), the Lead Agent treats them as a task queue. It pops the
   first task as `current_task` (which drives routing in graph.py) and stores
   the rest in `pending_tasks`. On subsequent invocations (after a specialized
   agent completes and the graph loops back), it simply pops the next task.

3. **Chat Fallback**: If the classified intent is "chat" (greetings, help
   requests, or ambiguous queries that mention no project), the Lead Agent
   generates a conversational reply itself using a streaming LLM, with a
   non-streaming fallback in case of API errors.
"""

from __future__ import annotations

import logging

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from repo_insight.agents.state import AgentState
from repo_insight.llm.provider import get_llm

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Classification prompt -- fed to a zero-temperature LLM to extract the user's
# intent as one or more comma-separated keywords. The prompt is designed to
# be aggressive about classifying any mention of a project/library name as
# "search" so the system proactively looks it up on GitHub. "analyze" is only
# triggered when the user explicitly asks for deeper examination. "chat" is
# the catch-all for greetings and generic help questions.
# ---------------------------------------------------------------------------
_CLASSIFY_PROMPT = SystemMessage(content="""You are a task classifier for a Repo Insight system.

This system is dedicated to GitHub projects. When a user mentions ANY project name,
library name, framework name, or tool name (e.g. "langchain", "fastapi", "react",
"kubernetes"), ASSUME it refers to a GitHub project and classify as "search" so the
system can proactively look it up.

Given a user message (in any language), identify ALL intents by understanding SEMANTICS.
Output one or more keywords separated by commas.

Available keywords:
- "search" — user mentions a project/library/tool name, wants to find/discover/collect
  GitHub repositories, or references a project not yet discussed. ANY mention of a
  concrete project or technology name that hasn't been analyzed yet should trigger search.
- "analyze" — user EXPLICITLY asks for deep/detailed/thorough analysis of a specific
  repository that was already found. Only classify as analyze when the user clearly
  requests deeper examination (e.g. "analyze it in detail", "dig deeper into X").
- "report" — user wants to generate a summary report or export results
- "chat" — ONLY for pure greetings, help requests, or questions that do NOT mention
  any project/library/tool name at all

Examples (do NOT output these, just learn the pattern):
- "tell me about langchain" -> search
- "what is fastapi?" -> search
- "find me 5 AI projects" -> search
- "analyze langchain-ai/langchain in detail" -> analyze
- "deep dive into that repo" -> analyze
- "search for react projects and create a report" -> search,report
- "hello" -> chat
- "what can you do?" -> chat

CRITICAL: Output ONLY the keyword(s), comma-separated, nothing else.
""")

# ---------------------------------------------------------------------------
# Chat system prompt -- used when the Lead Agent handles the "chat" intent
# itself (no specialized agent needed). Instructs the LLM to respond
# helpfully, explain capabilities, and match the user's language.
# ---------------------------------------------------------------------------
_CHAT_SYSTEM_PROMPT = SystemMessage(content="""You are a helpful Repo Insight assistant specialized in GitHub projects.

Your primary purpose is helping users discover and understand GitHub projects. You can:
- **Search** GitHub for trending/popular repositories by keyword or topic
- **Analyze** specific repositories (README, metadata, deep analysis)
- **Generate reports** summarizing analysis results (HTML or Markdown)

When a user mentions any project, library, framework, or tool by name, you should
suggest searching for it on GitHub. Proactively offer to look it up.

When you present search results, give a brief preliminary overview. Only do
in-depth analysis when the user explicitly asks for it.

IMPORTANT: Always respond in the SAME LANGUAGE as the user's message.
If the user writes in Chinese, respond entirely in Chinese.
If the user writes in English, respond entirely in English.

Respond naturally. If the user's intent is unclear, explain what you can do and ask for clarification.""")

# ---------------------------------------------------------------------------
# Task priority order -- when the classification LLM returns multiple keywords,
# they are sorted into this canonical order so that prerequisite tasks run
# first. For example, "search" must happen before "analyze" (which needs
# repo names), and "analyze" should happen before "report" (which needs
# analysis data).
# ---------------------------------------------------------------------------
_TASK_PRIORITY = ["search", "analyze", "report"]


async def lead_agent(state: AgentState) -> dict:
    """Lead Agent node -- classifies user intent, decomposes composite tasks, and routes.

    This function is called by LangGraph as a graph node. It is invoked at least
    once per user message (for initial classification) and may be invoked again
    if the graph loops back after a specialized agent finishes and there are
    still pending tasks in the queue.

    On first invocation:
        - Uses the classification LLM to determine intent keywords.
        - Splits composite intents into a task queue.
        - For "chat" intent, generates a conversational reply directly.

    On subsequent invocations (pending tasks exist):
        - Pops the next task from the pending_tasks queue.
        - Does NOT re-classify; simply continues processing the queue.

    Args:
        state: The current AgentState, containing messages, pending_tasks,
               steps counter, and results from previous agents.

    Returns:
        A partial state dict with updated current_task, pending_tasks, steps,
        and optionally new messages (for chat responses).
    """
    # Increment the step counter every time the Lead Agent is invoked.
    # This counter is checked by _should_continue() in graph.py and by
    # route_task() below to enforce the maximum iteration limit.
    steps = state.get("steps", 0) + 1
    pending = list(state.get("pending_tasks", []))

    # -----------------------------------------------------------------
    # Fast path: if there are already pending tasks left over from a
    # previous classification, simply pop the next one and route to the
    # appropriate specialized agent. No LLM call needed.
    # -----------------------------------------------------------------
    if pending:
        task = pending.pop(0)
        return {
            "current_task": task,
            "pending_tasks": pending,
            "steps": steps,
        }

    # -----------------------------------------------------------------
    # First invocation for this user message -- classify the intent.
    # We use temperature=0.0 and non-streaming to get deterministic,
    # reliable classification output (just keyword strings).
    # -----------------------------------------------------------------
    classify_llm = get_llm(streaming=False, temperature=0.0)

    # Extract the most recent user message for classification. If the
    # messages list is somehow empty, fall back to an empty HumanMessage
    # to avoid index errors.
    last_user_msg = state["messages"][-1] if state["messages"] else HumanMessage(content="")

    # Send only the classification prompt + the user message to the LLM.
    # Including the full conversation history here would be wasteful and
    # could confuse the classifier.
    classify_resp = await classify_llm.ainvoke([_CLASSIFY_PROMPT, last_user_msg])
    content = classify_resp.content.strip().lower()

    # -----------------------------------------------------------------
    # Parse the comma-separated keyword string returned by the LLM into
    # an ordered list of tasks. We iterate through _TASK_PRIORITY so that
    # tasks are always ordered as search -> analyze -> report, regardless
    # of the order the LLM returned them in.
    # -----------------------------------------------------------------
    tasks = []
    for keyword in _TASK_PRIORITY:
        if keyword in content:
            tasks.append(keyword)

    # If the LLM did not produce any recognized task keywords (e.g., it
    # returned only "chat" or something unexpected), default to "chat"
    # so the user still gets a helpful response.
    if not tasks:
        tasks = ["chat"]

    # Pop the first task as the current one to execute; the rest go into
    # the pending queue for future loop iterations.
    task = tasks.pop(0)
    updates: dict = {
        "current_task": task,
        "pending_tasks": tasks,
        "steps": steps,
    }

    # -----------------------------------------------------------------
    # Handle the "chat" intent directly within the Lead Agent. No
    # specialized agent is needed -- just generate a conversational reply.
    # We attempt streaming first (tagged with "stream_to_user" so the UI
    # can display tokens in real time). If streaming fails (some Ollama
    # proxies return 500 for streaming), we fall back to a non-streaming
    # invocation.
    # -----------------------------------------------------------------
    if task == "chat":
        try:
            # Streaming LLM with higher temperature for more natural chat.
            # The "stream_to_user" tag tells the frontend to display tokens
            # incrementally as they arrive.
            chat_llm = get_llm(streaming=True, temperature=0.7).with_config({"tags": ["stream_to_user"]})
            chat_msgs = [_CHAT_SYSTEM_PROMPT] + state["messages"]
            response = await chat_llm.ainvoke(chat_msgs)
        except Exception as e:
            # Fallback: non-streaming invocation. The user will see the
            # complete response at once instead of token-by-token.
            logger.warning("Streaming chat failed (%s), falling back to non-streaming", e)
            chat_llm = get_llm(streaming=False, temperature=0.7)
            chat_msgs = [_CHAT_SYSTEM_PROMPT] + state["messages"]
            response = await chat_llm.ainvoke(chat_msgs)

        # Wrap the response in an AIMessage and append it to the conversation
        # history via the state update.
        updates["messages"] = [AIMessage(content=response.content)]

    return updates


def route_task(state: AgentState) -> str:
    """Conditional edge function -- determines which node runs next based on
    the `current_task` field set by the Lead Agent.

    This function is referenced in graph.py as the predicate for
    `add_conditional_edges` from the Lead Agent node. Its return value is
    looked up in the routing map to select the next graph node.

    Safety: if the step counter exceeds 6, the function unconditionally
    returns "__end__" to prevent runaway loops regardless of the current task.

    Args:
        state: The current AgentState (read-only in this context).

    Returns:
        One of "search_agent", "analysis_agent", "report_agent", or "__end__".
    """
    # Hard safety limit: terminate the graph if too many iterations have occurred.
    # This guards against pathological cases where tasks keep being generated.
    if state.get("steps", 0) > 6:
        return "__end__"

    # Route to the appropriate specialized agent based on the classified task.
    task = state.get("current_task", "chat")
    if task == "search":
        return "search_agent"
    elif task == "analyze":
        return "analysis_agent"
    elif task == "report":
        return "report_agent"

    # For "chat" and any unrecognized task type, end the graph. The Lead Agent
    # has already produced a conversational reply for "chat", so no further
    # processing is needed.
    return "__end__"
