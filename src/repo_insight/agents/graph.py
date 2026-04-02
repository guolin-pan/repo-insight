"""LangGraph StateGraph assembly -- wires the multi-agent workflow together.

This module is the central wiring point for the entire multi-agent system.
It constructs a LangGraph StateGraph by:
  1. Registering each agent as a named node.
  2. Setting the Lead Agent as the entry point (first node to run).
  3. Adding conditional edges from the Lead Agent to route to the correct
     specialized agent based on the classified task type.
  4. Adding conditional edges *after* each specialized agent to either loop
     back to the Lead Agent (when there are remaining pending tasks) or
     terminate the graph.

The compiled graph is exposed as the module-level singleton `app_graph`,
which the rest of the application imports and invokes with user messages.
"""

from __future__ import annotations

from langgraph.graph import END, StateGraph

from repo_insight.agents.analysis_agent import analysis_agent
from repo_insight.agents.lead_agent import lead_agent, route_task
from repo_insight.agents.report_agent import report_agent
from repo_insight.agents.search_agent import search_agent
from repo_insight.agents.state import AgentState


def _should_continue(state: AgentState) -> str:
    """Decide whether the graph should loop back or terminate after a
    specialized agent (Search, Analysis, or Report) has finished executing.

    This function is used as a conditional edge predicate. It checks two
    conditions before allowing the graph to loop:
      1. There must be remaining tasks in `pending_tasks`.
      2. The step counter must not have exceeded the safety limit of 6,
         which prevents infinite loops caused by e.g. repeated task
         classification or cyclic task dependencies.

    Returns:
        "lead_agent" -- if there are pending tasks and the step limit has not
            been reached, routing control back to the Lead Agent so it can
            pop the next task from the queue.
        "__end__"    -- if there are no more pending tasks or the step limit
            has been exceeded, signaling LangGraph to terminate the run.
    """
    pending = state.get("pending_tasks", [])
    if pending and state.get("steps", 0) <= 6:
        return "lead_agent"
    return "__end__"


def build_graph() -> StateGraph:
    """Construct and compile the multi-agent StateGraph.

    The graph implements a hub-and-spoke architecture where the Lead Agent
    acts as the central hub (orchestrator) and three specialized agents
    (Search, Analysis, Report) act as spokes. The flow is:

        User Input --> Lead Agent --> [conditional routing based on current_task]
            |-- "search"  --> Search Agent   --> (back to Lead if pending, else END)
            |-- "analyze" --> Analysis Agent --> (back to Lead if pending, else END)
            |-- "report"  --> Report Agent   --> (back to Lead if pending, else END)
            |-- "chat"    --> END  (Lead Agent already produced a conversational reply)

    For composite requests (e.g., "search for X and generate a report"), the
    Lead Agent decomposes the intent into ["search", "report"], processes
    "search" first, then the graph loops back and the Lead Agent picks up
    "report" from the pending queue.

    Returns:
        A compiled LangGraph StateGraph ready to be invoked with an initial
        AgentState dict.
    """
    # Initialize the state graph with the AgentState schema. LangGraph uses
    # this schema to validate state updates returned by each node.
    graph = StateGraph(AgentState)

    # ---------------------------------------------------------------
    # Register each agent function as a named node in the graph.
    # The string name is used to reference the node in edges and routing.
    # ---------------------------------------------------------------
    graph.add_node("lead_agent", lead_agent)
    graph.add_node("search_agent", search_agent)
    graph.add_node("analysis_agent", analysis_agent)
    graph.add_node("report_agent", report_agent)

    # ---------------------------------------------------------------
    # Set the Lead Agent as the entry point. Every new invocation of the
    # graph begins here. The Lead Agent classifies the user's intent and
    # sets `current_task` to determine which specialized agent runs next.
    # ---------------------------------------------------------------
    graph.set_entry_point("lead_agent")

    # ---------------------------------------------------------------
    # Conditional routing from the Lead Agent to specialized agents.
    # `route_task` inspects `state["current_task"]` and returns the name
    # of the next node. The mapping dict translates those return values
    # into actual graph node references (including the special END node).
    # ---------------------------------------------------------------
    graph.add_conditional_edges(
        "lead_agent",
        route_task,
        {
            "search_agent": "search_agent",
            "analysis_agent": "analysis_agent",
            "report_agent": "report_agent",
            "__end__": END,
        },
    )

    # ---------------------------------------------------------------
    # After each specialized agent completes, decide whether to loop
    # back to the Lead Agent (to process the next pending task) or to
    # end the graph run. The same `_should_continue` predicate is
    # reused for all three agents because the looping logic is identical.
    # ---------------------------------------------------------------
    for agent_name in ("search_agent", "analysis_agent", "report_agent"):
        graph.add_conditional_edges(
            agent_name,
            _should_continue,
            {
                "lead_agent": "lead_agent",
                "__end__": END,
            },
        )

    # Compile the graph into an executable runnable. After compilation the
    # graph structure is frozen and can be invoked with `.ainvoke()`.
    return graph.compile()


# ---------------------------------------------------------------
# Pre-built compiled graph singleton.
# Imported by the application layer to handle incoming user requests.
# Building the graph once at module load avoids repeated compilation overhead.
# ---------------------------------------------------------------
app_graph = build_graph()
