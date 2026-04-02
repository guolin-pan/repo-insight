"""Chat route — POST /api/chat with SSE streaming response.

This module implements the main chat endpoint.  When the client sends a
message, the endpoint:

1. Validates the session exists.
2. Persists the user message to the database.
3. Optionally auto-generates a session title from the first user message.
4. Reconstructs the full conversation history from the database.
5. Loads any previously persisted graph context (search / analysis results).
6. Builds and runs the LangGraph agent, streaming LLM tokens back to the
   client as Server-Sent Events (SSE).
7. After streaming completes, persists the assistant reply and any updated
   graph context back to the database.

If the streaming path produces no content (e.g. the LLM responded only
through tool calls), the endpoint falls back to a non-streaming
``graph.ainvoke`` call, and if that still yields nothing, it returns a
generic apology message.
"""

from __future__ import annotations

import json
import logging

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage

from repo_insight.agents.graph import build_graph
from repo_insight.api.models import ChatRequest
from repo_insight.storage import session_store

router = APIRouter()
logger = logging.getLogger(__name__)


async def _auto_title(session_id: str, user_message: str):
    """Auto-generate a session title from the first user message.

    Only fires when the session still has the default title "New Chat".
    The title is capped at 50 characters with an ellipsis appended if
    truncation was necessary.  This gives the user a quick preview of
    what each conversation is about in the session list.
    """
    session = await session_store.get_session(session_id)
    if session and session.title == "New Chat":
        title = user_message.strip()[:50]
        if len(user_message.strip()) > 50:
            title += "..."
        if title:
            await session_store.update_session(session_id, title)


@router.post("/api/chat")
async def chat(request: ChatRequest):
    """Process a chat message and stream the response via SSE.

    Uses graph.astream_events() for real token-by-token streaming.
    Persists graph state (search/analysis results) across requests
    so follow-up commands like 'generate report' work correctly.
    """
    # ── 1. Validate that the session exists ──────────────────────────
    session = await session_store.get_session(request.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # ── 2. Persist the incoming user message ─────────────────────────
    # Save user message
    await session_store.add_message(request.session_id, "user", request.message)

    # ── 3. Auto-generate a title if this is the first message ────────
    # Auto-generate session title from first user message
    await _auto_title(request.session_id, request.message)

    # ── 4. Rebuild conversation history for the LLM ──────────────────
    # Build full conversation history (both user AND assistant messages)
    # so the LLM has context for multi-turn conversations.
    history = await session_store.get_messages(request.session_id)
    lang_messages = []
    for m in history:
        if m.role == "user":
            lang_messages.append(HumanMessage(content=m.content))
        elif m.role == "assistant":
            lang_messages.append(AIMessage(content=m.content))

    # ── 5. Load persisted graph context from previous requests ───────
    # This allows follow-up queries (e.g. "generate a report") to access
    # search/analysis data gathered in earlier turns of the conversation.
    ctx = await session_store.get_session_context(request.session_id)

    async def event_generator():
        """Async generator that yields SSE-formatted events to the client.

        The generator drives the LangGraph agent via ``astream_events``
        and filters for two kinds of events:

        * ``on_chat_model_stream`` — individual LLM tokens.  Only events
          tagged with ``"stream_to_user"`` are forwarded; internal chain-
          of-thought tokens from tool-calling sub-chains are suppressed.
        * ``on_chain_end`` — emitted when a graph node finishes.  The
          generator inspects the output dict to capture any updated graph
          state (search_results, analysis_results, report_data) for
          persistence after the stream ends.
        """
        graph = build_graph()

        # Seed the graph's input state with conversation history, any
        # previously persisted context, and bookkeeping fields.
        input_state = {
            "messages": lang_messages,
            "search_results": ctx.get("search_results", ""),
            "analysis_results": ctx.get("analysis_results", ""),
            "report_data": ctx.get("report_data", ""),
            "current_task": "",
            "pending_tasks": [],
            "session_id": request.session_id,
            "steps": 0,
        }

        # Accumulators — collect the full response text and any new graph
        # state produced during this invocation.
        full_content = ""
        new_search = ""
        new_analysis = ""
        new_report = ""

        def _sse_line(data: dict) -> str:
            """Format a single SSE event with proper double-newline terminator.

            The SSE protocol requires each event to end with two newline
            characters so the client knows where one event ends and the
            next begins.
            """
            return f"data: {json.dumps(data)}\n\n"

        try:
            # ── Stream tokens from the LangGraph agent ───────────────
            async for event in graph.astream_events(input_state, version="v2"):
                kind = event.get("event", "")

                # -- Token-level streaming from the chat model -----------
                # Only forward tokens that are explicitly tagged for user
                # display.  Tool-calling inner chains are excluded.
                if kind == "on_chat_model_stream":
                    tags = event.get("tags", [])
                    if "stream_to_user" not in tags:
                        continue
                    chunk = event.get("data", {}).get("chunk")
                    if chunk:
                        # -- Forward main content tokens -----------------
                        if hasattr(chunk, "content") and chunk.content:
                            # Scrub any surrogate characters that the model
                            # might emit to avoid encoding errors downstream.
                            token = chunk.content.encode("utf-8", errors="replace").decode("utf-8")
                            full_content += token
                            yield _sse_line({"content": token})

                        # -- Forward reasoning / thinking tokens ---------
                        # Reasoning-capable models (e.g. Ollama qwen3 with
                        # thinking, DeepSeek-R1) produce a separate stream
                        # of "thinking" tokens alongside the main content.
                        # The monkey-patch in provider.py (or the subclass
                        # ChatOpenAIWithReasoning) injects these tokens into
                        # chunk.additional_kwargs["reasoning"].
                        #
                        # We forward them to the frontend as a dedicated SSE
                        # field so the UI can render them differently (e.g.
                        # in a collapsible panel with distinct styling).
                        #
                        # If the model does not produce reasoning output,
                        # additional_kwargs will not have this key, and this
                        # block is a harmless no-op.
                        if hasattr(chunk, "additional_kwargs"):
                            reasoning_token = chunk.additional_kwargs.get("reasoning")
                            if reasoning_token:
                                reasoning_token = reasoning_token.encode(
                                    "utf-8", errors="replace"
                                ).decode("utf-8")
                                yield _sse_line({"reasoning": reasoning_token})

                # -- Capture updated graph state from chain outputs ------
                # When a graph node finishes, check if it produced new
                # search/analysis/report data that should be persisted.
                if kind == "on_chain_end":
                    output = event.get("data", {}).get("output")
                    if isinstance(output, dict):
                        if output.get("search_results"):
                            new_search = output["search_results"]
                        if output.get("analysis_results"):
                            new_analysis = output["analysis_results"]
                        if output.get("report_data"):
                            new_report = output["report_data"]

            # ── Fallback: non-streaming invocation ───────────────────
            # If the streaming path did not produce any user-visible
            # content (this can happen when the LLM responds entirely
            # via tool calls and the final answer node is not streamed),
            # fall back to a full synchronous invocation of the graph.
            if not full_content:
                result = await graph.ainvoke(input_state)
                messages = result.get("messages", [])
                # Walk backwards through messages to find the last AI reply.
                for msg in reversed(messages):
                    if isinstance(msg, AIMessage) and msg.content:
                        full_content = msg.content
                        break
                # Also capture any graph state from the non-streaming run.
                if result.get("search_results"):
                    new_search = result["search_results"]
                if result.get("analysis_results"):
                    new_analysis = result["analysis_results"]
                if result.get("report_data"):
                    new_report = result["report_data"]

                if full_content:
                    yield _sse_line({"content": full_content})

            # ── Last-resort fallback ─────────────────────────────────
            # If even the synchronous invocation produced nothing, send
            # a generic error message so the client always gets a reply.
            if not full_content:
                full_content = "I'm sorry, I couldn't process your request. Please try again."
                yield _sse_line({"content": full_content})

            # ── Persist the assistant reply ──────────────────────────
            # Encode-then-decode to replace any stray surrogates before
            # saving to the database.
            clean_content = full_content.encode("utf-8", errors="replace").decode("utf-8")
            await session_store.add_message(
                request.session_id, "assistant", clean_content
            )

            # ── Persist updated graph context ────────────────────────
            # This upsert ensures that the next user message in the same
            # session will have access to the latest search/analysis data.
            await session_store.save_session_context(
                request.session_id, new_search, new_analysis, new_report
            )

            # ── Signal stream completion to the client ───────────────
            yield _sse_line({"status": "complete"})

        except Exception as e:
            # Log the full traceback for debugging, but only send a terse
            # error message to the client to avoid leaking internals.
            logger.exception("Error during chat streaming")
            yield _sse_line({"error": f"Error: {e}"})

    # Return a StreamingResponse with SSE-appropriate headers.
    # * Cache-Control: no-cache  — prevent proxies from buffering events.
    # * Connection: keep-alive   — keep the TCP connection open for streaming.
    # * X-Accel-Buffering: no    — tell Nginx (if present) not to buffer.
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
