"""CLI terminal client — interactive chat using Rich for beautified output.

This module implements a fully interactive terminal interface for the Repo
Insight agent.  It uses the ``rich`` library for coloured, markdown-aware
output and manages chat sessions via the same storage layer as the web API.

Key concepts:
  - **Session**: a persistent conversation thread stored in SQLite.
  - **Graph**: the LangGraph agent graph that orchestrates search, analysis,
    and report-generation steps.
  - **Streaming**: tokens are received incrementally from the LLM and
    accumulated into *segments* (one per agent step) that are printed
    permanently to the terminal as soon as each step completes.

Slash-commands (``/new``, ``/list``, ``/load``, ``/report``, ``/quit``)
provide quick session management without leaving the REPL.
"""

from __future__ import annotations

import asyncio

from langchain_core.messages import AIMessage, HumanMessage
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from repo_insight.agents.graph import build_graph
from repo_insight.storage.database import init_db
from repo_insight.storage.session_store import (
    add_message,
    create_session,
    delete_session,
    get_messages,
    get_session,
    get_session_context,
    list_sessions,
    save_session_context,
    update_session,
)

# A single Rich Console instance shared by all output in the CLI.
# Rich handles automatic terminal-width detection and colour support.
console = Console()


async def run_cli():
    """Main CLI event loop — handles user input and agent responses.

    The function performs the following high-level steps:
      1. Initialise the database (create tables if needed).
      2. Print a welcome banner listing available commands.
      3. Create a default chat session.
      4. Enter an infinite input loop that reads user text, dispatches
         slash-commands, or forwards the message to the LangGraph agent.
      5. Stream the agent's reply token-by-token, printing each completed
         step as rendered Markdown.
      6. Persist both the user message and the assistant reply in the DB.
    """
    # Ensure SQLite tables exist before any storage operation.
    await init_db()

    HELP_TEXT = (
        "[bold]Commands:[/bold]\n"
        "  /new                       Start a new session\n"
        "  /list                      List recent sessions\n"
        "  /load   <id>               Load a session by ID prefix\n"
        "  /delete <id> ...           Delete sessions by ID prefix\n"
        "  /report [html|markdown]    Generate a report\n"
        "  /help  or  /?              Show this help\n"
        "  /quit                      Exit"
    )

    # Display a welcome banner with a summary of available slash-commands.
    console.print(Panel.fit(
        "[bold green]Repo Insight[/bold green]\n"
        "AI-powered GitHub project discovery\n\n"
        "Commands: /new  /list  /load <id>  /delete <id> ...  /report <html|markdown>  /help  /quit",
        title="Welcome",
    ))

    # Start with a fresh session; it will be auto-titled after the first
    # user message (see the "Auto-title" section below).
    session = await create_session(title="CLI Session")
    console.print(f"[dim]Session: {session.id}[/dim]\n")

    # Build the LangGraph agent graph once — it is re-used for every turn
    # so that compiled graph structure is not recreated unnecessarily.
    graph = build_graph()

    # --------- Main input loop ---------
    # Runs indefinitely until the user types /quit or sends EOF (Ctrl-D).
    while True:
        # Read a line of input from the user. EOFError is raised on Ctrl-D
        # and KeyboardInterrupt on Ctrl-C; both exit the loop gracefully.
        try:
            user_input = console.input("[bold cyan]You:[/bold cyan] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Goodbye![/dim]")
            break

        # Ignore blank lines so the agent is not invoked with empty input.
        if not user_input:
            continue

        # --------- Slash-command handling ---------
        # Each command is handled inline and ends with ``continue`` so that
        # the message is NOT forwarded to the agent.

        # /quit — exit the REPL immediately.
        if user_input == "/quit":
            console.print("[dim]Goodbye![/dim]")
            break

        # /help or /? — display available commands.
        if user_input in ("/help", "/?"):
            console.print(HELP_TEXT)
            console.print()
            continue

        # /new — discard the current session and start a fresh one.
        if user_input == "/new":
            session = await create_session(title="CLI Session")
            console.print(f"[green]New session created: {session.id}[/green]\n")
            continue

        # /delete <id1> <id2> ... — delete one or more sessions by UUID prefix.
        if user_input.startswith("/delete "):
            ids = user_input[8:].split()
            if not ids:
                console.print("[red]Usage: /delete <id1> [id2] ...[/red]")
            else:
                for sid in ids:
                    target = await get_session(sid)
                    if target:
                        if target.id == session.id:
                            console.print(f"  [yellow]{sid[:8]}[/yellow] — skipped (current session)")
                            continue
                        ok = await delete_session(target.id)
                        if ok:
                            console.print(f"  [green]{sid[:8]}[/green] — {target.title} — deleted")
                        else:
                            console.print(f"  [red]{sid[:8]}[/red] — delete failed")
                    else:
                        console.print(f"  [red]{sid}[/red] — not found")
            console.print()
            continue

        # /list — show up to 10 most recent sessions in a formatted table.
        if user_input == "/list":
            sessions = await list_sessions()
            if not sessions:
                console.print("[dim]No sessions found.[/dim]")
            else:
                table = Table(show_header=True, header_style="bold")
                table.add_column("", width=1)
                table.add_column("ID", width=8, style="cyan")
                table.add_column("Title", min_width=20)
                table.add_column("Updated", width=19, style="dim")
                for s in sessions[:10]:
                    marker = "*" if s.id == session.id else ""
                    table.add_row(marker, s.id[:8], s.title, s.updated_at[:19])
                console.print(table)
            console.print()
            continue

        # /load <id> — restore a previously saved session by its UUID prefix.
        if user_input.startswith("/load "):
            sid = user_input[6:].strip()
            loaded = await get_session(sid)
            if loaded:
                session = loaded
                console.print(f"[green]Loaded session: {session.title}[/green]")
                console.print(f"[dim]Session ID: {session.id}[/dim]")
                # Show recent conversation history so the user has context
                history = await get_messages(session.id)
                if history:
                    console.print(f"[dim]({len(history)} messages)[/dim]")
                    for m in history[-6:]:
                        if m.role == "user":
                            console.print(f"[bold cyan]You:[/bold cyan] {m.content[:120]}")
                        else:
                            preview = m.content[:120].replace('\n', ' ')
                            console.print(f"[bold green]AI:[/bold green] {preview}..." if len(m.content) > 120 else f"[bold green]AI:[/bold green] {preview}")
                console.print()
            else:
                console.print("[red]Session not found. Use /list to see available sessions.[/red]")
            continue

        # /report [html|markdown] — ask the agent to produce a formatted
        # report summarizing all repositories discussed in the session.
        # The command is rewritten into a natural-language prompt that the
        # agent can understand, then execution falls through to the normal
        # agent-invocation path below.
        if user_input.startswith("/report"):
            parts = user_input.split()
            fmt = parts[1] if len(parts) > 1 else "markdown"
            user_input = f"Generate a {fmt} report of all the repositories we discussed"
            is_report_cmd = True
        else:
            is_report_cmd = False

        # --------- Auto-title from first message ---------
        # If the session still has its default placeholder title, replace it
        # with a truncated version of the first real user message so that
        # ``/list`` output is meaningful.
        if session.title == "CLI Session" or session.title == "New Chat":
            title = user_input[:50] + ("..." if len(user_input) > 50 else "")
            await update_session(session.id, title)

        # --------- Persist the user message ---------
        await add_message(session.id, "user", user_input)

        # --------- Build LangChain message history ---------
        # Retrieve every message in the current session from the database and
        # convert them into LangChain message objects so the LLM receives the
        # full conversation context.
        history = await get_messages(session.id)
        lang_messages = []
        for m in history:
            if m.role == "user":
                lang_messages.append(HumanMessage(content=m.content))
            elif m.role == "assistant":
                lang_messages.append(AIMessage(content=m.content))

        # Load any accumulated context (search results, analyses, reports)
        # from previous turns so the agent can reference them.
        ctx = await get_session_context(session.id)

        # --------- Segment-based streaming output ---------
        # The agent graph may execute multiple steps (e.g. search -> analyse
        # -> respond).  Each step that produces LLM tokens is collected into
        # a *segment*.  When a step ends (``on_chain_end`` event), the
        # accumulated segment is printed as rendered Markdown and then a new
        # segment begins.  This approach ensures that earlier output is never
        # overwritten by later steps.
        segments: list[str] = []       # Completed, printed segments
        current_segment = ""           # Tokens accumulated for the current step
        new_search = ""                # Search results produced during this turn
        new_analysis = ""              # Analysis results produced during this turn
        new_report = ""                # Report data produced during this turn

        try:
            # Assemble the input state dictionary expected by the LangGraph
            # agent.  It merges conversation messages with persisted context
            # and zero-initialized control fields.
            input_state = {
                "messages": lang_messages,
                "session_id": session.id,
                "search_results": ctx.get("search_results", ""),
                "analysis_results": ctx.get("analysis_results", ""),
                "report_data": ctx.get("report_data", ""),
                "current_task": "",
                "pending_tasks": [],
                "steps": 0,
            }

            # Show a Rich spinner while the agent is working.  The spinner
            # is stopped temporarily whenever a completed segment is printed,
            # then restarted for the next step.
            status = console.status("Thinking...")
            status.start()

            # --------- Event-driven streaming loop ---------
            # ``astream_events`` yields fine-grained events from every node
            # in the LangGraph.  We care about two event types:
            #   - ``on_chat_model_stream``: individual LLM tokens (for
            #     progressive display).
            #   - ``on_chain_end``: signals that a graph node finished, so
            #     we can flush the current segment and capture context updates.
            async for event in graph.astream_events(input_state, version="v2"):
                kind = event.get("event", "")

                # --- Token-level streaming ---
                # Only tokens tagged with "stream_to_user" should be shown;
                # internal chain-of-thought or tool-call tokens are skipped.
                if kind == "on_chat_model_stream":
                    tags = event.get("tags", [])
                    if "stream_to_user" not in tags:
                        continue
                    chunk = event.get("data", {}).get("chunk")
                    if chunk and hasattr(chunk, "content") and chunk.content:
                        # Encode then decode to replace any surrogate or
                        # invalid bytes, preventing UnicodeEncodeError when
                        # the terminal cannot handle certain characters.
                        token = chunk.content.encode("utf-8", errors="replace").decode("utf-8")
                        current_segment += token

                # --- Step completion ---
                if kind == "on_chain_end":
                    output = event.get("data", {}).get("output")
                    if isinstance(output, dict):
                        # Capture any context data the step produced so it
                        # can be persisted after the full turn completes.
                        if output.get("search_results"):
                            new_search = output["search_results"]
                        if output.get("analysis_results"):
                            new_analysis = output["analysis_results"]
                        if output.get("report_data"):
                            new_report = output["report_data"]

                        # ---- Segment printing ----
                        # When a step finishes and there are accumulated
                        # tokens, stop the spinner, render the segment as
                        # Markdown, print it permanently, then restart the
                        # spinner for the next step.
                        if current_segment:
                            status.stop()
                            console.print(Markdown(current_segment))
                            segments.append(current_segment)
                            current_segment = ""

                        # Update the spinner text to reflect which step the
                        # agent is about to execute next, giving the user
                        # visual feedback on progress.
                        task = output.get("current_task", "")
                        if task == "search":
                            status.update("Searching GitHub...")
                        elif task == "analyze":
                            status.update("Analyzing repositories...")
                        elif task == "report":
                            status.update("Generating report...")
                        else:
                            status.update("Thinking...")
                        status.start()

            # All events consumed — stop the spinner for good.
            status.stop()

            # Flush any trailing tokens that arrived after the last
            # ``on_chain_end`` event (edge case, but possible).
            if current_segment:
                console.print(Markdown(current_segment))
                segments.append(current_segment)

            # Join all segments with double-newlines to form the complete
            # assistant response for storage.
            full_response = "\n\n".join(segments)

            # --------- Non-streaming fallback ---------
            # If streaming produced no output (e.g. the graph or model does
            # not support streaming events), fall back to a single blocking
            # ``ainvoke`` call and display the result at once.
            if not full_response:
                with console.status("Thinking..."):
                    result = await graph.ainvoke(input_state)
                messages = result.get("messages", [])
                if messages:
                    last = messages[-1]
                    if hasattr(last, "content"):
                        full_response = last.content
                        console.print(Markdown(full_response))
                if result.get("search_results"):
                    new_search = result["search_results"]
                if result.get("analysis_results"):
                    new_analysis = result["analysis_results"]
                if result.get("report_data"):
                    new_report = result["report_data"]

            # --------- Report file path display ---------
            # When the user ran /report, the LLM-streamed text may not
            # reflect the actual filename on disk.  Extract the real path
            # from the report_generator tool result and display it.
            if is_report_cmd and new_report:
                import re
                m = re.search(r'\*\*File\*\*:\s*(.+)', new_report)
                if m:
                    console.print(f"\n[bold green]Report saved:[/bold green] {m.group(1).strip()}")

            # --------- Persist assistant reply ---------
            # Store the complete assistant response in the database, after
            # sanitizing it to valid UTF-8 to avoid encoding errors.
            if full_response:
                clean = full_response.encode("utf-8", errors="replace").decode("utf-8")
                await add_message(session.id, "assistant", clean)

            # Persist accumulated context (search results, analysis, report)
            # so they are available in subsequent turns or after session reload.
            await save_session_context(session.id, new_search, new_analysis, new_report)

        except Exception as e:
            # Gracefully stop the spinner if it is still running, then print
            # the error in red so the user can see what went wrong.
            try:
                status.stop()
            except Exception:
                pass
            console.print(f"\n[red]Error: {e}[/red]")

        # Print an empty line between turns for visual separation.
        console.print()


def main():
    """Entry point for the CLI.

    Uses ``asyncio.run()`` to bootstrap the async event loop and execute
    the ``run_cli()`` coroutine.  This is the function referenced by the
    console-script entrypoint in pyproject.toml.
    """
    asyncio.run(run_cli())


# Allow running this module directly with ``python -m repo_insight.cli``.
if __name__ == "__main__":
    main()
