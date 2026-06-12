"""
mini CLI — Connects to mini-core daemon over TCP NDJSON + JSON-RPC 2.0.

Commands:
  mini                  Ping daemon and show status (default)
  mini run "goal"       Run an agent task
  mini events <run_id>  Show events for a completed run
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import textwrap

from mini_core.protocol import JSONRPC_VERSION
from mini_core.transport import JsonRpcConnection

logger = logging.getLogger("mini")


async def send_request(
    conn: JsonRpcConnection,
    request_id: int,
    method: str,
    params: dict | None = None,
    timeout: float = 10.0,
) -> dict:
    """Send a JSON-RPC request and wait for the matching response."""
    req = {
        "jsonrpc": JSONRPC_VERSION,
        "id": request_id,
        "method": method,
        "params": params or {},
    }

    await conn.send(req)

    deadline = asyncio.get_event_loop().time() + timeout

    while True:
        remaining = deadline - asyncio.get_event_loop().time()
        if remaining <= 0:
            raise asyncio.TimeoutError(
                f"Request '{method}' (id={request_id}) timed out after {timeout}s"
            )

        try:
            raw = await asyncio.wait_for(conn.receive(), timeout=remaining)
        except asyncio.TimeoutError:
            raise asyncio.TimeoutError(
                f"Request '{method}' (id={request_id}) timed out after {timeout}s"
            )

        if raw is None:
            raise ConnectionError("Connection closed by daemon")

        resp_id = raw.get("id")
        if resp_id == request_id:
            return raw


async def _connect(host: str, port: int, timeout: float) -> JsonRpcConnection:
    """Connect to the daemon, returning a JsonRpcConnection.

    Prints an error and exits with code 1 on failure.
    """
    try:
        reader, writer = await asyncio.wait_for(
            asyncio.open_connection(host, port),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        print(f"[mini] Failed to connect to daemon at {host}:{port}: timeout")
        sys.exit(1)
    except ConnectionRefusedError:
        print(f"[mini] Failed to connect to daemon at {host}:{port}: connection refused")
        sys.exit(1)
    except OSError as exc:
        print(f"[mini] Failed to connect to daemon at {host}:{port}: {exc}")
        sys.exit(1)

    return JsonRpcConnection(reader, writer)


# ── Default command (no subcommand): ping + status ────────────────────────────


async def cmd_ping_status(host: str, port: int, timeout: float) -> int:
    """Default behaviour: ping, then status."""
    conn = await _connect(host, port, timeout)

    try:
        # Ping
        ping_resp = await send_request(conn, 1, "ping", timeout=timeout)

        if "error" in ping_resp:
            err = ping_resp["error"]
            print(f"[mini] Ping error: [{err.get('code')}] {err.get('message')}")
            return 1

        result = ping_resp.get("result", {})
        message = result.get("message", "?")
        print(f"[mini] Connected. Daemon says: {message}")

        # Status
        status_resp = await send_request(conn, 2, "status", timeout=timeout)

        if "error" in status_resp:
            err = status_resp["error"]
            print(f"[mini] Status error: [{err.get('code')}] {err.get('message')}")
            return 1

        status = status_resp.get("result", {})
        uptime = status.get("uptime_seconds", "?")
        connections = status.get("connections", "?")
        uptime_int = int(float(uptime)) if isinstance(uptime, (int, float)) else uptime
        print(f"[mini] Daemon status: uptime={uptime_int}s, connections={connections}")

        return 0

    except (asyncio.TimeoutError, ConnectionError) as exc:
        print(f"[mini] ERROR: {exc}")
        return 1
    finally:
        await conn.close()


# ── "run" subcommand ──────────────────────────────────────────────────────────


async def cmd_run(host: str, port: int, timeout: float, goal: str, workdir: str, stream: bool = False) -> int:
    """Send agent.run to the daemon, optionally streaming events."""
    conn = await _connect(host, port, timeout)
    agent_timeout = 300.0

    if stream:
        return await _cmd_run_stream(conn, host, port, timeout, goal, workdir, agent_timeout)

    try:
        resp = await send_request(
            conn, 1, "agent.run",
            params={"goal": goal, "workdir": workdir},
            timeout=agent_timeout,
        )

        if "error" in resp:
            err = resp["error"]
            print(f"[mini] Agent error: [{err.get('code')}] {err.get('message')}")
            return 1

        result = resp.get("result", {})
        run_id = result.get("run_id", "?")
        success = result.get("success", False)
        steps = result.get("steps", 0)
        usage = result.get("token_usage", {})
        events_file = result.get("events_file", "")
        final_answer = result.get("final_answer", "")
        error = result.get("error")

        print(f"[mini] Run started: {run_id}")

        if not success:
            print(f"[mini] Run failed after {steps} steps: {error}")
            return 1

        print(f"[mini] Run completed: {steps} steps, {usage.get('total_tokens', 0)} tokens")
        print(f"[mini] Events saved: {events_file}")
        print()
        print(final_answer)

        return 0

    except asyncio.TimeoutError:
        print(f"[mini] ERROR: Agent run timed out after {agent_timeout}s")
        return 1
    except ConnectionError as exc:
        print(f"[mini] ERROR: {exc}")
        return 1
    finally:
        await conn.close()


async def _cmd_run_stream(
    conn, host: str, port: int, timeout: float, goal: str, workdir: str, agent_timeout: float
) -> int:
    """Stream events while running an agent task."""
    # Subscribe to events first
    sub_resp = await send_request(conn, 1, "events.subscribe", {"event_types": ["*"]}, timeout=timeout)
    if "error" in sub_resp:
        print(f"[mini] Subscribe failed: {sub_resp['error']}")
        return 1

    # Send agent.run request (different id so we can distinguish responses)
    run_req = {
        "jsonrpc": JSONRPC_VERSION,
        "id": 2,
        "method": "agent.run",
        "params": {"goal": goal, "workdir": workdir},
    }
    await conn.send(run_req)

    # Define ANSI colors
    C = {
        "run.started": "\033[1;36m",     # bold cyan
        "llm.request.start": "\033[2;37m",  # dim white
        "llm.response": "\033[33m",       # yellow
        "tool.call.start": "\033[34m",    # blue
        "tool.call.result": "\033[32m",   # green
        "step.completed": "\033[35m",     # magenta
        "run.completed": "\033[1;32m",    # bold green
        "run.error": "\033[1;31m",        # bold red
        "reset": "\033[0m",
    }

    deadline = asyncio.get_event_loop().time() + agent_timeout

    while True:
        remaining = deadline - asyncio.get_event_loop().time()
        if remaining <= 0:
            print("[mini] Timed out waiting for run completion")
            return 1

        try:
            raw = await asyncio.wait_for(conn.receive(), timeout=min(remaining, 5.0))
        except asyncio.TimeoutError:
            continue

        if raw is None:
            print("[mini] Connection closed")
            return 1

        msg_type = raw.get("type")
        if msg_type == "event":
            etype = raw.get("event_type", "")
            data = raw.get("data", {})
            color = C.get(etype, "")
            reset = C["reset"]

            if etype == "run.started":
                print(f"{color}[mini] Run started: {data.get('run_id', '?')}{reset}")
            elif etype == "run.completed":
                print(f"{color}[mini] ✓ Completed: {data.get('total_steps', 0)} steps, "
                      f"{data.get('token_usage', {}).get('total_tokens', 0)} tokens{reset}")
                print(data.get("final_answer", "")[:500])
                return 0
            elif etype == "run.error":
                print(f"{color}[mini] ✗ Error: {data.get('error_message', '?')}{reset}")
                return 1
            elif etype == "llm.request.start":
                step = data.get("step_number", 0)
                print(f"{color}  Step {step}: Thinking...{reset}")
            elif etype == "llm.response":
                finish = data.get("finish_reason", "?")
                tools = data.get("tool_calls", [])
                print(f"{color}  ← LLM: finish={finish}, tools={tools}{reset}")
            elif etype == "tool.call.start":
                print(f"{color}  ⚙ Calling: {data.get('tool_name', '?')}(...){reset}")
            elif etype == "tool.call.result":
                ok = "✓" if data.get("success") else "✗"
                print(f"{color}    {ok} {data.get('tool_name', '?')} ({data.get('duration_ms', 0):.0f}ms){reset}")
            elif etype == "step.completed":
                print(f"{color}  ← Step {data.get('step_number', 0)} done ({data.get('action_type', '?')}){reset}")

        elif "id" in raw and raw.get("id") == 2:
            # This is the agent.run response
            result = raw.get("result", {})
            if not result.get("success"):
                print(f"[mini] Run failed: {result.get('error')}")
                return 1
            # The run.completed event already has the final answer
            return 0

    return 0


# ── "events" subcommand ───────────────────────────────────────────────────────


async def cmd_events(host: str, port: int, timeout: float, run_id: str) -> int:
    """Query daemon for events of a specific run."""
    conn = await _connect(host, port, timeout)

    try:
        resp = await send_request(
            conn, 1, "agent.events",
            params={"run_id": run_id},
            timeout=timeout,
        )

        if "error" in resp:
            err = resp["error"]
            print(f"[mini] Error: [{err.get('code')}] {err.get('message')}")
            return 1

        result = resp.get("result", {})
        events = result.get("events", [])
        count = result.get("count", 0)

        if count == 0:
            print(f"[mini] No events found for run: {run_id}")
            return 1

        print(f"[mini] Events for run {run_id} ({count} events):")
        print("-" * 60)
        for evt in events:
            ts = evt.get("timestamp", "")
            etype = evt.get("type", "?")
            step = evt.get("step_number", 0)
            data = evt.get("data", {})

            # Emoji + summary per event type
            if etype == "run_started":
                goal_preview = data.get("goal", "")[:80]
                print(f"  {ts}  [step={step}]  {etype}")
                print(f"           goal: {goal_preview}")
            elif etype == "llm_request_start":
                print(f"  {ts}  [step={step}]  {etype}")
            elif etype == "llm_response":
                finish = data.get("finish_reason", "?")
                tc_names = data.get("tool_calls", [])
                print(f"  {ts}  [step={step}]  {etype}  finish={finish}  tools={tc_names}")
            elif etype == "tool_call_start":
                print(f"  {ts}  [step={step}]  {etype}  tool={data.get('tool_name', '?')}")
            elif etype == "tool_call_result":
                success_str = "OK" if data.get("success") else "FAIL"
                preview = data.get("content_preview", "")[:80]
                print(f"  {ts}  [step={step}]  {etype}  [{success_str}]  {preview}")
            elif etype == "step_completed":
                print(f"  {ts}  [step={step}]  {etype}  tools_executed={data.get('tools_executed', 0)}")
            elif etype == "run_completed":
                print(f"  {ts}  [step={step}]  {etype}  total_steps={data.get('total_steps', 0)}")
            elif etype == "run_error":
                print(f"  {ts}  [step={step}]  {etype}  error={data.get('error', '?')[:100]}")
            else:
                print(f"  {ts}  [step={step}]  {etype}")

        return 0

    except (asyncio.TimeoutError, ConnectionError) as exc:
        print(f"[mini] ERROR: {exc}")
        return 1
    finally:
        await conn.close()


# ── "trace" subcommand ───────────────────────────────────────────────────────


async def _cmd_trace_list(host: str, port: int, timeout: float, limit: int) -> int:
    """List recent traces."""
    conn = await _connect(host, port, timeout)
    try:
        resp = await send_request(conn, 1, "trace.list", {"limit": limit}, timeout=timeout)
        if "error" in resp:
            print(f"[mini] Error: {resp['error']}")
            return 1
        runs = resp.get("result", {}).get("runs", [])
        if not runs:
            print("[mini] No traces found.")
            return 0
        print(f"{'RUN_ID':<10} {'STATUS':<10} {'TOKENS':>8} {'EVENTS':>8} {'DURATION':>10} GOAL")
        print("-" * 80)
        for r in runs:
            rid = r.get("run_id", "?")[:8]
            status = r.get("status", "?")
            tokens = r.get("total_tokens", 0)
            events = r.get("event_count", 0)
            dur = f"{r.get('duration_ms', 0) / 1000:.1f}s"
            goal = (r.get("goal", "") or "")[:40]
            print(f"{rid:<10} {status:<10} {tokens:>8,} {events:>8} {dur:>10} {goal}")
        return 0
    finally:
        await conn.close()


async def _cmd_trace_get(host: str, port: int, timeout: float, run_id: str) -> int:
    """Get full trace report."""
    conn = await _connect(host, port, timeout)
    try:
        resp = await send_request(conn, 1, "trace.get", {"run_id": run_id}, timeout=timeout)
        if "error" in resp:
            print(f"[mini] Error: {resp['error']}")
            return 1
        trace = resp.get("result", {}).get("trace", {})
        import json
        print(json.dumps(trace, indent=2, ensure_ascii=False))
        return 0
    finally:
        await conn.close()


# ── "session" + "notes" command handlers ──────────────────────────────────────


async def _cmd_session(host: str, port: int, timeout: float, args) -> int:
    conn = await _connect(host, port, timeout)
    try:
        if args.sess_cmd == "create":
            resp = await send_request(conn, 1, "session.create", {"name": args.name, "workdir": args.workdir}, timeout=timeout)
            s = resp.get("result", {}).get("session", {})
            print(f"[mini] Session created: {s.get('id')} ({s.get('name')})")
        elif args.sess_cmd == "list":
            resp = await send_request(conn, 1, "session.list", {}, timeout=timeout)
            sessions = resp.get("result", {}).get("sessions", [])
            for s in sessions:
                print(f"  {s['id']}  {s['name']}  threads={s['thread_count']}  {s['updated_at'][:10]}")
        elif args.sess_cmd == "switch":
            resp = await send_request(conn, 1, "session.switch", {"session_id": args.session_id}, timeout=timeout)
            print(f"[mini] Switched to: {args.session_id}")
        elif args.sess_cmd == "delete":
            resp = await send_request(conn, 1, "session.delete", {"session_id": args.session_id}, timeout=timeout)
            print(f"[mini] Deleted: {resp.get('result', {}).get('deleted')}")
        return 0
    finally:
        await conn.close()


async def _cmd_notes(host: str, port: int, timeout: float, args) -> int:
    conn = await _connect(host, port, timeout)
    try:
        if args.notes_cmd == "list":
            resp = await send_request(conn, 1, "notes.list",
                                      {"session_id": args.session_id or "", "note_type": args.note_type}, timeout=timeout)
            notes = resp.get("result", {}).get("notes", [])
            for n in notes:
                print(f"  [{n['note_type']}] ⭐{n['importance']} {n['title']}: {n['content'][:60]}")
        elif args.notes_cmd == "create":
            resp = await send_request(conn, 1, "notes.create",
                                      {"session_id": args.session_id, "title": args.title,
                                       "content": args.content, "note_type": args.note_type}, timeout=timeout)
            n = resp.get("result", {}).get("note", {})
            print(f"[mini] Note created: {n.get('id')}")
        elif args.notes_cmd == "search":
            resp = await send_request(conn, 1, "notes.search",
                                      {"session_id": args.session_id, "query": args.query}, timeout=timeout)
            notes = resp.get("result", {}).get("notes", [])
            for n in notes:
                print(f"  [{n['note_type']}] ⭐{n['importance']} {n['title']}: {n['content'][:60]}")
        return 0
    finally:
        await conn.close()


# ── Entry point ───────────────────────────────────────────────────────────────


def main() -> None:
    """Entry point for `mini` CLI."""
    parser = argparse.ArgumentParser(
        description="mini CLI — connects to mini-core daemon"
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Daemon address (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=9527,
        help="Daemon port (default: 9527)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=10.0,
        help="Request timeout in seconds (default: 10)",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # mini run "goal"
    run_parser = subparsers.add_parser("run", help="Run an agent task")
    run_parser.add_argument("goal", help="The task goal / instruction for the agent")
    run_parser.add_argument(
        "--workdir",
        default=".",
        help="Working directory for the agent (default: .)",
    )
    run_parser.add_argument(
        "--stream",
        action="store_true",
        default=False,
        help="Stream events in real time",
    )

    # mini events <run_id>
    events_parser = subparsers.add_parser("events", help="Show events for a run")
    events_parser.add_argument("run_id", help="The run ID")

    # mini trace list
    trace_list_parser = subparsers.add_parser("trace", help="Trace commands")
    trace_sub = trace_list_parser.add_subparsers(dest="trace_cmd")
    trace_list = trace_sub.add_parser("list", help="List recent traces")
    trace_list.add_argument("--limit", type=int, default=20)
    trace_get = trace_sub.add_parser("get", help="Get full trace report")
    trace_get.add_argument("run_id", help="The run ID")

    # mini session ...
    session_parser = subparsers.add_parser("session", help="Session management")
    sess_sub = session_parser.add_subparsers(dest="sess_cmd")
    sess_create = sess_sub.add_parser("create", help="Create a new session")
    sess_create.add_argument("--name", required=True)
    sess_create.add_argument("--workdir", default=".")
    sess_list = sess_sub.add_parser("list", help="List all sessions")
    sess_switch = sess_sub.add_parser("switch", help="Switch to a session")
    sess_switch.add_argument("session_id")
    sess_delete = sess_sub.add_parser("delete", help="Delete a session")
    sess_delete.add_argument("session_id")

    # mini notes ...
    notes_parser = subparsers.add_parser("notes", help="Notes management")
    notes_sub = notes_parser.add_subparsers(dest="notes_cmd")
    notes_list = notes_sub.add_parser("list", help="List notes")
    notes_list.add_argument("--session", dest="session_id", default=None)
    notes_list.add_argument("--type", dest="note_type", default=None)
    notes_create = notes_sub.add_parser("create", help="Create a note")
    notes_create.add_argument("--session", dest="session_id", required=True)
    notes_create.add_argument("--type", dest="note_type", default="project_context")
    notes_create.add_argument("--title", required=True)
    notes_create.add_argument("--content", required=True)
    notes_search = notes_sub.add_parser("search", help="Search notes")
    notes_search.add_argument("query")
    notes_search.add_argument("--session", dest="session_id", required=True)

    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)

    try:
        if args.command == "run":
            exit_code = asyncio.run(
                cmd_run(args.host, args.port, args.timeout, args.goal, args.workdir, args.stream)
            )
        elif args.command == "events":
            exit_code = asyncio.run(
                cmd_events(args.host, args.port, args.timeout, args.run_id)
            )
        elif args.command == "trace":
            if args.trace_cmd == "list":
                exit_code = asyncio.run(
                    _cmd_trace_list(args.host, args.port, args.timeout, args.limit)
                )
            elif args.trace_cmd == "get":
                exit_code = asyncio.run(
                    _cmd_trace_get(args.host, args.port, args.timeout, args.run_id)
                )
            else:
                print("[mini] Usage: mini trace list|get <run_id>")
                exit_code = 1
        elif args.command == "session":
            exit_code = asyncio.run(
                _cmd_session(args.host, args.port, args.timeout, args)
            )
        elif args.command == "notes":
            exit_code = asyncio.run(
                _cmd_notes(args.host, args.port, args.timeout, args)
            )
        else:
            # Default: ping + status
            exit_code = asyncio.run(
                cmd_ping_status(args.host, args.port, args.timeout)
            )
    except KeyboardInterrupt:
        exit_code = 130

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
