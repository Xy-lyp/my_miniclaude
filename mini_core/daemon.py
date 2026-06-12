"""
mini-core daemon — TCP NDJSON + JSON-RPC 2.0 server.

Listens on a configurable host:port, accepts multiple concurrent TCP
connections, and dispatches JSON-RPC requests to registered handler methods.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import signal
import time
from datetime import datetime, timezone
from typing import Any, Callable, Awaitable

from mini_core.protocol import (
    JSONRPC_VERSION,
    PARSE_ERROR,
    INVALID_REQUEST,
    METHOD_NOT_FOUND,
    INTERNAL_ERROR,
    make_response,
    make_error_response,
)
from mini_core.transport import JsonRpcConnection
from mini_core.agent.runner import AgentRunner, RunResult
from mini_core.agent.event import read_events_file
from mini_core.agent.planner import Planner, PlanExecutor
from mini_core.agent.task_dag import TaskPlan, PlanResult
from mini_core.events.bus import EventBus
from mini_core.events.subscriber import IPCSubscriberManager
from mini_core.llm.provider import OpenAICompatibleProvider
from mini_core.tools.registry import ToolRegistry
from mini_core.tools.builtin.read_file import ReadFileTool
from mini_core.tools.builtin.write_file import WriteFileTool
from mini_core.tools.builtin.run_shell import RunShellTool
from mini_core.tools.builtin.task_planner import TaskPlannerTool
from mini_core.trace.collector import TraceCollector
from mini_core.trace.storage import TraceStorage
from mini_core.trace.replayer import TraceReplayer
from mini_core.memory.store import MemoryStore
from mini_core.memory.session import SessionManager
from mini_core.memory.thread import ThreadManager
from mini_core.memory.notes import NotesManager, MemoryExtractor
from mini_core.memory.recall import MemoryRecall

logger = logging.getLogger("mini-core")

# ── Method dispatcher ─────────────────────────────────────────────────────────

HandlerFunc = Callable[[dict[str, Any]], Awaitable[Any]]


class MethodDispatcher:
    """Registers and dispatches JSON-RPC methods by name."""

    def __init__(self) -> None:
        self._methods: dict[str, HandlerFunc] = {}

    def register(self, name: str, handler: HandlerFunc) -> None:
        """Register a method handler."""
        self._methods[name] = handler

    def has(self, name: str) -> bool:
        return name in self._methods

    async def dispatch(self, method: str, params: dict[str, Any]) -> Any:
        """Invoke a registered handler and return its result.

        Raises KeyError if the method is not registered.
        """
        handler = self._methods[method]
        return await handler(params)


# ── Daemon state ──────────────────────────────────────────────────────────────


class DaemonState:
    """Shared mutable state for the daemon."""

    def __init__(self) -> None:
        self._start_time = time.monotonic()
        self._connections: set[asyncio.Task[None]] = set()

    @property
    def uptime_seconds(self) -> float:
        return time.monotonic() - self._start_time

    @property
    def connection_count(self) -> int:
        return len(self._connections)

    def add_connection(self, task: asyncio.Task[None]) -> None:
        self._connections.add(task)

    def remove_connection(self, task: asyncio.Task[None]) -> None:
        self._connections.discard(task)


# ── Request handler ───────────────────────────────────────────────────────────


async def handle_jsonrpc_message(
    raw_message: dict[str, Any],
    connection: JsonRpcConnection,
    dispatcher: MethodDispatcher,
) -> None:
    """Parse and dispatch a single incoming JSON-RPC message.

    Only request objects are processed; responses from a client are ignored.
    Notifications (no id) are silently dropped as we do not need them yet.
    """
    # Validate basic structure
    if not isinstance(raw_message, dict):
        return

    if raw_message.get("jsonrpc") != JSONRPC_VERSION:
        # Not a JSON-RPC 2.0 message — silently ignore per spec
        return

    # Extract id (may be None for notifications)
    req_id = raw_message.get("id")

    # Notifications have no id — silently drop them
    if req_id is None:
        return

    # Validate method field exists
    method = raw_message.get("method")
    if not isinstance(method, str):
        err_resp = make_error_response(
            req_id, INVALID_REQUEST, "Invalid Request: missing or invalid 'method'"
        )
        await connection.send(err_resp)
        return

    params = raw_message.get("params", {})
    if not isinstance(params, dict):
        params = {}

    # Dispatch
    try:
        if not dispatcher.has(method):
            err_resp = make_error_response(
                req_id, METHOD_NOT_FOUND, f"Method not found: {method}"
            )
            await connection.send(err_resp)
            return

        result = await dispatcher.dispatch(method, params)
        resp = make_response(req_id, result)
        await connection.send(resp)

    except Exception as exc:
        logger.error("Unhandled error dispatching method '%s': %s", method, exc)
        err_resp = make_error_response(
            req_id, INTERNAL_ERROR, "Internal error", data={"detail": str(exc)}
        )
        try:
            await connection.send(err_resp)
        except Exception:
            pass


async def handle_connection(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    dispatcher: MethodDispatcher,
    state: DaemonState,
    conn_id: str = "",
    ipc_manager: IPCSubscriberManager | None = None,
) -> None:
    """Handle one TCP client connection for its entire lifetime.

    Reads NDJSON lines in a loop, dispatches each request, and sends back
    responses.  Runs as an independent asyncio Task.
    """
    conn = JsonRpcConnection(reader, writer)
    peer = conn.peer_info()
    logger.info("Client connected: %s (id=%s)", peer, conn_id)

    current_task = asyncio.current_task()
    if current_task is not None:
        state.add_connection(current_task)

    try:
        while True:
            try:
                raw = await conn.receive()
            except json.JSONDecodeError as exc:
                logger.error("ERROR: Parse error from %s: %s", peer, exc)
                err_resp = make_error_response(
                    None, PARSE_ERROR, "Parse error", data={"detail": str(exc)}
                )
                try:
                    await conn.send(err_resp)
                except Exception:
                    pass
                continue

            if raw is None:
                break

            # Intercept events.subscribe to register IPC subscriber
            if raw.get("method") == "events.subscribe" and ipc_manager and conn_id:
                params = raw.get("params", {})
                event_types = params.get("event_types", ["*"])
                await ipc_manager.add_subscription(conn_id, event_types, writer)
                resp = make_response(raw.get("id"), {"subscribed": True})
                await conn.send(resp)
                # Connection now enters "event mode" — but still accepts other RPC calls
                continue

            try:
                await handle_jsonrpc_message(raw, conn, dispatcher)
            except Exception as exc:
                logger.error("ERROR: Error handling message from %s: %s", peer, exc)

    except asyncio.CancelledError:
        pass
    except Exception as exc:
        logger.error("ERROR: Connection error with %s: %s", peer, exc)
    finally:
        await conn.close()
        if current_task is not None:
            state.remove_connection(current_task)
        if ipc_manager and conn_id:
            ipc_manager.remove_connection(conn_id)
        logger.info("Client disconnected: %s", peer)


# ── Server ────────────────────────────────────────────────────────────────────


async def run_daemon(host: str, port: int) -> None:
    """Start the TCP daemon and listen until cancelled."""
    dispatcher = MethodDispatcher()
    state = DaemonState()

    # ── Initialise EventBus + IPC subscriber manager ────────────────────────
    event_bus = EventBus()
    ipc_manager = IPCSubscriberManager(event_bus)
    await ipc_manager.start()

    # Background task to drain IPC writers periodically
    async def _drain_ipc():
        while True:
            await asyncio.sleep(0.1)
            await ipc_manager.drain_all()

    # ── Initialise Agent subsystem ──────────────────────────────────────────
    llm = OpenAICompatibleProvider()
    tools = ToolRegistry()
    default_workdir = "."
    _setup_builtin_tools(tools, workdir=default_workdir, llm=llm)

    agent_runner = AgentRunner(llm=llm, tools=tools, event_bus=event_bus)
    trace_collector = TraceCollector(event_bus)
    trace_storage = TraceStorage()
    trace_replayer = TraceReplayer(storage=trace_storage)

    # ── Initialise Memory subsystem ───────────────────────────────────────
    memory_store = MemoryStore()
    session_mgr = SessionManager(memory_store)
    thread_mgr = ThreadManager(memory_store)
    notes_mgr = NotesManager(memory_store)
    memory_recall = MemoryRecall(memory_store)

    # Pass memory context to agent runner
    agent_runner._memory = {
        "store": memory_store, "session_mgr": session_mgr,
        "thread_mgr": thread_mgr, "notes_mgr": notes_mgr,
        "recall": memory_recall, "extractor": MemoryExtractor(memory_store, llm),
    }

    # ── Register built-in methods ───────────────────────────────────────────
    dispatcher.register("ping", _handle_ping)
    dispatcher.register("status", lambda _: _handle_status(state))
    dispatcher.register(
        "agent.run",
        lambda params: _handle_agent_run(params, agent_runner, tools, trace_collector, trace_storage),
    )
    dispatcher.register(
        "agent.run_with_plan",
        lambda params: _handle_agent_run_with_plan(params, agent_runner, tools, llm, event_bus, trace_collector, trace_storage),
    )
    dispatcher.register(
        "agent.events",
        lambda params: _handle_agent_events(params, event_bus),
    )
    dispatcher.register("trace.list", lambda params: _handle_trace_list(params, trace_storage))
    dispatcher.register("trace.get", lambda params: _handle_trace_get(params, trace_storage))
    dispatcher.register("trace.replay", lambda params: _handle_trace_replay(params, trace_storage))
    # Session/Thread/Notes RPC
    dispatcher.register("session.create", lambda params: _handle_session_create(params, session_mgr))
    dispatcher.register("session.list", lambda params: _handle_session_list(params, session_mgr))
    dispatcher.register("session.get", lambda params: _handle_session_get(params, session_mgr))
    dispatcher.register("session.update", lambda params: _handle_session_update(params, session_mgr))
    dispatcher.register("session.delete", lambda params: _handle_session_delete(params, session_mgr))
    dispatcher.register("session.switch", lambda params: _handle_session_switch(params, session_mgr))
    dispatcher.register("thread.list", lambda params: _handle_thread_list(params, thread_mgr))
    dispatcher.register("thread.get", lambda params: _handle_thread_get(params, thread_mgr))
    dispatcher.register("thread.continue", lambda params: _handle_thread_continue(params, thread_mgr, agent_runner, tools, trace_collector, trace_storage))
    dispatcher.register("notes.list", lambda params: _handle_notes_list(params, notes_mgr))
    dispatcher.register("notes.create", lambda params: _handle_notes_create(params, notes_mgr))
    dispatcher.register("notes.update", lambda params: _handle_notes_update(params, notes_mgr))
    dispatcher.register("notes.delete", lambda params: _handle_notes_delete(params, notes_mgr))
    dispatcher.register("notes.search", lambda params: _handle_notes_search(params, notes_mgr))
    # events.subscribe is intercepted directly in handle_connection

    # We need to pass ipc_manager and conn_id tracking to handle_connection
    # We'll use a simple id counter stored in a mutable container
    conn_counter = [0]

    async def _on_connect(reader, writer):
        conn_counter[0] += 1
        conn_id = f"conn_{conn_counter[0]}"
        await handle_connection(reader, writer, dispatcher, state, conn_id, ipc_manager)

    server = await asyncio.start_server(_on_connect, host=host, port=port)

    addrs = ", ".join(str(s.getsockname()) for s in server.sockets)
    print(f"[mini-core] daemon started on {host}:{port}")
    logger.info("Listening on %s", addrs)

    drain_task = asyncio.create_task(_drain_ipc())

    try:
        async with server:
            await server.serve_forever()
    finally:
        drain_task.cancel()
        await ipc_manager.stop()
        event_bus.close_all()


# ── Built-in method handlers ──────────────────────────────────────────────────


async def _handle_ping(params: dict[str, Any]) -> dict[str, Any]:
    """Return a pong message with current UTC timestamp."""
    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return {"message": "pong", "timestamp": now_utc}


async def _handle_status(state: DaemonState) -> dict[str, Any]:
    """Return daemon uptime and current connection count."""
    return {
        "uptime_seconds": round(state.uptime_seconds, 3),
        "connections": state.connection_count,
    }


# ── Agent subsystem ────────────────────────────────────────────────────────────


def _setup_builtin_tools(tools: ToolRegistry, workdir: str, llm=None) -> None:
    """Register all built-in tools on the given registry."""
    tools.register(ReadFileTool(workdir=workdir))
    tools.register(WriteFileTool(workdir=workdir))
    tools.register(RunShellTool(workdir=workdir))
    tp = TaskPlannerTool()
    if llm:
        tp.set_llm(llm)
    tools.register(tp)
    tools.register(WriteFileTool(workdir=workdir))
    tools.register(RunShellTool(workdir=workdir))


async def _handle_agent_run(
    params: dict[str, Any],
    runner: AgentRunner,
    tools: ToolRegistry,
    trace_collector: TraceCollector | None = None,
    trace_storage: TraceStorage | None = None,
) -> dict[str, Any]:
    """Handle the agent.run JSON-RPC method."""
    goal = params.get("goal", "")
    workdir = params.get("workdir", ".")

    if not goal or not goal.strip():
        return {"success": False, "error": "Missing required parameter: goal"}

    if workdir != ".":
        for name in ("read_file", "write_file", "run_shell", "task_planner"):
            tool = tools.get(name)
            if tool is not None:
                tool._workdir = __import__("pathlib").Path(workdir).resolve()

    result: RunResult = await runner.run(goal=goal, workdir=workdir)

    # Record trace
    if trace_collector and trace_storage:
        trace_collector.start_trace(result.run_id)
        report = trace_collector.stop_trace(result.run_id)
        if report:
            trace_storage.save(report, goal=goal)

    return {
        "run_id": result.run_id,
        "final_answer": result.final_answer,
        "steps": result.steps,
        "token_usage": result.token_usage,
        "events_file": result.events_file,
        "success": result.success,
        "error": result.error,
    }


async def _handle_agent_run_with_plan(
    params: dict[str, Any],
    runner: AgentRunner,
    tools: ToolRegistry,
    llm,
    event_bus,
    trace_collector: TraceCollector | None = None,
    trace_storage: TraceStorage | None = None,
) -> dict[str, Any]:
    """Handle agent.run_with_plan: plan first, then execute in DAG order."""
    goal = params.get("goal", "")
    workdir = params.get("workdir", ".")
    auto_plan = params.get("auto_plan", True)

    if not goal or not goal.strip():
        return {"success": False, "error": "Missing required parameter: goal"}

    if workdir != ".":
        for name in ("read_file", "write_file", "run_shell", "task_planner"):
            tool = tools.get(name)
            if tool is not None:
                tool._workdir = __import__("pathlib").Path(workdir).resolve()

    run_id = __import__("uuid").uuid4().hex[:8]

    # Start trace
    if trace_collector:
        trace_collector.start_trace(run_id)

    # Phase 1: Plan
    planner = Planner(llm)
    plan = await planner.plan(goal=goal, context=f"Working directory: {workdir}")

    # Phase 2: Execute plan
    executor = PlanExecutor(llm=llm, tools=tools, event_bus=event_bus)
    plan_result: PlanResult = await executor.execute(plan)

    # Stop trace
    report = None
    if trace_collector:
        report = trace_collector.stop_trace(run_id)
        if report and trace_storage:
            trace_storage.save(report, goal=goal)

    return {
        "run_id": run_id,
        "success": plan_result.success,
        "final_answer": plan_result.final_summary,
        "plan": plan.to_dict(),
        "plan_result": {
            "completed": plan_result.completed,
            "failed": plan_result.failed,
            "skipped": plan_result.skipped,
            "total_tool_calls": plan_result.total_tool_calls,
        },
        "steps": plan_result.total_tool_calls,
        "token_usage": {},
        "events_file": f"events/{run_id}.jsonl",
    }


async def _handle_trace_list(params: dict[str, Any], storage: TraceStorage) -> dict[str, Any]:
    limit = params.get("limit", 50)
    summaries = storage.list_runs(limit=limit)
    return {"runs": [s.__dict__ for s in summaries]}


async def _handle_trace_get(params: dict[str, Any], storage: TraceStorage) -> dict[str, Any]:
    run_id = params.get("run_id", "")
    if not run_id:
        return {"error": "Missing run_id"}
    report = storage.query(run_id)
    if report is None:
        return {"error": f"No trace found for run: {run_id}"}
    return {"trace": report}


async def _handle_trace_replay(params: dict[str, Any], storage: TraceStorage) -> dict[str, Any]:
    run_id = params.get("run_id", "")
    if not run_id:
        return {"error": "Missing run_id"}
    report = storage.query(run_id)
    if report is None:
        return {"error": f"No trace found for run: {run_id}"}
    return {"events": report.get("ipc_messages", []), "event_count": report.get("event_count", 0)}


async def _handle_agent_events(params: dict[str, Any], bus: EventBus | None = None) -> dict[str, Any]:
    """Handle the agent.events JSON-RPC method."""
    run_id = params.get("run_id", "")
    if not run_id:
        return {"error": "Missing required parameter: run_id"}

    if bus:
        events = bus.read_events(run_id)
    else:
        from pathlib import Path
        events = read_events_file(Path("events") / f"{run_id}.jsonl")
    return {"events": events, "count": len(events)}


# ── Memory / Session / Thread / Notes handlers ────────────────────────────────


async def _handle_session_create(params: dict, sm) -> dict:
    s = sm.create(name=params.get("name", "unnamed"), workdir=params.get("workdir", "."),
                  system_prompt_override=params.get("system_prompt_override"),
                  model_override=params.get("model_override"))
    return {"session": s}


async def _handle_session_list(params: dict, sm) -> dict:
    return {"sessions": sm.list_all()}


async def _handle_session_get(params: dict, sm) -> dict:
    sid = params.get("session_id", "")
    s = sm.get(sid)
    return {"session": s} if s else {"error": f"Session not found: {sid}"}


async def _handle_session_update(params: dict, sm) -> dict:
    sid = params.pop("session_id", "")
    s = sm.update(sid, **params)
    return {"session": s} if s else {"error": f"Session not found: {sid}"}


async def _handle_session_delete(params: dict, sm) -> dict:
    ok = sm.delete(params.get("session_id", ""))
    return {"deleted": ok}


async def _handle_session_switch(params: dict, sm) -> dict:
    s = sm.switch(params.get("session_id", ""))
    return {"session": s} if s else {"error": "Session not found"}


async def _handle_thread_list(params: dict, tm) -> dict:
    sid = params.get("session_id", "")
    if not sid:
        return {"error": "Missing session_id"}
    return {"threads": tm.list_by_session(sid)}


async def _handle_thread_get(params: dict, tm) -> dict:
    t = tm.get(params.get("thread_id", ""))
    return {"thread": t} if t else {"error": "Thread not found"}


async def _handle_thread_continue(params: dict, tm, runner, tools, tc, ts) -> dict:
    tid = params.get("thread_id", "")
    goal = params.get("goal", "")
    thread = tm.get(tid)
    if not thread:
        return {"error": "Thread not found"}
    # Load historical messages then run with continuation
    result = await _handle_agent_run({"goal": goal, "workdir": "."}, runner, tools, tc, ts)
    return {"thread_id": tid, **result}


async def _handle_notes_list(params: dict, nm) -> dict:
    sid = params.get("session_id", "")
    ntype = params.get("note_type")
    return {"notes": nm.list_by_session(sid, note_type=ntype)}


async def _handle_notes_create(params: dict, nm) -> dict:
    n = nm.create(session_id=params.get("session_id", ""), title=params["title"],
                  content=params["content"], note_type=params.get("note_type", "project_context"),
                  source=params.get("source", "manual"), importance=params.get("importance", 5),
                  tags=params.get("tags"))
    return {"note": n}


async def _handle_notes_update(params: dict, nm) -> dict:
    nid = params.pop("note_id", "")
    n = nm.update(nid, **params)
    return {"note": n} if n else {"error": "Note not found"}


async def _handle_notes_delete(params: dict, nm) -> dict:
    return {"deleted": nm.delete(params.get("note_id", ""))}


async def _handle_notes_search(params: dict, nm) -> dict:
    sid = params.get("session_id", "")
    query = params.get("query", "")
    return {"notes": nm.search(sid, query)}


# ── CLI entry point ───────────────────────────────────────────────────────────


def main() -> None:
    """Entry point for `mini-core`."""
    parser = argparse.ArgumentParser(
        description="mini-core daemon — JSON-RPC 2.0 over TCP NDJSON"
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Listen address (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=9527,
        help="Listen port (default: 9527)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="[mini-core] %(levelname)s: %(message)s",
    )

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    main_task = loop.create_task(run_daemon(args.host, args.port))

    def shutdown(signame: str) -> None:
        print(f"[mini-core] daemon shutting down")
        logger.info("Received %s, shutting down...", signame)
        main_task.cancel()

    # Register signal handlers on the event loop
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, lambda s=sig: shutdown(s.name))
        except NotImplementedError:
            # Windows does not support add_signal_handler for SIGTERM in some versions
            signal.signal(sig, lambda signum, frame, s=sig: shutdown(s.name))

    try:
        loop.run_until_complete(main_task)
    except asyncio.CancelledError:
        pass
    finally:
        # Cancel any remaining tasks
        pending = asyncio.all_tasks(loop)
        for task in pending:
            task.cancel()
        # Gather with return_exceptions to suppress cancellation errors
        if pending:
            loop.run_until_complete(
                asyncio.gather(*pending, return_exceptions=True)
            )
        loop.close()


if __name__ == "__main__":
    main()
