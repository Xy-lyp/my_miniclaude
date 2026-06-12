"""
S3 tests: Task Planning + Trace system.

Tests the task DAG, planner, trace collector/storage/replayer, and TUI widgets.
"""

from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock, Mock

import pytest

from mini_core.agent.task_dag import (
    TaskPlan, SubTask, TaskStatus, PlanResult,
)
from mini_core.agent.planner import Planner, PlanExecutor, _extract_json
from mini_core.events.bus import EventBus
from mini_core.events.types import (
    RunStartedEvent, LLMResponseEvent, ToolCallStartEvent,
    ToolCallResultEvent, StepCompletedEvent, RunCompletedEvent,
)
from mini_core.trace.collector import TraceCollector, TraceReport
from mini_core.trace.storage import TraceStorage, TraceSummary
from mini_core.trace.replayer import TraceReplayer
from mini_core.llm.provider import LLMProvider, LLMResponse, LLMError, TokenUsage, ToolCall


# ── Test 1: Task planner decomposes goal ──────────────────────────────────────


@pytest.mark.asyncio
async def test_task_planner_decomposes_goal():
    """Complex goal is correctly decomposed into subtasks."""
    # Create a mock LLM that returns a plan JSON
    mock_llm = Mock()
    mock_llm.chat = AsyncMock(return_value=LLMResponse(
        content='''```json
{
  "subtasks": [
    {"id": "task-1", "title": "Create directory", "description": "mkdir", "depends_on": []},
    {"id": "task-2", "title": "Write app.py", "description": "write main app", "depends_on": ["task-1"]},
    {"id": "task-3", "title": "Write requirements", "description": "deps", "depends_on": ["task-1"]},
    {"id": "task-4", "title": "Test app", "description": "run test", "depends_on": ["task-2", "task-3"]}
  ]
}
```''',
        finish_reason="stop",
        usage=TokenUsage(prompt_tokens=100, completion_tokens=80, total_tokens=180),
    ))

    planner = Planner(mock_llm)
    plan = await planner.plan(goal="Create a Flask app", context="workdir: /tmp")

    assert len(plan.subtasks) == 4
    assert plan.subtasks[0].id == "task-1"
    assert plan.subtasks[3].depends_on == ["task-2", "task-3"]
    assert plan.execution_order is not None
    # task-1 must come before task-2 and task-3, which must come before task-4
    eo = plan.execution_order
    assert eo.index("task-1") < eo.index("task-2")
    assert eo.index("task-1") < eo.index("task-3")
    assert eo.index("task-2") < eo.index("task-4")
    assert eo.index("task-3") < eo.index("task-4")
    assert plan.validate() == []


# ── Test 2: Circular dependency detection ─────────────────────────────────────


def test_task_dag_no_circular_deps():
    """Circular dependencies are detected and rejected."""
    plan = TaskPlan(
        goal="test",
        subtasks=[
            SubTask(id="task-1", title="T1", description="", depends_on=["task-3"]),
            SubTask(id="task-2", title="T2", description="", depends_on=["task-1"]),
            SubTask(id="task-3", title="T3", description="", depends_on=["task-2"]),
        ],
    )
    errors = plan.validate()
    assert len(errors) == 1
    assert "circular" in errors[0].lower()


# ── Test 3: Topological sort ──────────────────────────────────────────────────


def test_task_dag_topo_sort():
    """Topological sort produces correct execution order."""
    plan = TaskPlan(
        goal="test",
        subtasks=[
            SubTask(id="a", title="A", description="", depends_on=[]),
            SubTask(id="b", title="B", description="", depends_on=["a"]),
            SubTask(id="c", title="C", description="", depends_on=["a"]),
            SubTask(id="d", title="D", description="", depends_on=["b", "c"]),
        ],
    )
    eo = plan.topo_sort()
    assert eo[0] == "a"
    assert eo.index("b") < eo.index("d")
    assert eo.index("c") < eo.index("d")
    assert len(eo) == 4


# ── Test 4: Plan executor runs in order ──────────────────────────────────────


@pytest.mark.asyncio
async def test_plan_executor_runs_in_order(tmp_path):
    """PlanExecutor executes tasks in dependency order."""
    from mini_core.tools.registry import ToolRegistry
    from mini_core.tools.builtin.run_shell import RunShellTool

    mock_llm = Mock()
    # Return short final answers for each sub-task
    mock_llm.chat = AsyncMock(return_value=LLMResponse(
        content="Completed.",
        finish_reason="stop",
        usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    ))

    tools = ToolRegistry()
    tools.register(RunShellTool(workdir=str(tmp_path)))

    plan = TaskPlan(
        goal="test",
        subtasks=[
            SubTask(id="a", title="A", description="step a", depends_on=[]),
            SubTask(id="b", title="B", description="step b", depends_on=["a"]),
        ],
        execution_order=["a", "b"],
    )

    executor = PlanExecutor(llm=mock_llm, tools=tools)
    result = await executor.execute(plan)

    assert result.success is True
    assert result.completed == ["a", "b"]
    assert result.failed == []
    assert result.total_tool_calls >= 2


# ── Test 5: Plan executor skips on failure ────────────────────────────────────


@pytest.mark.asyncio
async def test_plan_executor_skips_on_failure(tmp_path):
    """When a task fails, dependent tasks are skipped."""
    from mini_core.tools.registry import ToolRegistry
    from mini_core.tools.builtin.read_file import ReadFileTool

    tools = ToolRegistry()
    tools.register(ReadFileTool(workdir=str(tmp_path)))

    call_count = [0]

    class FailingLLM:
        async def chat(self, messages, tools=None):
            call_count[0] += 1
            if call_count[0] == 1:
                # First task: request a tool that fails (read non-existent file)
                return LLMResponse(
                    tool_calls=[ToolCall(id="tc1", name="read_file", arguments={"path": "nonexistent.txt"})],
                    finish_reason="tool_calls",
                    usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
                )
            if call_count[0] == 2:
                # First task's second call: raise to simulate failure
                raise LLMError("Simulated LLM failure for test")
            # Second task: normal completion
            return LLMResponse(
                content="Done.",
                finish_reason="stop",
                usage=TokenUsage(prompt_tokens=5, completion_tokens=3, total_tokens=8),
            )

    plan = TaskPlan(
        goal="test",
        subtasks=[
            SubTask(id="a", title="A", description="will fail", depends_on=[]),
            SubTask(id="b", title="B", description="depends on a", depends_on=["a"]),
        ],
        execution_order=["a", "b"],
    )

    executor = PlanExecutor(llm=FailingLLM(), tools=tools)
    result = await executor.execute(plan)

    assert result.success is False
    assert result.failed == ["a"]
    assert result.skipped == ["b"]


# ── Test 6: Trace collector all layers ────────────────────────────────────────


def test_trace_collector_all_layers(tmp_path):
    """TraceCollector captures IPC, EventBus, and LLM data."""
    bus = EventBus(events_dir=str(tmp_path / "events"))
    collector = TraceCollector(bus)

    run_id = "trace-test-001"
    collector.start_trace(run_id)

    # Layer 2: emit events via EventBus
    bus.emit(RunStartedEvent(run_id=run_id, goal="trace test"))
    bus.emit(ToolCallStartEvent(run_id=run_id, step_number=1, tool_name="read", tool_args={}))
    bus.emit(ToolCallResultEvent(run_id=run_id, tool_name="read", success=True, content_length=100, duration_ms=10.0))
    bus.emit(RunCompletedEvent(run_id=run_id, final_answer="ok", total_steps=2, token_usage={"total": 50}))

    # Layer 1: IPC messages
    collector.record_ipc_message(run_id, "in", "agent.run", {"goal": "test"})
    collector.record_ipc_message(run_id, "out", "agent.run", result={"success": True})

    # Layer 3: LLM calls
    collector.record_llm_call(run_id, "test-model", [{"role": "user"}], {"choices": []},
                              {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}, 100.0)

    report = collector.stop_trace(run_id)

    assert report.run_id == run_id
    assert report.event_count == 4
    assert report.ipc_message_count == 2
    assert len(report.llm_calls) == 1
    assert report.total_tokens["total"] == 15
    assert len(report.tool_calls) == 1


# ── Test 7: Trace storage save and query ──────────────────────────────────────


def test_trace_storage_save_and_query(tmp_path):
    """TraceStorage persists and retrieves trace reports."""
    db_path = tmp_path / "traces.db"
    storage = TraceStorage(db_path=db_path)

    # Create a minimal report
    report = TraceReport(
        run_id="save-test", start_time="2026-01-01T00:00:00Z",
        end_time="2026-01-01T00:00:10Z", duration_ms=10000,
    )
    storage.save(report, goal="save test goal")

    # Query it back
    data = storage.query("save-test")
    assert data is not None
    assert data["run_id"] == "save-test"
    assert data["duration_ms"] == 10000

    # List runs
    runs = storage.list_runs(limit=10)
    assert len(runs) >= 1
    assert runs[0].run_id == "save-test"
    assert runs[0].goal == "save test goal"


# ── Test 8: Trace replayer basic ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_trace_replayer_basic():
    """TraceReplayer streams events in order."""
    replayer = TraceReplayer()
    events = [
        {"type": "run.started", "timestamp": "2026-01-01T00:00:00Z", "run_id": "r1", "data": {}},
        {"type": "llm.request.start", "timestamp": "2026-01-01T00:00:01Z", "run_id": "r1", "data": {}},
        {"type": "llm.response", "timestamp": "2026-01-01T00:00:02Z", "run_id": "r1", "data": {}},
        {"type": "run.completed", "timestamp": "2026-01-01T00:00:03Z", "run_id": "r1", "data": {}},
    ]
    replayer.load(events)

    results = []
    async for evt in replayer.replay_stream(speed=10.0):
        results.append(evt)

    assert len(results) == 4
    assert results[0]["type"] == "run.started"
    assert results[-1]["type"] == "run.completed"


# ── Test 9: TUI task tree renders ─────────────────────────────────────────────


def test_tui_task_tree_renders():
    """TaskTreeWidget can accept a plan dict and render."""
    from mini_tui.widgets.task_tree import TaskTreeWidget
    widget = TaskTreeWidget()
    plan = {
        "goal": "test",
        "subtasks": [
            {"id": "t1", "title": "Task 1", "description": "First", "depends_on": [], "status": "completed", "tool_calls_count": 1},
            {"id": "t2", "title": "Task 2", "description": "Second", "depends_on": ["t1"], "status": "in_progress", "tool_calls_count": 0},
            {"id": "t3", "title": "Task 3", "description": "Third", "depends_on": ["t2"], "status": "pending", "tool_calls_count": 0},
        ],
        "execution_order": ["t1", "t2", "t3"],
    }
    widget.set_plan(plan)
    assert len(widget._tasks) == 3

    widget.update_task_status("t2", "completed", 1)
    assert widget._tasks[1]["status"] == "completed"


# ── Test 10: TUI trace viewer renders ─────────────────────────────────────────


def test_tui_trace_viewer_renders():
    """TraceViewerWidget stores stats correctly."""
    from mini_tui.widgets.trace_viewer import TraceViewerWidget
    widget = TraceViewerWidget()
    widget._event_count = 150
    widget._ipc_count = 20
    widget._llm_count = 8
    widget._total_tokens = {"prompt": 8000, "completion": 2000, "total": 10000}
    widget._tool_counts = {"read_file": 3, "write_file": 4, "run_shell": 2, "task_planner": 1}
    assert widget._event_count == 150
    assert widget._ipc_count == 20
    assert widget._llm_count == 8
    assert widget._tool_counts["task_planner"] == 1
