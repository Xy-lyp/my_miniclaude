"""
S2 tests: EventBus + IPC subscriptions + TUI.

Tests the EventBus publish/subscribe system, JSONL persistence,
IPC event forwarding, and event type coverage.
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch

import pytest

from mini_core.events.types import (
    ALL_EVENT_TYPES,
    RunStartedEvent,
    RunCompletedEvent,
    RunErrorEvent,
    LLMRequestStartEvent,
    LLMTokenEvent,
    LLMResponseEvent,
    ToolCallStartEvent,
    ToolCallResultEvent,
    StepCompletedEvent,
)
from mini_core.events.bus import EventBus, Subscription
from mini_core.events.subscriber import IPCSubscriberManager
from mini_core.transport import JsonRpcConnection


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def event_bus(tmp_path):
    """Create an EventBus writing to a temp directory."""
    return EventBus(events_dir=str(tmp_path / "events"))


@pytest.fixture
def sample_run_id():
    return "test-run-001"


# ── Test 1: Subscribe and emit ────────────────────────────────────────────────


def test_eventbus_subscribe_and_emit(event_bus, sample_run_id):
    """Subscribing a callback and emitting an event invokes the callback."""
    received = []

    def handler(event):
        received.append(event)

    sub = event_bus.subscribe("run.started", handler)
    event = RunStartedEvent(run_id=sample_run_id, goal="test")
    event_bus.emit(event)

    assert len(received) == 1
    assert received[0].run_id == sample_run_id
    assert received[0].goal == "test"
    assert isinstance(sub, Subscription)


# ── Test 2: Multiple subscribers ──────────────────────────────────────────────


def test_eventbus_multiple_subscribers(event_bus, sample_run_id):
    """Multiple subscribers all receive the event."""
    received_a = []
    received_b = []

    event_bus.subscribe("run.started", lambda e: received_a.append(e))
    event_bus.subscribe("run.started", lambda e: received_b.append(e))

    event = RunStartedEvent(run_id=sample_run_id, goal="test")
    event_bus.emit(event)

    assert len(received_a) == 1
    assert len(received_b) == 1


# ── Test 3: Wildcard subscription ─────────────────────────────────────────────


def test_eventbus_wildcard_subscription(event_bus, sample_run_id):
    """'*' wildcard should receive ALL events regardless of type."""
    received = []

    event_bus.subscribe("*", lambda e: received.append(e))

    event_bus.emit(RunStartedEvent(run_id=sample_run_id, goal="g1"))
    event_bus.emit(LLMRequestStartEvent(run_id=sample_run_id, step_number=1, messages_count=2))
    event_bus.emit(ToolCallResultEvent(run_id=sample_run_id, tool_name="read", success=True, content_length=10, duration_ms=5.0))

    assert len(received) == 3
    assert received[0].type == "run.started"
    assert received[1].type == "llm.request.start"
    assert received[2].type == "tool.call.result"


# ── Test 4: Unsubscribe ───────────────────────────────────────────────────────


def test_eventbus_unsubscribe(event_bus, sample_run_id):
    """After unsubscribing, the callback no longer receives events."""
    received = []

    sub = event_bus.subscribe("run.started", lambda e: received.append(e))
    sub.unsubscribe()

    event_bus.emit(RunStartedEvent(run_id=sample_run_id, goal="test"))
    assert len(received) == 0


# ── Test 5: EventBus writes to JSONL ──────────────────────────────────────────


def test_eventbus_writes_to_jsonl(event_bus, sample_run_id, tmp_path):
    """Emitted events should be persisted to events/<run_id>.jsonl."""
    event_bus.emit(RunStartedEvent(run_id=sample_run_id, goal="test goal"))
    event_bus.emit(RunCompletedEvent(run_id=sample_run_id, final_answer="done", total_steps=3, token_usage={"total": 100}))
    event_bus.close_run(sample_run_id)

    file_path = tmp_path / "events" / f"{sample_run_id}.jsonl"
    assert file_path.exists()

    lines = file_path.read_text().strip().split("\n")
    assert len(lines) == 2

    e1 = json.loads(lines[0])
    assert e1["type"] == "run.started"
    assert e1["goal"] == "test goal"

    e2 = json.loads(lines[1])
    assert e2["type"] == "run.completed"
    assert e2["final_answer"] == "done"


# ── Test 6: IPC event subscribe ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_ipc_event_subscribe(event_bus, sample_run_id):
    """IPC subscriber should receive event pushes."""
    ipc_mgr = IPCSubscriberManager(event_bus)
    await ipc_mgr.start()

    # Create a mock writer — write() is sync, drain() is async, is_closing() is sync
    from unittest.mock import MagicMock
    mock_writer = MagicMock()
    mock_writer.is_closing.return_value = False
    mock_writer.drain = AsyncMock()
    mock_writer.write = MagicMock()

    # Subscribe mock client to all events
    await ipc_mgr.add_subscription("conn_1", ["*"], mock_writer)

    # Emit an event
    event_bus.emit(RunStartedEvent(run_id=sample_run_id, goal="ipc test"))
    event_bus.emit(ToolCallStartEvent(run_id=sample_run_id, step_number=1, tool_name="read", tool_args={"path": "x"}))

    # Give the async task time to process
    await asyncio.sleep(0.1)
    await ipc_mgr.drain_all()

    # Verify writer.write was called with NDJSON event messages
    assert mock_writer.write.call_count >= 2

    # Check the content of one call
    first_call = mock_writer.write.call_args_list[0][0][0]
    decoded = json.loads(first_call.decode("utf-8"))
    assert decoded["type"] == "event"
    assert decoded["event_type"] == "run.started"

    await ipc_mgr.stop()


# ── Test 7: Multiple IPC clients receive same events ──────────────────────────


@pytest.mark.asyncio
async def test_ipc_multiple_clients_same_events(event_bus, sample_run_id):
    """Two IPC subscribers should both receive the same events."""
    ipc_mgr = IPCSubscriberManager(event_bus)
    await ipc_mgr.start()

    from unittest.mock import MagicMock
    w1 = MagicMock()
    w1.is_closing.return_value = False
    w1.drain = AsyncMock()
    w1.write = MagicMock()
    w2 = MagicMock()
    w2.is_closing.return_value = False
    w2.drain = AsyncMock()
    w2.write = MagicMock()

    await ipc_mgr.add_subscription("c1", ["*"], w1)
    await ipc_mgr.add_subscription("c2", ["tool.call.start"], w2)

    event_bus.emit(ToolCallStartEvent(run_id=sample_run_id, step_number=1, tool_name="write", tool_args={}))
    event_bus.emit(StepCompletedEvent(run_id=sample_run_id, step_number=1, action_type="tool"))

    await asyncio.sleep(0.1)
    await ipc_mgr.drain_all()

    # w1 (wildcard) gets both events
    assert w1.write.call_count == 2
    # w2 (tool.call.start only) gets only the first
    assert w2.write.call_count == 1

    await ipc_mgr.stop()


# ── Test 8: IPC client disconnect cleanup ─────────────────────────────────────


@pytest.mark.asyncio
async def test_ipc_client_disconnect_cleanup(event_bus, sample_run_id):
    """When a client disconnects, its subscriptions are cleaned up."""
    ipc_mgr = IPCSubscriberManager(event_bus)
    await ipc_mgr.start()

    from unittest.mock import MagicMock
    w = MagicMock()
    w.is_closing.return_value = False
    w.drain = AsyncMock()
    w.write = MagicMock()

    await ipc_mgr.add_subscription("conn_x", ["*"], w)
    assert len(ipc_mgr._subscriptions) == 1

    # Simulate disconnect
    ipc_mgr.remove_connection("conn_x")
    await asyncio.sleep(0.05)

    assert len(ipc_mgr._subscriptions) == 0
    assert len(ipc_mgr._connections) == 0

    await ipc_mgr.stop()


# ── Test 9: TUI app starts ────────────────────────────────────────────────────


def test_tui_app_starts():
    """Verify the TUI app can be imported and instantiated."""
    from mini_tui.app import KamaTUI
    app = KamaTUI(host="127.0.0.1", port=9999)
    assert app is not None
    assert app._host == "127.0.0.1"
    assert app._port == 9999


# ── Test 10: Event type coverage ──────────────────────────────────────────────


def test_event_type_coverage():
    """Verify all defined event types are in the ALL_EVENT_TYPES registry."""
    expected = [
        "run.started", "run.completed", "run.error",
        "llm.request.start", "llm.token", "llm.response",
        "tool.call.start", "tool.call.progress", "tool.call.result",
        "step.completed",
        "permission.request", "permission.response",
        "context.warning", "compact.start", "compact.completed",
    ]

    for etype in expected:
        assert etype in ALL_EVENT_TYPES, f"Missing event type: {etype}"
        assert ALL_EVENT_TYPES[etype].type == etype

    assert len(ALL_EVENT_TYPES) == len(expected)
