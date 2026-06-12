"""
TraceCollector — side-channel three-layer data collection.

Layer 1: IPC messages (RPC requests/responses)
Layer 2: EventBus events (all emitted events)
Layer 3: LLM calls (full request/response pairs)

Designed to be non-blocking — collection should never slow down the agent.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from mini_core.events.bus import EventBus, Subscription
from mini_core.events.types import BaseEvent


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass
class IPCMessage:
    timestamp: str
    direction: str  # "in" (request) or "out" (response/event)
    method: str
    params: dict[str, Any] = field(default_factory=dict)
    result: dict[str, Any] | None = None
    error: str | None = None


@dataclass
class LLMCallRecord:
    timestamp: str
    model: str
    messages_json: str  # serialized
    response_json: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    duration_ms: float = 0.0


@dataclass
class ToolCallSummary:
    tool_name: str
    count: int = 0
    total_duration_ms: float = 0.0
    success_count: int = 0
    fail_count: int = 0


@dataclass
class TraceError:
    timestamp: str
    layer: str
    message: str
    detail: str = ""


@dataclass
class TraceReport:
    run_id: str
    start_time: str
    end_time: str
    duration_ms: int

    ipc_messages: list[IPCMessage] = field(default_factory=list)
    ipc_message_count: int = 0

    events: list[dict] = field(default_factory=list)
    event_count: int = 0
    event_type_breakdown: dict[str, int] = field(default_factory=dict)

    llm_calls: list[LLMCallRecord] = field(default_factory=list)
    total_tokens: dict[str, int] = field(default_factory=dict)

    tool_calls: list[ToolCallSummary] = field(default_factory=list)
    errors: list[TraceError] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": self.duration_ms,
            "ipc_message_count": self.ipc_message_count,
            "ipc_messages": [{
                "timestamp": m.timestamp, "direction": m.direction,
                "method": m.method, "params": m.params,
                "result": m.result, "error": m.error,
            } for m in self.ipc_messages],
            "event_count": self.event_count,
            "event_type_breakdown": self.event_type_breakdown,
            "llm_call_count": len(self.llm_calls),
            "llm_calls": [{
                "timestamp": c.timestamp, "model": c.model,
                "prompt_tokens": c.prompt_tokens, "completion_tokens": c.completion_tokens,
                "total_tokens": c.total_tokens, "duration_ms": c.duration_ms,
            } for c in self.llm_calls],
            "total_tokens": self.total_tokens,
            "tool_calls": [{"tool_name": t.tool_name, "count": t.count,
                            "success": t.success_count, "fail": t.fail_count,
                            "total_ms": t.total_duration_ms} for t in self.tool_calls],
            "error_count": len(self.errors),
        }


class TraceCollector:
    """Passive trace collector — hooks into EventBus and wraps LLM calls.

    Usage:
        collector = TraceCollector(event_bus)
        collector.start_trace("run-123")
        # ... agent runs ...
        report = collector.stop_trace("run-123")
    """

    def __init__(self, event_bus: EventBus | None = None) -> None:
        self._bus = event_bus
        self._active: dict[str, TraceReport] = {}
        self._subscriptions: dict[str, Subscription] = {}
        # Tool call counters
        self._tool_counters: dict[str, dict[str, ToolCallSummary]] = {}

    def start_trace(self, run_id: str) -> None:
        """Begin collecting trace data for a run."""
        now = _now_iso()
        report = TraceReport(
            run_id=run_id,
            start_time=now,
            end_time="",
            duration_ms=0,
            total_tokens={"prompt": 0, "completion": 0, "total": 0},
        )
        self._active[run_id] = report
        self._tool_counters[run_id] = {}

        # Subscribe to EventBus for Layer 2
        if self._bus:
            def _on_event(event: BaseEvent) -> None:
                if run_id in self._active:
                    r = self._active[run_id]
                    r.events.append(event.to_dict())
                    r.event_count += 1
                    r.event_type_breakdown[event.type] = r.event_type_breakdown.get(event.type, 0) + 1
                    # Track tool calls
                    if event.type == "tool.call.result":
                        tn = getattr(event, "tool_name", "?")
                        if tn not in self._tool_counters[run_id]:
                            self._tool_counters[run_id][tn] = ToolCallSummary(tool_name=tn)
                        ts = self._tool_counters[run_id][tn]
                        ts.count += 1
                        if getattr(event, "success", False):
                            ts.success_count += 1
                        else:
                            ts.fail_count += 1
                        ts.total_duration_ms += getattr(event, "duration_ms", 0)

            sub = self._bus.subscribe("*", _on_event)
            self._subscriptions[run_id] = sub

    def stop_trace(self, run_id: str) -> TraceReport:
        """Stop collecting and return the completed report."""
        report = self._active.pop(run_id, None)
        if report is None:
            return TraceReport(run_id=run_id, start_time="", end_time="", duration_ms=0)

        # Unsubscribe from EventBus
        sub = self._subscriptions.pop(run_id, None)
        if sub:
            sub.unsubscribe()

        # Finalize
        report.end_time = _now_iso()
        report.tool_calls = list(self._tool_counters.pop(run_id, {}).values())
        return report

    def get_trace(self, run_id: str) -> TraceReport | None:
        return self._active.get(run_id)

    def record_ipc_message(self, run_id: str, direction: str, method: str,
                           params: dict | None = None, result: dict | None = None,
                           error: str | None = None) -> None:
        """Record an IPC message (Layer 1). Call from transport layer."""
        if run_id not in self._active:
            return
        msg = IPCMessage(
            timestamp=_now_iso(), direction=direction, method=method,
            params=params or {}, result=result, error=error,
        )
        self._active[run_id].ipc_messages.append(msg)
        self._active[run_id].ipc_message_count += 1

    def record_llm_call(self, run_id: str, model: str, messages: list, response: dict,
                        usage: dict, duration_ms: float) -> None:
        """Record an LLM call (Layer 3). Call from LLM provider."""
        if run_id not in self._active:
            return
        rec = LLMCallRecord(
            timestamp=_now_iso(), model=model,
            messages_json=json.dumps(messages, ensure_ascii=False, default=str),
            response_json=json.dumps(response, ensure_ascii=False, default=str),
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            total_tokens=usage.get("total_tokens", 0),
            duration_ms=duration_ms,
        )
        report = self._active[run_id]
        report.llm_calls.append(rec)
        report.total_tokens["prompt"] = report.total_tokens.get("prompt", 0) + rec.prompt_tokens
        report.total_tokens["completion"] = report.total_tokens.get("completion", 0) + rec.completion_tokens
        report.total_tokens["total"] = report.total_tokens.get("total", 0) + rec.total_tokens

    def record_error(self, run_id: str, layer: str, message: str, detail: str = "") -> None:
        if run_id in self._active:
            self._active[run_id].errors.append(TraceError(
                timestamp=_now_iso(), layer=layer, message=message, detail=detail,
            ))
