"""
AgentEvent type definitions and NDJSON file writer.

Events are written immediately (not buffered) so that even if the
process crashes mid-run, partial event logs are preserved.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any


class AgentEventType(Enum):
    RUN_STARTED = "run_started"
    LLM_REQUEST_START = "llm_request_start"
    LLM_RESPONSE = "llm_response"
    TOOL_CALL_START = "tool_call_start"
    TOOL_CALL_RESULT = "tool_call_result"
    STEP_COMPLETED = "step_completed"
    RUN_COMPLETED = "run_completed"
    RUN_ERROR = "run_error"


@dataclass
class AgentEvent:
    type: AgentEventType
    timestamp: str  # ISO 8601
    run_id: str
    step_number: int
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type.value,
            "timestamp": self.timestamp,
            "run_id": self.run_id,
            "step_number": self.step_number,
            "data": self.data,
        }


def _now_iso() -> str:
    """Current UTC time as ISO 8601 string."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def make_event(
    event_type: AgentEventType,
    run_id: str,
    step_number: int,
    data: dict[str, Any] | None = None,
) -> AgentEvent:
    """Factory: create an AgentEvent with the current timestamp."""
    return AgentEvent(
        type=event_type,
        timestamp=_now_iso(),
        run_id=run_id,
        step_number=step_number,
        data=data or {},
    )


class EventWriter:
    """Writes AgentEvent objects to an NDJSON file, one per line.

    Flushes after every write so events are durable even on crash.
    """

    def __init__(self, file_path: Path) -> None:
        self._file_path = file_path
        self._file: Any = None

    def open(self) -> None:
        """Create (or overwrite) the events file."""
        self._file_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self._file_path, "a", encoding="utf-8")

    def write(self, event: AgentEvent) -> None:
        """Write one event as an NDJSON line and flush."""
        if self._file is None:
            raise RuntimeError("EventWriter not opened")
        line = json.dumps(event.to_dict(), ensure_ascii=False) + "\n"
        self._file.write(line)
        self._file.flush()

    def close(self) -> None:
        """Close the underlying file."""
        if self._file is not None:
            self._file.close()
            self._file = None

    @property
    def path(self) -> Path:
        return self._file_path


def read_events_file(file_path: Path) -> list[dict[str, Any]]:
    """Read an events.jsonl file and return all events as dicts."""
    if not file_path.exists():
        return []
    events: list[dict[str, Any]] = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events
