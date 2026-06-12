"""
TraceReplayer — replays a saved trace, streaming events in chronological order.

Supports play/pause, speed control, and seeking.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncIterator

from mini_core.events.types import BaseEvent


@dataclass
class ReplayState:
    playing: bool = False
    speed: float = 1.0
    position: int = 0  # event index
    total_events: int = 0


class TraceReplayer:
    """Replay trace events with playback control."""

    def __init__(self, storage=None) -> None:
        self._storage = storage
        self._state = ReplayState()

    @property
    def state(self) -> ReplayState:
        return self._state

    def load(self, events: list[dict[str, Any]]) -> None:
        """Load pre-fetched events for replay."""
        self._events = events
        self._state.total_events = len(events)
        self._state.position = 0

    async def replay_stream(self, speed: float = 1.0) -> AsyncIterator[dict[str, Any]]:
        """Stream events one by one with simulated timing."""
        if not hasattr(self, "_events") or not self._events:
            return

        self._state.speed = speed
        self._state.playing = True
        self._state.position = 0

        prev_ts: float | None = None

        for i, event in enumerate(self._events):
            if not self._state.playing:
                break

            # Simulate timing based on original timestamps
            if prev_ts is not None and i > 0:
                try:
                    curr = _parse_ts(event.get("timestamp", ""))
                    delay = (curr - prev_ts) / speed if curr > prev_ts else 0
                    delay = min(delay, 2.0)  # cap at 2s
                    if delay > 0.01:
                        await asyncio.sleep(delay)
                except Exception:
                    await asyncio.sleep(0.1 / speed)

            prev_ts = _parse_ts(event.get("timestamp", ""))

            self._state.position = i + 1
            yield event

        self._state.playing = False

    def pause(self) -> None:
        self._state.playing = False

    def resume(self) -> None:
        self._state.playing = True

    def seek(self, index: int) -> None:
        """Jump to a specific event index."""
        if hasattr(self, "_events") and 0 <= index < len(self._events):
            self._state.position = index


def _parse_ts(ts: str) -> float:
    """Parse ISO timestamp to float seconds. Crude but functional."""
    try:
        from datetime import datetime
        dt = datetime.strptime(ts[:19], "%Y-%m-%dT%H:%M:%S")
        return dt.timestamp()
    except Exception:
        return time.time()
