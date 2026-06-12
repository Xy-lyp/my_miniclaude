"""Thread manager — individual task execution records with full message history."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from mini_core.memory.store import MemoryStore


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class ThreadManager:
    def __init__(self, store: MemoryStore) -> None:
        self._store = store

    def create(self, session_id: str, goal: str, title: str = "", **kwargs) -> dict:
        data = {
            "id": uuid.uuid4().hex[:12],
            "session_id": session_id,
            "title": title or goal[:60],
            "goal": goal,
            "status": "running",
            "messages": [],
            "step_count": 0,
            "tool_call_count": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "created_at": _now(),
            "completed_at": None,
            "run_id": kwargs.get("run_id"),
            "parent_thread_id": kwargs.get("parent_thread_id"),
        }
        return self._store.create_thread(data)

    def get(self, thread_id: str) -> dict | None:
        return self._store.get_thread(thread_id)

    def list_by_session(self, session_id: str) -> list[dict]:
        return self._store.list_threads(session_id)

    def update(self, thread_id: str, **kwargs) -> dict | None:
        return self._store.update_thread(thread_id, kwargs)

    def complete(self, thread_id: str, messages: list[dict],
                 steps: int, tool_calls: int, prompt_tokens: int,
                 completion_tokens: int, run_id: str = "") -> dict | None:
        return self._store.update_thread(thread_id, {
            "status": "completed",
            "messages": messages,
            "step_count": steps,
            "tool_call_count": tool_calls,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "completed_at": _now(),
            "run_id": run_id,
        })

    def fail(self, thread_id: str, messages: list[dict] | None = None) -> dict | None:
        updates: dict = {"status": "failed", "completed_at": _now()}
        if messages is not None:
            updates["messages"] = messages
        return self._store.update_thread(thread_id, updates)
