"""Session manager — CRUD for isolated project/user contexts."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from mini_core.memory.store import MemoryStore


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class SessionManager:
    def __init__(self, store: MemoryStore) -> None:
        self._store = store
        self._current_session_id: str | None = None

    @property
    def current_session_id(self) -> str | None:
        return self._current_session_id

    def switch(self, session_id: str) -> dict | None:
        s = self._store.get_session(session_id)
        if s:
            self._current_session_id = session_id
        return s

    def create(self, name: str, workdir: str = ".", **kwargs) -> dict:
        data = {
            "id": uuid.uuid4().hex[:12],
            "name": name,
            "workdir": workdir,
            "created_at": _now(),
            "updated_at": _now(),
            "metadata": kwargs.pop("metadata", {}),
            "thread_count": 0,
            "total_tokens_used": 0,
            "system_prompt_override": kwargs.pop("system_prompt_override", None),
            "model_override": kwargs.pop("model_override", None),
            "allowed_tools": kwargs.pop("allowed_tools", None),
        }
        return self._store.create_session(data)

    def get(self, session_id: str) -> dict | None:
        return self._store.get_session(session_id)

    def list_all(self) -> list[dict]:
        return self._store.list_sessions()

    def update(self, session_id: str, **kwargs) -> dict | None:
        updates = {k: v for k, v in kwargs.items() if v is not None}
        updates["updated_at"] = _now()
        return self._store.update_session(session_id, updates)

    def delete(self, session_id: str) -> bool:
        if self._current_session_id == session_id:
            self._current_session_id = None
        return self._store.delete_session(session_id)

    def find_or_create(self, name: str, workdir: str = ".") -> dict:
        """Find a session by name, or create one."""
        for s in self._store.list_sessions():
            if s["name"] == name:
                self._current_session_id = s["id"]
                return s
        return self.create(name=name, workdir=workdir)
