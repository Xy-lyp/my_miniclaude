"""
Notes manager + auto-extraction from completed threads.

Note types: user_preference, project_context, learning, decision
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from mini_core.memory.store import MemoryStore


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


NOTE_TYPE_LABELS = {
    "user_preference": "用户偏好",
    "project_context": "项目上下文",
    "learning": "经验教训",
    "decision": "决策记录",
}

EXTRACTION_PROMPTS = {
    "user_preference": (
        "从以下对话中提取用户偏好。用户偏好包括：代码风格、工具选择、命名规范等。"
        "如果对话中没有明显的用户偏好，返回空列表。"
        "返回 JSON: [{\"title\": \"<简短标题>\", \"content\": \"<具体描述>\", \"importance\": 6}]"
    ),
    "project_context": (
        "从以下对话中提取项目上下文信息。包括：技术栈、项目结构、配置文件位置等。"
        "返回 JSON: [{\"title\": \"<简短标题>\", \"content\": \"<具体描述>\", \"importance\": 5}]"
    ),
    "learning": (
        "从以下对话中提取经验教训。包括：遇到的错误、解决方案、最佳实践等。"
        "返回 JSON: [{\"title\": \"<简短标题>\", \"content\": \"<经验教训>\", \"importance\": 8}]"
    ),
    "decision": (
        "从以下对话中提取用户做出的重要决策。包括：架构选择、工具选型、策略决定等。"
        "返回 JSON: [{\"title\": \"<简短标题>\", \"content\": \"<决策及理由>\", \"importance\": 7}]"
    ),
}


class NotesManager:
    def __init__(self, store: MemoryStore) -> None:
        self._store = store

    def create(self, session_id: str, title: str, content: str,
               note_type: str = "project_context", source: str = "manual",
               importance: int = 5, tags: list[str] | None = None) -> dict:
        return self._store.create_note({
            "id": uuid.uuid4().hex[:12],
            "session_id": session_id,
            "title": title,
            "content": content,
            "source": source,
            "note_type": note_type,
            "importance": importance,
            "access_count": 0,
            "created_at": _now(),
            "updated_at": _now(),
            "tags": tags or [],
        })

    def get(self, note_id: str) -> dict | None:
        return self._store.get_note(note_id)

    def list_by_session(self, session_id: str, note_type: str | None = None) -> list[dict]:
        return self._store.list_notes(session_id, note_type=note_type)

    def update(self, note_id: str, **kwargs) -> dict | None:
        kwargs["updated_at"] = _now()
        return self._store.update_note(note_id, kwargs)

    def delete(self, note_id: str) -> bool:
        return self._store.delete_note(note_id)

    def search(self, session_id: str, query: str) -> list[dict]:
        return self._store.search_notes(session_id, query)

    def upsert(self, session_id: str, title: str, content: str,
               note_type: str, source: str, importance: int = 5,
               tags: list[str] | None = None) -> dict:
        """Create or update a note with the same title."""
        existing = self._store.find_duplicate_note(session_id, title)
        if existing:
            return self.update(existing["id"], content=content, importance=importance,
                               tags=tags, source=source)
        return self.create(session_id=session_id, title=title, content=content,
                          note_type=note_type, source=source, importance=importance,
                          tags=tags)


class MemoryExtractor:
    """Auto-extract notes from completed threads using LLM prompts."""

    def __init__(self, store: MemoryStore, llm=None) -> None:
        self._store = store
        self._llm = llm
        self._notes_mgr = NotesManager(store)

    async def extract_from_thread(self, thread: dict) -> list[dict]:
        """Extract notes of all 4 types from a completed thread."""
        if not self._llm:
            return []

        messages = thread.get("messages", [])
        if not messages:
            return []

        # Build a compact conversation text
        convo_text = ""
        for m in messages[-20:]:  # last 20 messages
            role = m.get("role", "?")
            content = str(m.get("content", ""))[:300]
            convo_text += f"[{role}]: {content}\n"

        extracted: list[dict] = []
        for note_type, prompt_template in EXTRACTION_PROMPTS.items():
            try:
                notes = await self._extract_type(convo_text, note_type, prompt_template)
                for n in notes:
                    created = self._notes_mgr.upsert(
                        session_id=thread["session_id"],
                        title=n["title"],
                        content=n["content"],
                        note_type=note_type,
                        source=thread.get("id", "unknown"),
                        importance=n.get("importance", 5),
                        tags=[note_type],
                    )
                    extracted.append(created)
            except Exception:
                continue  # Skip failed extractions

        return extracted

    async def _extract_type(self, convo_text: str, note_type: str, prompt: str) -> list[dict]:
        """Call LLM to extract notes of a specific type."""
        import json as _json
        full_prompt = f"{prompt}\n\n对话内容：\n{convo_text}\n\n只返回 JSON 数组，不要其他文字。"
        messages = [{"role": "user", "content": full_prompt}]

        try:
            response = await self._llm.chat(messages, tools=None)
            text = (response.content or "").strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1]
                if text.endswith("```"):
                    text = text[:-3]
            result = _json.loads(text)
            return result if isinstance(result, list) else []
        except Exception:
            return []
