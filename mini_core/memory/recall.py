"""
MemoryRecall — retrieve relevant notes before each Agent execution.

Strategies: keyword match, high-importance, recently used.
"""

from __future__ import annotations

from mini_core.memory.store import MemoryStore
from mini_core.memory.notes import NotesManager


class MemoryRecall:
    """Recall relevant notes from the current session before each run."""

    def __init__(self, store: MemoryStore) -> None:
        self._store = store
        self._notes = NotesManager(store)

    def recall(self, session_id: str, goal: str, top_k: int = 5) -> list[dict]:
        """Retrieve top-K relevant notes for the given goal.

        Strategies (in order):
        1. Keyword match against goal
        2. High-importance notes (importance >= 7)
        3. Recently accessed notes
        """
        if not session_id:
            return []

        all_notes = self._store.list_notes(session_id)
        if not all_notes:
            return []

        # Extract keywords from goal
        keywords = _extract_keywords(goal)

        scored: list[tuple[dict, float]] = []
        for note in all_notes:
            score = 0.0
            # Keyword matching
            note_text = f"{note['title']} {note['content']} {' '.join(note.get('tags', []))}"
            for kw in keywords:
                if kw.lower() in note_text.lower():
                    score += 3.0
            # Importance bonus
            score += note.get("importance", 5) * 0.5
            # Recency bonus
            score += min(note.get("access_count", 0), 10) * 0.3
            # Tag matching
            for tag in note.get("tags", []):
                if tag.lower() in goal.lower():
                    score += 2.0

            if score > 0:
                scored.append((note, score))

        # Sort by score descending, take top_k
        scored.sort(key=lambda x: -x[1])
        result = [n for n, _ in scored[:top_k]]

        # Update access counts
        for n in result:
            self._store.update_note(n["id"], {"access_count": n.get("access_count", 0) + 1})

        return result

    def format_for_prompt(self, notes: list[dict]) -> str:
        """Format recalled notes as a system prompt injection."""
        if not notes:
            return ""

        lines = ["\n## Relevant Context from Previous Sessions"]
        for n in notes:
            ntype = n.get("note_type", "")
            label = {"user_preference": "偏好", "project_context": "项目",
                     "learning": "经验", "decision": "决策"}.get(ntype, "记忆")
            lines.append(f"- [{label}] {n['title']}: {n['content']}")
        return "\n".join(lines)


def _extract_keywords(text: str) -> list[str]:
    """Simple keyword extraction — split and filter common words."""
    stopwords = {"的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都", "一",
                 "一个", "上", "也", "很", "到", "说", "要", "去", "你", "会", "着",
                 "the", "a", "an", "is", "in", "to", "of", "and", "for", "with",
                 "this", "that", "it", "on", "be", "as", "at", "or"}
    words = text.lower().replace(",", " ").replace(".", " ").replace("，", " ").split()
    keywords = [w for w in words if len(w) > 1 and w not in stopwords]
    return list(set(keywords))[:10]
