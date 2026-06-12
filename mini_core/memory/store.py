"""
MemoryStore — SQLite-backed persistence for Session/Thread/Notes.

Schema: sessions, threads, notes tables with cascading deletes.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any


class MemoryStore:
    """Unified SQLite store for the three-layer memory system."""

    def __init__(self, db_path: str | Path = "memory.db") -> None:
        self._db_path = Path(db_path)
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(str(self._db_path)) as conn:
            conn.execute("PRAGMA foreign_keys = ON")
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    workdir TEXT NOT NULL DEFAULT '.',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    metadata_json TEXT DEFAULT '{}',
                    thread_count INTEGER DEFAULT 0,
                    total_tokens_used INTEGER DEFAULT 0,
                    system_prompt_override TEXT,
                    model_override TEXT,
                    allowed_tools_json TEXT
                );
                CREATE TABLE IF NOT EXISTS threads (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                    title TEXT NOT NULL,
                    goal TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'running',
                    messages_json TEXT NOT NULL DEFAULT '[]',
                    step_count INTEGER DEFAULT 0,
                    tool_call_count INTEGER DEFAULT 0,
                    prompt_tokens INTEGER DEFAULT 0,
                    completion_tokens INTEGER DEFAULT 0,
                    created_at TEXT NOT NULL,
                    completed_at TEXT,
                    run_id TEXT,
                    parent_thread_id TEXT
                );
                CREATE TABLE IF NOT EXISTS notes (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                    title TEXT NOT NULL,
                    content TEXT NOT NULL,
                    source TEXT NOT NULL DEFAULT 'manual',
                    note_type TEXT NOT NULL DEFAULT 'project_context',
                    importance INTEGER DEFAULT 5,
                    embedding_json TEXT,
                    access_count INTEGER DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    tags_json TEXT DEFAULT '[]'
                );
                CREATE INDEX IF NOT EXISTS idx_threads_session ON threads(session_id);
                CREATE INDEX IF NOT EXISTS idx_notes_session ON notes(session_id);
                CREATE INDEX IF NOT EXISTS idx_notes_type ON notes(session_id, note_type);
            """)
            conn.commit()

    # ── Sessions ──────────────────────────────────────────────────────────

    def create_session(self, data: dict) -> dict:
        with sqlite3.connect(str(self._db_path)) as conn:
            conn.execute("PRAGMA foreign_keys = ON")
            conn.execute(
                """INSERT INTO sessions (id, name, workdir, created_at, updated_at,
                   metadata_json, thread_count, total_tokens_used,
                   system_prompt_override, model_override, allowed_tools_json)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                (data["id"], data["name"], data.get("workdir", "."),
                 data["created_at"], data["updated_at"],
                 json.dumps(data.get("metadata", {})),
                 data.get("thread_count", 0), data.get("total_tokens_used", 0),
                 data.get("system_prompt_override"), data.get("model_override"),
                 json.dumps(data.get("allowed_tools"))),
            )
            conn.commit()
        return data

    def get_session(self, session_id: str) -> dict | None:
        with sqlite3.connect(str(self._db_path)) as conn:
            row = conn.execute("SELECT * FROM sessions WHERE id=?", (session_id,)).fetchone()
        return self._row_to_session(row) if row else None

    def list_sessions(self) -> list[dict]:
        with sqlite3.connect(str(self._db_path)) as conn:
            rows = conn.execute("SELECT * FROM sessions ORDER BY updated_at DESC").fetchall()
        return [self._row_to_session(r) for r in rows if r]

    def update_session(self, session_id: str, updates: dict) -> dict | None:
        existing = self.get_session(session_id)
        if not existing:
            return None
        merged = {**existing, **updates}
        with sqlite3.connect(str(self._db_path)) as conn:
            conn.execute("PRAGMA foreign_keys = ON")
            conn.execute(
                """UPDATE sessions SET name=?, workdir=?, updated_at=?,
                   metadata_json=?, thread_count=?, total_tokens_used=?,
                   system_prompt_override=?, model_override=?, allowed_tools_json=?
                   WHERE id=?""",
                (merged["name"], merged["workdir"], merged["updated_at"],
                 json.dumps(merged.get("metadata", {})),
                 merged.get("thread_count", 0), merged.get("total_tokens_used", 0),
                 merged.get("system_prompt_override"), merged.get("model_override"),
                 json.dumps(merged.get("allowed_tools")), session_id),
            )
            conn.commit()
        return merged

    def delete_session(self, session_id: str) -> bool:
        with sqlite3.connect(str(self._db_path)) as conn:
            conn.execute("PRAGMA foreign_keys = ON")
            cur = conn.execute("DELETE FROM sessions WHERE id=?", (session_id,))
            conn.commit()
        return cur.rowcount > 0

    # ── Threads ───────────────────────────────────────────────────────────

    def create_thread(self, data: dict) -> dict:
        with sqlite3.connect(str(self._db_path)) as conn:
            conn.execute("PRAGMA foreign_keys = ON")
            conn.execute(
                """INSERT INTO threads (id, session_id, title, goal, status,
                   messages_json, step_count, tool_call_count,
                   prompt_tokens, completion_tokens, created_at, completed_at,
                   run_id, parent_thread_id)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (data["id"], data["session_id"], data["title"], data["goal"],
                 data.get("status", "running"), json.dumps(data.get("messages", [])),
                 data.get("step_count", 0), data.get("tool_call_count", 0),
                 data.get("prompt_tokens", 0), data.get("completion_tokens", 0),
                 data["created_at"], data.get("completed_at"),
                 data.get("run_id"), data.get("parent_thread_id")),
            )
            conn.execute(
                "UPDATE sessions SET thread_count = thread_count + 1, updated_at = ? WHERE id = ?",
                (data["created_at"], data["session_id"]),
            )
            conn.commit()
        return data

    def get_thread(self, thread_id: str) -> dict | None:
        with sqlite3.connect(str(self._db_path)) as conn:
            row = conn.execute("SELECT * FROM threads WHERE id=?", (thread_id,)).fetchone()
        return self._row_to_thread(row) if row else None

    def list_threads(self, session_id: str) -> list[dict]:
        with sqlite3.connect(str(self._db_path)) as conn:
            rows = conn.execute(
                "SELECT * FROM threads WHERE session_id=? ORDER BY created_at DESC", (session_id,)
            ).fetchall()
        return [self._row_to_thread(r) for r in rows if r]

    def update_thread(self, thread_id: str, updates: dict) -> dict | None:
        existing = self.get_thread(thread_id)
        if not existing:
            return None
        merged = {**existing, **updates}
        with sqlite3.connect(str(self._db_path)) as conn:
            conn.execute("PRAGMA foreign_keys = ON")
            conn.execute(
                """UPDATE threads SET title=?, status=?, messages_json=?,
                   step_count=?, tool_call_count=?, prompt_tokens=?,
                   completion_tokens=?, completed_at=?, run_id=?
                   WHERE id=?""",
                (merged["title"], merged["status"], json.dumps(merged.get("messages", [])),
                 merged.get("step_count", 0), merged.get("tool_call_count", 0),
                 merged.get("prompt_tokens", 0), merged.get("completion_tokens", 0),
                 merged.get("completed_at"), merged.get("run_id"), thread_id),
            )
            conn.commit()
        return merged

    # ── Notes ─────────────────────────────────────────────────────────────

    def create_note(self, data: dict) -> dict:
        with sqlite3.connect(str(self._db_path)) as conn:
            conn.execute("PRAGMA foreign_keys = ON")
            conn.execute(
                """INSERT INTO notes (id, session_id, title, content, source,
                   note_type, importance, embedding_json, access_count,
                   created_at, updated_at, tags_json)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                (data["id"], data["session_id"], data["title"], data["content"],
                 data.get("source", "manual"), data.get("note_type", "project_context"),
                 data.get("importance", 5), json.dumps(data.get("embedding")),
                 data.get("access_count", 0), data["created_at"], data["updated_at"],
                 json.dumps(data.get("tags", []))),
            )
            conn.commit()
        return data

    def get_note(self, note_id: str) -> dict | None:
        with sqlite3.connect(str(self._db_path)) as conn:
            row = conn.execute("SELECT * FROM notes WHERE id=?", (note_id,)).fetchone()
        return self._row_to_note(row) if row else None

    def list_notes(self, session_id: str, note_type: str | None = None) -> list[dict]:
        with sqlite3.connect(str(self._db_path)) as conn:
            if note_type:
                rows = conn.execute(
                    "SELECT * FROM notes WHERE session_id=? AND note_type=? ORDER BY importance DESC",
                    (session_id, note_type),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM notes WHERE session_id=? ORDER BY importance DESC",
                    (session_id,),
                ).fetchall()
        return [self._row_to_note(r) for r in rows if r]

    def update_note(self, note_id: str, updates: dict) -> dict | None:
        existing = self.get_note(note_id)
        if not existing:
            return None
        merged = {**existing, **updates}
        with sqlite3.connect(str(self._db_path)) as conn:
            conn.execute("PRAGMA foreign_keys = ON")
            conn.execute(
                """UPDATE notes SET title=?, content=?, source=?, note_type=?,
                   importance=?, embedding_json=?, access_count=?,
                   updated_at=?, tags_json=? WHERE id=?""",
                (merged["title"], merged["content"], merged.get("source", "manual"),
                 merged.get("note_type", "project_context"), merged.get("importance", 5),
                 json.dumps(merged.get("embedding")), merged.get("access_count", 0),
                 merged["updated_at"], json.dumps(merged.get("tags", [])), note_id),
            )
            conn.commit()
        return merged

    def delete_note(self, note_id: str) -> bool:
        with sqlite3.connect(str(self._db_path)) as conn:
            conn.execute("PRAGMA foreign_keys = ON")
            cur = conn.execute("DELETE FROM notes WHERE id=?", (note_id,))
            conn.commit()
        return cur.rowcount > 0

    def search_notes(self, session_id: str, query: str) -> list[dict]:
        """Keyword search across title, content, and tags.

        Splits multi-word queries into individual keywords for broader matching.
        """
        keywords = query.strip().split()
        if not keywords:
            return []

        with sqlite3.connect(str(self._db_path)) as conn:
            # Build OR chain for each keyword
            conditions = " OR ".join(["(title LIKE ? OR content LIKE ? OR tags_json LIKE ?)"] * len(keywords))
            params = []
            for kw in keywords:
                p = f"%{kw}%"
                params.extend([p, p, p])

            rows = conn.execute(
                f"""SELECT * FROM notes WHERE session_id=?
                   AND ({conditions})
                   ORDER BY importance DESC, access_count DESC""",
                (session_id, *params),
            ).fetchall()
        return [self._row_to_note(r) for r in rows if r]

    def find_duplicate_note(self, session_id: str, title: str) -> dict | None:
        """Find a note with a similar title (for dedup)."""
        with sqlite3.connect(str(self._db_path)) as conn:
            row = conn.execute(
                "SELECT * FROM notes WHERE session_id=? AND title=? LIMIT 1",
                (session_id, title),
            ).fetchone()
        return self._row_to_note(row) if row else None

    # ── Row parsers ───────────────────────────────────────────────────────

    def _row_to_session(self, row: tuple) -> dict:
        return {
            "id": row[0], "name": row[1], "workdir": row[2],
            "created_at": row[3], "updated_at": row[4],
            "metadata": json.loads(row[5]) if row[5] else {},
            "thread_count": row[6], "total_tokens_used": row[7],
            "system_prompt_override": row[8], "model_override": row[9],
            "allowed_tools": json.loads(row[10]) if row[10] else None,
        }

    def _row_to_thread(self, row: tuple) -> dict:
        return {
            "id": row[0], "session_id": row[1], "title": row[2], "goal": row[3],
            "status": row[4],
            "messages": json.loads(row[5]) if row[5] else [],
            "step_count": row[6], "tool_call_count": row[7],
            "prompt_tokens": row[8], "completion_tokens": row[9],
            "created_at": row[10], "completed_at": row[11],
            "run_id": row[12], "parent_thread_id": row[13],
        }

    def _row_to_note(self, row: tuple) -> dict:
        return {
            "id": row[0], "session_id": row[1], "title": row[2], "content": row[3],
            "source": row[4], "note_type": row[5], "importance": row[6],
            "embedding": json.loads(row[7]) if row[7] else None,
            "access_count": row[8], "created_at": row[9], "updated_at": row[10],
            "tags": json.loads(row[11]) if row[11] else [],
        }
