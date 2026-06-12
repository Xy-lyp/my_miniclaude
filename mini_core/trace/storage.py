"""
TraceStorage — SQLite-backed persistent trace storage and query.
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mini_core.trace.collector import TraceReport


@dataclass
class TraceSummary:
    run_id: str
    goal: str = ""
    start_time: str = ""
    end_time: str = ""
    duration_ms: int = 0
    status: str = "unknown"
    total_tokens: int = 0
    event_count: int = 0
    tool_call_count: int = 0


class TraceStorage:
    """Persist and query trace reports via SQLite."""

    def __init__(self, db_path: str | Path = "traces.db") -> None:
        self._db_path = Path(db_path)
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(str(self._db_path)) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS traces (
                    run_id TEXT PRIMARY KEY,
                    goal TEXT,
                    start_time TEXT,
                    end_time TEXT,
                    duration_ms INTEGER,
                    status TEXT,
                    total_tokens INTEGER DEFAULT 0,
                    event_count INTEGER DEFAULT 0,
                    tool_call_count INTEGER DEFAULT 0,
                    report_json TEXT
                )
            """)
            conn.commit()

    def save(self, report: TraceReport, goal: str = "") -> None:
        """Persist a trace report."""
        tool_count = sum(t.count for t in report.tool_calls)
        status = "completed" if not report.errors else "error"
        with sqlite3.connect(str(self._db_path)) as conn:
            conn.execute(
                """INSERT OR REPLACE INTO traces
                   (run_id, goal, start_time, end_time, duration_ms, status,
                    total_tokens, event_count, tool_call_count, report_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    report.run_id, goal, report.start_time, report.end_time,
                    report.duration_ms, status,
                    report.total_tokens.get("total", 0), report.event_count,
                    tool_count, json.dumps(report.to_dict(), ensure_ascii=False),
                ),
            )
            conn.commit()

    def query(self, run_id: str) -> dict[str, Any] | None:
        """Retrieve a full trace report as a dict."""
        with sqlite3.connect(str(self._db_path)) as conn:
            row = conn.execute(
                "SELECT report_json FROM traces WHERE run_id = ?", (run_id,)
            ).fetchone()
        if row:
            return json.loads(row[0])
        return None

    def list_runs(self, limit: int = 50) -> list[TraceSummary]:
        """List recent runs."""
        with sqlite3.connect(str(self._db_path)) as conn:
            rows = conn.execute(
                """SELECT run_id, goal, start_time, end_time, duration_ms,
                          status, total_tokens, event_count, tool_call_count
                   FROM traces ORDER BY start_time DESC LIMIT ?""",
                (limit,),
            ).fetchall()
        return [
            TraceSummary(
                run_id=r[0], goal=r[1] or "", start_time=r[2] or "",
                end_time=r[3] or "", duration_ms=r[4] or 0, status=r[5] or "unknown",
                total_tokens=r[6] or 0, event_count=r[7] or 0, tool_call_count=r[8] or 0,
            )
            for r in rows
        ]

    def delete(self, run_id: str) -> None:
        with sqlite3.connect(str(self._db_path)) as conn:
            conn.execute("DELETE FROM traces WHERE run_id = ?", (run_id,))
            conn.commit()
