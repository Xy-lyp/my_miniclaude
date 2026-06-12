"""
Task DAG — data structures for autonomous task planning.

A TaskPlan is a directed acyclic graph of SubTasks.  The planner
produces the plan; PlanExecutor walks the DAG in topological order.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum


class TaskStatus(Enum):
    PENDING = "pending"
    READY = "ready"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class SubTask:
    id: str
    title: str
    description: str
    depends_on: list[str] = field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    result: str | None = None
    error: str | None = None
    tool_calls_count: int = 0

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "depends_on": self.depends_on,
            "status": self.status.value,
            "result": self.result,
            "error": self.error,
            "tool_calls_count": self.tool_calls_count,
        }

    @classmethod
    def from_dict(cls, d: dict) -> SubTask:
        return cls(
            id=d["id"],
            title=d["title"],
            description=d["description"],
            depends_on=d.get("depends_on", []),
            status=TaskStatus(d.get("status", "pending")),
            result=d.get("result"),
            error=d.get("error"),
            tool_calls_count=d.get("tool_calls_count", 0),
        )


@dataclass
class TaskPlan:
    goal: str
    subtasks: list[SubTask] = field(default_factory=list)
    execution_order: list[str] = field(default_factory=list)
    created_at: str = ""

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    def to_dict(self) -> dict:
        return {
            "goal": self.goal,
            "subtasks": [s.to_dict() for s in self.subtasks],
            "execution_order": self.execution_order,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> TaskPlan:
        plan = cls(
            goal=d["goal"],
            subtasks=[SubTask.from_dict(s) for s in d.get("subtasks", [])],
            execution_order=d.get("execution_order", []),
            created_at=d.get("created_at", ""),
        )
        return plan

    def validate(self) -> list[str]:
        """Validate the plan. Returns a list of error messages (empty = valid)."""
        errors: list[str] = []
        ids = {s.id for s in self.subtasks}

        if not self.subtasks:
            errors.append("Plan must have at least one subtask")

        for s in self.subtasks:
            for dep in s.depends_on:
                if dep not in ids:
                    errors.append(f"Task '{s.id}' depends on unknown task '{dep}'")

        # Check for cycles
        if not errors:
            cycle = self._find_cycle()
            if cycle:
                errors.append(f"Circular dependency detected: {' -> '.join(cycle)}")

        return errors

    def topo_sort(self) -> list[str]:
        """Topological sort. Returns ordered list of task IDs."""
        in_degree: dict[str, int] = {s.id: 0 for s in self.subtasks}
        adj: dict[str, list[str]] = {s.id: [] for s in self.subtasks}

        for s in self.subtasks:
            for dep in s.depends_on:
                adj[dep].append(s.id)
                in_degree[s.id] += 1

        queue = deque([tid for tid, deg in in_degree.items() if deg == 0])
        result: list[str] = []

        while queue:
            tid = queue.popleft()
            result.append(tid)
            for neighbor in adj.get(tid, []):
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        return result

    def _find_cycle(self) -> list[str] | None:
        """Detect cycles using DFS. Returns a cycle path or None."""
        ids = {s.id for s in self.subtasks}
        adj = {s.id: s.depends_on for s in self.subtasks}
        WHITE, GRAY, BLACK = 0, 1, 2
        color = {tid: WHITE for tid in ids}
        path: list[str] = []

        def dfs(node: str) -> bool:
            color[node] = GRAY
            path.append(node)
            for neighbor in adj.get(node, []):
                if neighbor not in ids:
                    continue
                if color[neighbor] == GRAY:
                    # Found cycle — extract it
                    idx = path.index(neighbor)
                    return True  # path[idx:] is the cycle
                if color[neighbor] == WHITE:
                    if dfs(neighbor):
                        return True
            path.pop()
            color[node] = BLACK
            return False

        for tid in ids:
            if color[tid] == WHITE:
                if dfs(tid):
                    # Return the cycle path
                    return path

        return None


@dataclass
class PlanResult:
    """Result of executing a full TaskPlan."""
    success: bool
    plan: TaskPlan
    completed: list[str]  # task IDs that completed
    failed: list[str]     # task IDs that failed
    skipped: list[str]    # task IDs that were skipped
    total_tool_calls: int
    final_summary: str
