"""TaskTree — visual TaskPlan DAG display."""

from textual.widgets import Static
from textual.containers import VerticalScroll


class TaskTreeWidget(VerticalScroll):
    """Renders a task plan as a tree with status icons.

    Icons: ✓ completed | 🔄 in_progress | ⏳ pending | ✗ failed | ⏭ skipped
    """

    def __init__(self):
        super().__init__()
        self._tasks: list[dict] = []
        self._expanded: bool = True

    def set_plan(self, plan: dict | None) -> None:
        """Update the tree from a TaskPlan dict."""
        if plan is None:
            self._tasks = []
            self._update_display()
            return

        self._tasks = plan.get("subtasks", [])
        self._update_display()

    def update_task_status(self, task_id: str, status: str, tool_count: int = 0) -> None:
        """Update the status of a single task."""
        for t in self._tasks:
            if t.get("id") == task_id:
                t["status"] = status
                t["tool_calls_count"] = tool_count
                break
        self._update_display()

    def _update_display(self) -> None:
        """Render the task tree."""
        if not self._tasks:
            text = "[dim]No task plan yet[/dim]"
        else:
            lines = ["╭─ 任务计划 " + "─" * 45]
            for t in self._tasks:
                status = t.get("status", "pending")
                tid = t.get("id", "?")
                title = t.get("title", "")[:30]
                tc = t.get("tool_calls_count", 0)
                icon = {"completed": "✓", "in_progress": "🔄", "pending": "⏳",
                        "failed": "✗", "skipped": "⏭"}.get(status, "·")
                color = {"completed": "green", "in_progress": "yellow",
                         "failed": "red", "skipped": "dim"}.get(status, "white")
                deps = t.get("depends_on", [])
                dep_str = f" [{', '.join(deps)}]" if deps else ""
                tc_str = f" [{tc} calls]" if tc > 0 else ""
                lines.append(f"  [{color}]{icon} {tid}[/{color}]: {title}{dep_str}{tc_str}")
            lines.append("╰" + "─" * 55)
            text = "\n".join(lines)
        self._lines = lines  # store for tests
