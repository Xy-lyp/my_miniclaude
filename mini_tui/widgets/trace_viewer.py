"""TraceViewer — live trace statistics panel."""

from textual.widgets import Static
from textual.containers import Container


class TraceViewerWidget(Container):
    """Shows run statistics: IPC count, event count, LLM calls, tool breakdown."""

    def __init__(self):
        super().__init__()
        self._run_id: str = "—"
        self._elapsed: float = 0.0
        self._ipc_count: int = 0
        self._event_count: int = 0
        self._llm_count: int = 0
        self._total_tokens: dict = {"prompt": 0, "completion": 0, "total": 0}
        self._max_tokens: int = 200000
        self._tool_counts: dict[str, int] = {}

    def set_run_id(self, run_id: str) -> None:
        self._run_id = run_id
        self._update_display()

    def set_elapsed(self, seconds: float) -> None:
        self._elapsed = seconds

    def update_stats(self, event_count: int = 0, ipc_count: int = 0,
                     llm_count: int = 0, tokens: dict | None = None,
                     tool_counts: dict | None = None) -> None:
        if event_count:
            self._event_count = event_count
        if ipc_count:
            self._ipc_count = ipc_count
        if llm_count:
            self._llm_count = llm_count
        if tokens:
            self._total_tokens = tokens
        if tool_counts:
            self._tool_counts = tool_counts
        self._update_display()

    def _update_display(self) -> None:
        pct = min(100, (self._total_tokens.get("total", 0) / self._max_tokens) * 100)
        bar = "█" * int(pct / 10) + "░" * (10 - int(pct / 10))
        color = "green" if pct < 70 else "yellow" if pct < 90 else "red"

        lines = [
            "╭─ Trace 信息 " + "─" * 16,
            f"│ Run: {self._run_id}",
            f"│ 耗时: {self._elapsed:.1f}s",
            f"│ IPC: {self._ipc_count}  事件: {self._event_count}  LLM: {self._llm_count}",
            f"│ Token: [{color}]{bar}[/{color}] {pct:.0f}%",
            f"│ 输入: {self._total_tokens.get('prompt', 0):,}  输出: {self._total_tokens.get('completion', 0):,}",
        ]
        if self._tool_counts:
            lines.append("│ 工具调用:")
            for name, count in sorted(self._tool_counts.items(), key=lambda x: -x[1])[:8]:
                lines.append(f"│   {name}: {count}")

        lines.append("╰" + "─" * 28)
        self._lines = lines  # store for tests
