"""StatusBar — top status bar."""
from textual.widgets import Static


class StatusBarWidget(Static):
    def __init__(self) -> None:
        super().__init__("X Disconnected  |  Steps: 0/20  |  Tokens: 0/200K  |  Run: -  0.0s")
        self._connected = False
        self._elapsed = 0.0
        self._total_tokens = 0
        self._max_tokens = 200000
        self._run_id = "-"
        self._steps = 0
        self._max_steps = 20

    def set_connected(self, yes: bool) -> None: self._connected = yes; self._update_display()
    def set_elapsed(self, s: float) -> None: self._elapsed = s; self._update_display()
    def set_token_usage(self, total: int, max_tokens: int = 200000) -> None: self._total_tokens = total; self._max_tokens = max_tokens; self._update_display()
    def set_run_id(self, run_id: str) -> None: self._run_id = run_id; self._update_display()
    def set_steps(self, c: int, m: int) -> None: self._steps = c; self._max_steps = m; self._update_display()

    def _update_display(self) -> None:
        if not self.is_attached:
            return
        pct = min(100, (self._total_tokens / self._max_tokens) * 100) if self._max_tokens > 0 else 0
        bar = "#" * int(10 * pct / 100) + "." * (10 - int(10 * pct / 100))
        self.update(
            f"{'O' if self._connected else 'X'} {'Connected' if self._connected else 'Disconnected'}  |  "
            f"Steps: {self._steps}/{self._max_steps}  |  "
            f"Tokens: {self._total_tokens}/{self._max_tokens} [{bar}]  |  "
            f"Run: {self._run_id}  {self._elapsed:.1f}s"
        )
