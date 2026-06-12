"""StepProgress — visual step counter."""
from textual.widgets import Static


class StepProgressWidget(Static):
    def __init__(self, max_steps: int = 20) -> None:
        super().__init__("Steps: 0/20  [. . . . . . . . . . . . . . . . . . . .]")
        self._current_step = 0
        self._max_steps = max_steps
        self._step_statuses: dict[int, str] = {}

    def set_max_steps(self, n: int) -> None: self._max_steps = n; self._update_display()
    def on_step_started(self, step: int) -> None: self._current_step = step; self._step_statuses[step] = "running"; self._update_display()
    def on_step_completed(self, step: int, action_type: str = "") -> None: self._step_statuses[step] = "done"; self._update_display()
    def on_step_error(self, step: int) -> None: self._step_statuses[step] = "error"; self._update_display()

    def _update_display(self) -> None:
        if not self.is_attached:
            return
        parts = []
        for i in range(1, self._current_step + 1):
            s = self._step_statuses.get(i, "done")
            parts.append(">" if s == "running" else "X" if s == "error" else "-")
        for _ in range(self._current_step + 1, self._max_steps + 1):
            parts.append(".")
        self.update(f"Steps: {self._current_step}/{self._max_steps}  [{' '.join(parts)}]")
