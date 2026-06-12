"""ToolCall — collapsible tool call display block."""

from textual.widgets import Static
from textual.containers import Container


class ToolCallWidget(Container):
    """A collapsible block showing tool call details.

    States:
      - executing: yellow border
      - success: green border
      - failure: red border
    """

    def __init__(self):
        super().__init__()
        self._tool_name: str = ""
        self._args: dict = {}
        self._status: str = "executing"  # "executing" | "success" | "failure"
        self._result_content: str = ""
        self._duration_ms: float = 0.0
        self._expanded: bool = False

    def on_tool_call_start(self, tool_name: str, tool_args: dict) -> None:
        """Create the block in 'executing' state."""
        self._tool_name = tool_name
        self._args = tool_args
        self._status = "executing"
        self._expanded = False
        args_summary = ", ".join(f"{k}={str(v)[:30]}" for k, v in tool_args.items())
        self.mount(Static(f"  ▶ {tool_name}({args_summary})  [yellow]...[/yellow]"))
        self.styles.border = ("solid", "yellow")

    def on_tool_call_result(self, success: bool, content: str, duration_ms: float) -> None:
        """Update state after tool execution."""
        self._status = "success" if success else "failure"
        self._result_content = content[:500]
        self._duration_ms = duration_ms
        border_color = "green" if success else "red"
        self.styles.border = ("solid", border_color)

        # Update the initial line
        icon = "✓" if success else "✗"
        args_summary = ", ".join(f"{k}={str(v)[:20]}" for k, v in self._args.items())
        label = f"  {icon} {self._tool_name}({args_summary})  [{border_color}]{duration_ms:.0f}ms[/{border_color}]"
        self.mount(Static(label))

    def toggle_expand(self) -> None:
        """Toggle expanded/collapsed view."""
        self._expanded = not self._expanded
        if self._expanded and self._result_content:
            self.mount(Static(f"    {self._result_content[:300]}"))
