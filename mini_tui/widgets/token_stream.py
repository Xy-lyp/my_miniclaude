"""TokenStream — real-time streaming LLM output display."""

from textual.widgets import Static
from textual.containers import VerticalScroll


class TokenStreamWidget(VerticalScroll):
    """Displays LLM token output in real time.

    Creates a new block on each llm.request.start event.
    Appends text from llm.token events.
    Colours: white (thinking), cyan (tool notes), green (final answer).
    """

    def __init__(self):
        super().__init__()
        self._current_content: list[str] = []
        self._current_mode: str = "thinking"  # "thinking" | "tool" | "final"

    def on_llm_request_start(self, step: int) -> None:
        """Start a new token block."""
        self._current_content = [f"╭─ Step {step}: Thinking " + "─" * 40]

    def on_llm_token(self, text: str) -> None:
        """Append a token."""
        self._current_content.append(text)

    def on_llm_response(self, content: str | None, finish_reason: str, tool_calls: list[str]) -> None:
        """Finalise the block after response."""
        if finish_reason == "tool_calls":
            self._current_mode = "tool"
            self._current_content.append(f"\n[cyan]→ Calling: {', '.join(tool_calls)}[/cyan]")
        elif finish_reason == "stop":
            self._current_mode = "final"
        self._current_content.append("╰" + "─" * 55)
        self._update_display()

    def on_tool_call_result(self, tool_name: str, success: bool, content_preview: str) -> None:
        """Show a brief tool result inline."""
        icon = "✓" if success else "✗"
        color = "green" if success else "red"
        short = content_preview[:100].replace("\n", " ")
        self._current_content.append(f"[{color}]  {icon} {tool_name}: {short}[/{color}]")

    def _render(self) -> None:
        """Update the widget with current content."""
        text = "\n".join(self._current_content[-50:])  # last 50 lines
        self.mount(Static(text))
        self.scroll_end(animate=False)
