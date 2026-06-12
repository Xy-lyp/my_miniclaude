"""
KamaTUI — Textual-based TUI for KamaClaude.
"""
from __future__ import annotations

import asyncio
import time

from textual.app import App, ComposeResult
from textual.widgets import Input, Static

from mini_tui.client import TuiIPCClient
from mini_tui.widgets.status_bar import StatusBarWidget
from mini_tui.widgets.step_progress import StepProgressWidget


class KamaTUI(App):
    CSS = """
    StatusBarWidget { height: 1; dock: top; }
    #main-area { height: 1fr; border: solid $primary; padding: 1; }
    #tool-area { height: 6; border: solid $secondary; padding: 1; }
    StepProgressWidget { height: 1; }
    #goal-input { dock: bottom; height: 3; }
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 9527) -> None:
        super().__init__()
        self._host = host
        self._port = port
        self._client = TuiIPCClient(host, port)
        self._status_bar = StatusBarWidget()
        self._step_progress = StepProgressWidget()
        self._start_time: float = 0.0
        self._current_run_id: str = ""
        self._mounted = False

    def compose(self) -> ComposeResult:
        yield self._status_bar
        yield Static("Ready. Enter a goal below and press Enter.", id="main-area")
        yield Static("", id="tool-area")
        yield self._step_progress
        yield Input(placeholder="Enter your goal and press Enter...", id="goal-input")

    async def on_mount(self) -> None:
        self._status_bar.set_connected(False)
        await self._client.connect()
        if self._client.connected:
            self._status_bar.set_connected(True)
            await self._client.subscribe_events(["*"])
        else:
            self.query_one("#main-area", Static).update(
                "[red]Cannot connect to daemon. Is mini-core running?[/red]")
        self._mounted = True
        asyncio.create_task(self._event_processor())

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        goal = event.value.strip()
        if not goal:
            return
        event.input.value = ""
        self._start_time = time.time()
        main = self.query_one("#main-area", Static)
        main.update(f"[bold]Goal:[/bold] {goal}\n")
        try:
            result = await self._client.call("agent.run", {"goal": goal, "workdir": "."})
            rd = result.get("result", {})
            self._current_run_id = rd.get("run_id", "")
            self._status_bar.set_run_id(self._current_run_id)
            main.update(rd.get("final_answer", main.renderable))
        except Exception as exc:
            main.update(f"[red]Error: {exc}[/red]")

    async def _event_processor(self) -> None:
        while not self._mounted:
            await asyncio.sleep(0.05)
        main = self.query_one("#main-area", Static)
        tools = self.query_one("#tool-area", Static)

        while self._client.connected:
            try:
                msg = await asyncio.wait_for(self._client.event_queue.get(), timeout=0.5)
            except asyncio.TimeoutError:
                if self._start_time > 0:
                    self._status_bar.set_elapsed(time.time() - self._start_time)
                continue

            data = msg.get("data", {})
            etype = msg.get("event_type", "")

            if etype == "run.started":
                self._current_run_id = data.get("run_id", "")
                self._start_time = time.time()
                self._status_bar.set_run_id(self._current_run_id)
                self._status_bar.set_steps(0, data.get("max_steps", 20))
                self._step_progress.set_max_steps(data.get("max_steps", 20))

            elif etype == "llm.request.start":
                self._step_progress.on_step_started(data.get("step_number", 0))

            elif etype == "llm.response":
                usage = data.get("usage", {})
                self._status_bar.set_token_usage(usage.get("total_tokens", 0))

            elif etype == "tool.call.start":
                tn = data.get("tool_name", "?")
                ta = data.get("tool_args", {})
                a = ", ".join(f"{k}={str(v)[:20]}" for k, v in ta.items())
                tools.update(f"▶ {tn}({a})  ...")

            elif etype == "tool.call.result":
                tn = data.get("tool_name", "?")
                ok = data.get("success", False)
                dur = data.get("duration_ms", 0)
                tools.update(f"  {'OK' if ok else 'FAIL'} {tn}  {dur:.0f}ms")

            elif etype == "step.completed":
                self._step_progress.on_step_completed(data.get("step_number", 0))

            elif etype == "run.completed":
                main.update(f"[green]{data.get('final_answer', '')}[/green]")
                self._status_bar.set_token_usage(
                    data.get("token_usage", {}).get("total_tokens", 0))

            elif etype == "run.error":
                main.update(f"[red]ERROR: {data.get('error_message', '?')}[/red]")

        self._status_bar.set_connected(False)

    async def on_unmount(self) -> None:
        await self._client.disconnect()


def main() -> None:
    import argparse
    p = argparse.ArgumentParser(description="KamaClaude TUI")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=9527)
    args = p.parse_args()
    KamaTUI(host=args.host, port=args.port).run()


if __name__ == "__main__":
    main()
