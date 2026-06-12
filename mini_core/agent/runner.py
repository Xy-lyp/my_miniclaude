"""
AgentRunner — orchestrates a complete agent run from goal to result.

Uses EventBus for event emission (persistence + in-process subscribers).
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from pathlib import Path

from mini_core.agent.event import EventWriter  # kept for backward compat
from mini_core.agent.loop import AgentLoop, MaxStepsExceeded
from mini_core.events.bus import EventBus
from mini_core.events.types import RunStartedEvent, RunCompletedEvent, RunErrorEvent
from mini_core.llm.provider import LLMProvider, LLMError
from mini_core.tools.registry import ToolRegistry


@dataclass
class RunResult:
    run_id: str
    final_answer: str
    steps: int
    token_usage: dict[str, int]
    events_file: str
    success: bool = True
    error: str | None = None


def _load_system_prompt() -> str:
    prompt_path = Path(__file__).parent / "system_prompt.md"
    if prompt_path.exists():
        return prompt_path.read_text("utf-8")
    return "You are an AI programming assistant."


class AgentRunner:
    """Orchestrates a single agent run using EventBus for event emission.

    Usage:
        bus = EventBus()
        runner = AgentRunner(llm, tools, event_bus=bus)
        result = await runner.run("Create a hello.py file")
    """

    def __init__(
        self,
        llm: LLMProvider,
        tools: ToolRegistry,
        max_steps: int = 20,
        event_bus: EventBus | None = None,
    ) -> None:
        self.llm = llm
        self.tools = tools
        self.max_steps = max_steps
        self._event_bus = event_bus

    async def run(self, goal: str, workdir: str = ".") -> RunResult:
        run_id = uuid.uuid4().hex[:8]
        workdir_path = Path(workdir).resolve()
        workdir_path.mkdir(parents=True, exist_ok=True)

        # Use provided bus or create a standalone one
        bus = self._event_bus or EventBus()
        events_file = str(bus.get_events_file_path(run_id))

        # Create legacy EventWriter first (opens in append mode)
        from mini_core.agent.event import EventWriter
        writer = EventWriter(Path(events_file))
        writer.open()

        try:
            # Emit run.started via EventBus (appends to the same file)
            bus.emit(RunStartedEvent(
                run_id=run_id,
                goal=goal,
                workdir=str(workdir_path),
                max_steps=self.max_steps,
            ))

            system_prompt = _load_system_prompt()

            loop = AgentLoop(
                llm=self.llm,
                tools=self.tools,
                events=writer,
                run_id=run_id,
                max_steps=self.max_steps,
                event_bus=bus,
            )

            final_answer = await loop.run(goal=goal, system_prompt=system_prompt)

            # Emit run.completed via EventBus
            bus.emit(RunCompletedEvent(
                run_id=run_id,
                final_answer=final_answer,
                total_steps=loop.step_number,
                token_usage=loop.total_usage,
            ))

            bus.close_run(run_id)
            writer.close()

            return RunResult(
                run_id=run_id,
                final_answer=final_answer,
                steps=loop.step_number,
                token_usage=loop.total_usage,
                events_file=events_file,
                success=True,
            )

        except MaxStepsExceeded as exc:
            bus.emit(RunErrorEvent(
                run_id=run_id,
                error_type="max_steps_exceeded",
                error_message=str(exc),
            ))
            bus.close_run(run_id)
            writer.close()
            return RunResult(run_id=run_id, final_answer="", steps=exc.steps_taken, token_usage={}, events_file=events_file, success=False, error=str(exc))

        except LLMError as exc:
            bus.emit(RunErrorEvent(run_id=run_id, error_type="llm_error", error_message=str(exc)))
            bus.close_run(run_id)
            writer.close()
            return RunResult(run_id=run_id, final_answer="", steps=0, token_usage={}, events_file=events_file, success=False, error=f"LLM error: {exc}")

        except Exception as exc:
            bus.emit(RunErrorEvent(run_id=run_id, error_type=type(exc).__name__, error_message=str(exc)))
            bus.close_run(run_id)
            writer.close()
            return RunResult(run_id=run_id, final_answer="", steps=0, token_usage={}, events_file=events_file, success=False, error=str(exc))
