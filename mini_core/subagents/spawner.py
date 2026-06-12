"""Spawner — creates and manages sub-agent execution."""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from typing import Any

from mini_core.agent.loop import AgentLoop, MaxStepsExceeded
from mini_core.agent.event import EventWriter
from mini_core.llm.provider import LLMProvider, TokenUsage
from mini_core.tools.registry import ToolRegistry
from mini_core.tools.base import Tool, ToolResult
from mini_core.subagents.context import build_subagent_context
from pathlib import Path


@dataclass
class SubAgentConfig:
    task: str
    tools: list[str] = field(default_factory=list)
    model: str | None = None
    max_steps: int = 10
    context_window: int = 100000
    parent_run_id: str = ""


@dataclass
class SubAgentResult:
    success: bool
    final_answer: str = ""
    steps: int = 0
    token_usage: dict[str, int] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    created_files: list[str] = field(default_factory=list)
    modified_files: list[str] = field(default_factory=list)


class SubAgentHandle:
    def __init__(self, agent_id: str, task: asyncio.Task) -> None:
        self.id = agent_id
        self._task = task
        self.status: str = "queued"
        self.result: SubAgentResult | None = None

    async def wait(self) -> SubAgentResult:
        self.status = "running"
        try:
            result = await self._task
            self.result = result
            self.status = "completed" if result.success else "failed"
            return result
        except asyncio.CancelledError:
            self.status = "cancelled"
            return SubAgentResult(success=False, errors=["Cancelled"])

    def cancel(self) -> None:
        self._task.cancel()


class SubAgentSpawner:
    def __init__(self, llm: LLMProvider, all_tools: ToolRegistry,
                 workdir: str = ".", event_bus=None) -> None:
        self._llm = llm
        self._all_tools = all_tools
        self._workdir = workdir
        self._event_bus = event_bus

    async def spawn(self, config: SubAgentConfig) -> SubAgentHandle:
        agent_id = uuid.uuid4().hex[:8]
        sub_tools = ToolRegistry()
        for tname in config.tools:
            tool = self._all_tools.get(tname)
            if tool:
                sub_tools.register(tool)

        context_msgs = build_subagent_context(
            [], config.task, working_directory=self._workdir,
        )

        async def _run():
            events_file = Path("events") / f"_sub_{agent_id}.jsonl"
            writer = EventWriter(events_file)
            writer.open()

            sub_loop = AgentLoop(
                llm=self._llm, tools=sub_tools, events=writer,
                run_id=f"sub_{agent_id}", max_steps=config.max_steps,
                event_bus=self._event_bus,
            )

            try:
                system_prompt = context_msgs[0]["content"]
                answer = await sub_loop.run(goal=config.task, system_prompt=system_prompt)
                return SubAgentResult(
                    success=True, final_answer=answer,
                    steps=sub_loop.step_number,
                    token_usage=sub_loop.total_usage,
                )
            except MaxStepsExceeded as exc:
                return SubAgentResult(success=False, errors=[str(exc)], steps=exc.steps_taken)
            except Exception as exc:
                return SubAgentResult(success=False, errors=[str(exc)])
            finally:
                writer.close()

        task = asyncio.create_task(_run())
        return SubAgentHandle(agent_id=agent_id, task=task)

    async def spawn_parallel(self, configs: list[SubAgentConfig]) -> list[SubAgentResult]:
        handles = [await self.spawn(c) for c in configs]
        results = await asyncio.gather(*[h.wait() for h in handles])
        return list(results)
