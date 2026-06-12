"""SubAgentManager — tracks and manages all spawned sub-agents."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mini_core.llm.provider import LLMProvider
from mini_core.tools.registry import ToolRegistry
from mini_core.subagents.spawner import (
    SubAgentSpawner, SubAgentHandle, SubAgentConfig, SubAgentResult,
)


class SubAgentManager:
    def __init__(self, llm: LLMProvider, tools: ToolRegistry,
                 workdir: str = ".", event_bus=None) -> None:
        self._spawner = SubAgentSpawner(llm, tools, workdir=workdir, event_bus=event_bus)
        self._handles: dict[str, SubAgentHandle] = {}

    async def spawn(self, task: str, tool_names: list[str],
                     max_steps: int = 10, parent_run_id: str = "") -> SubAgentHandle:
        config = SubAgentConfig(task=task, tools=tool_names, max_steps=max_steps,
                                parent_run_id=parent_run_id)
        handle = await self._spawner.spawn(config)
        self._handles[handle.id] = handle
        return handle

    async def spawn_parallel(self, tasks: list[tuple[str, list[str]]]) -> list[SubAgentResult]:
        configs = [SubAgentConfig(task=t, tools=tools) for t, tools in tasks]
        return await self._spawner.spawn_parallel(configs)

    def get(self, agent_id: str) -> SubAgentHandle | None:
        return self._handles.get(agent_id)

    def list_all(self) -> list[dict]:
        return [{"id": h.id, "status": h.status} for h in self._handles.values()]

    def cancel(self, agent_id: str) -> bool:
        h = self._handles.get(agent_id)
        if h:
            h.cancel()
            return True
        return False
