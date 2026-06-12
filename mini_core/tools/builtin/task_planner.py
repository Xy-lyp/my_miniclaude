"""
task_planner tool — decomposes a complex goal into ordered subtasks.

This tool uses the LLM via Planner to produce a TaskPlan, then returns
it as a formatted string the agent can use.
"""

from __future__ import annotations

from typing import Any

from mini_core.tools.base import Tool, ToolResult
from mini_core.agent.planner import Planner


class TaskPlannerTool(Tool):
    name = "task_planner"
    description = (
        "将复杂目标分解为有序子任务列表。在开始执行多步骤任务前调用此工具进行规划。"
        "返回一个结构化的任务计划，包含子任务列表和依赖关系。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "goal": {
                "type": "string",
                "description": "需要分解的原始目标",
            },
            "context": {
                "type": "string",
                "description": "当前工作目录、已有文件等相关上下文",
            },
        },
        "required": ["goal"],
    }

    def __init__(self, llm=None) -> None:
        self._llm = llm  # Will be set by factory

    def set_llm(self, llm) -> None:
        self._llm = llm

    async def execute(self, **kwargs: Any) -> ToolResult:
        goal = kwargs.get("goal", "")
        context = kwargs.get("context", "")

        if not self._llm:
            return ToolResult(
                success=False,
                content="Error: task_planner requires an LLM provider (not configured)",
                raw_content="",
                error="no llm configured",
            )

        planner = Planner(self._llm)
        try:
            plan = await planner.plan(goal=goal, context=context)
        except Exception as exc:
            return ToolResult(
                success=False,
                content=f"Planning failed: {exc}",
                raw_content=str(exc),
                error=str(exc),
            )

        # Format the plan as readable text
        lines = [f"任务计划：{plan.goal}", f"共 {len(plan.subtasks)} 个子任务：", ""]
        for tid in plan.execution_order:
            t = next(s for s in plan.subtasks if s.id == tid)
            deps = f" (依赖: {', '.join(t.depends_on)})" if t.depends_on else ""
            lines.append(f"  {t.id}: {t.title}{deps}")
            lines.append(f"       {t.description}")
        lines.append("")
        lines.append("请按顺序逐个执行。每个子任务完成后验证结果再继续。")

        formatted = "\n".join(lines)
        return ToolResult(
            success=True,
            content=formatted,
            raw_content=plan.to_dict(),
            metadata={
                "subtask_count": len(plan.subtasks),
                "execution_order": plan.execution_order,
                "plan_json": plan.to_dict(),
            },
        )
