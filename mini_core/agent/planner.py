"""
Planner — uses LLM to decompose complex goals into TaskPlans,
and PlanExecutor to execute them in DAG order.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from mini_core.agent.task_dag import (
    TaskPlan,
    SubTask,
    TaskStatus,
    PlanResult,
)
from mini_core.agent.loop import AgentLoop
from mini_core.agent.event import EventWriter
from mini_core.llm.provider import LLMProvider, LLMResponse, TokenUsage
from mini_core.tools.registry import ToolRegistry

logger = logging.getLogger("mini-core.agent.planner")

PLANNING_PROMPT = """你是一个任务规划专家。请将以下目标分解为有序的子任务列表。

目标：{goal}
上下文：{context}

要求：
1. 每个子任务必须具体、可执行（不是抽象描述）
2. 明确标注子任务之间的依赖关系
3. 子任务数量：3-7 个
4. 子任务标题 ≤ 30 字
5. 每个子任务描述清楚需要用什么工具、达成什么效果

输出严格的 JSON 格式（不要添加任何额外的文本）：
{{
  "subtasks": [
    {{
      "id": "task-1",
      "title": "创建项目目录结构",
      "description": "使用 run_shell 创建 src/ 和 tests/ 目录",
      "depends_on": []
    }}
  ]
}}"""

PLANNING_THRESHOLD = 3  # estimated steps ≥ threshold triggers planning


class Planner:
    """Uses LLM to decompose goals into TaskPlans."""

    def __init__(self, llm: LLMProvider) -> None:
        self._llm = llm

    async def plan(self, goal: str, context: str = "") -> TaskPlan:
        """Generate a TaskPlan for the given goal."""
        prompt = PLANNING_PROMPT.format(goal=goal, context=context or "无")
        messages = [{"role": "user", "content": prompt}]

        response = await self._llm.chat(messages, tools=None)
        text = response.content or ""

        # Extract JSON from response (may be wrapped in ```json blocks)
        try:
            plan_dict = _extract_json(text)
        except json.JSONDecodeError:
            # Best-effort fallback: create a single-task plan
            logger.warning("Failed to parse planner JSON output, using fallback plan")
            return TaskPlan(
                goal=goal,
                subtasks=[
                    SubTask(
                        id="task-1",
                        title="执行目标",
                        description=goal,
                        depends_on=[],
                    )
                ],
                execution_order=["task-1"],
            )

        subtasks = [
            SubTask(
                id=s["id"],
                title=s.get("title", s["id"]),
                description=s.get("description", ""),
                depends_on=s.get("depends_on", []),
            )
            for s in plan_dict.get("subtasks", [])
        ]

        if not subtasks:
            subtasks = [
                SubTask(id="task-1", title="执行目标", description=goal, depends_on=[])
            ]

        plan = TaskPlan(goal=goal, subtasks=subtasks)
        errors = plan.validate()
        if errors:
            logger.warning("Plan validation errors: %s", errors)
            # Still return the plan — execution will handle issues

        plan.execution_order = plan.topo_sort()
        return plan


class PlanExecutor:
    """Executes a TaskPlan by running each SubTask in DAG order.

    Each ready SubTask is executed via a mini AgentLoop (one step with
    tool access).  The AgentLoop reports to the parent, and status is
    propagated through the DAG.
    """

    def __init__(
        self,
        llm: LLMProvider,
        tools: ToolRegistry,
        event_bus=None,
    ) -> None:
        self._llm = llm
        self._tools = tools
        self._event_bus = event_bus

    async def execute(self, plan: TaskPlan) -> PlanResult:
        """Execute all subtasks in dependency order."""
        completed: list[str] = []
        failed: list[str] = []
        skipped: list[str] = []
        total_tool_calls = 0

        # Build index
        task_map: dict[str, SubTask] = {s.id: s for s in plan.subtasks}
        pending = set(plan.execution_order)

        while pending:
            # Find ready tasks
            ready = [
                tid for tid in pending
                if _all_deps_completed(task_map[tid].depends_on, completed)
            ]

            if not ready:
                # Deadlock: remaining tasks have unmet dependencies (failures)
                for tid in list(pending):
                    if any(d in failed for d in task_map[tid].depends_on):
                        task_map[tid].status = TaskStatus.SKIPPED
                        skipped.append(tid)
                        pending.discard(tid)
                break

            for tid in ready:
                task = task_map[tid]
                task.status = TaskStatus.IN_PROGRESS

                # Execute via a single-step AgentLoop
                sub_goal = f"【子任务 {task.id}】{task.title}\n\n详细描述：{task.description}"
                system_prompt = (
                    "你是任务执行专家。只执行当前子任务，完成后立即报告结果。"
                    "不要做子任务范围之外的事情。"
                )

                from mini_core.agent import loop as loop_mod
                from pathlib import Path
                import uuid
                import tempfile

                # Use a temporary EventWriter for the sub-loop
                events_file = Path("events") / f"_sub_{task.id}_{uuid.uuid4().hex[:6]}.jsonl"
                writer = EventWriter(events_file)
                writer.open()

                sub_loop = AgentLoop(
                    llm=self._llm,
                    tools=self._tools,
                    events=writer,
                    run_id=f"sub_{task.id}",
                    max_steps=10,
                    event_bus=self._event_bus,
                )

                try:
                    result_text = await sub_loop.run(goal=sub_goal, system_prompt=system_prompt)
                    task.result = result_text
                    task.tool_calls_count = sub_loop.step_number
                    total_tool_calls += sub_loop.step_number
                    task.status = TaskStatus.COMPLETED
                    completed.append(tid)
                except Exception as exc:
                    task.error = str(exc)
                    task.status = TaskStatus.FAILED
                    failed.append(tid)
                finally:
                    writer.close()

                pending.discard(tid)

        # Build summary
        summary_parts = [
            f"执行完成: {len(completed)} 成功, {len(failed)} 失败, {len(skipped)} 跳过",
        ]
        for tid in plan.execution_order:
            t = task_map[tid]
            icon = "✓" if t.status == TaskStatus.COMPLETED else "✗" if t.status == TaskStatus.FAILED else "⏭"
            summary_parts.append(f"  {icon} {t.id}: {t.title}")
        final_summary = "\n".join(summary_parts)

        return PlanResult(
            success=len(failed) == 0,
            plan=plan,
            completed=completed,
            failed=failed,
            skipped=skipped,
            total_tool_calls=total_tool_calls,
            final_summary=final_summary,
        )


def _all_deps_completed(depends_on: list[str], completed: list[str]) -> bool:
    return all(d in completed for d in depends_on)


def _extract_json(text: str) -> dict:
    """Extract a JSON object from LLM output that may have markdown fences."""
    text = text.strip()
    # Remove ```json ... ``` fences
    if text.startswith("```"):
        lines = text.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)
    return json.loads(text)
