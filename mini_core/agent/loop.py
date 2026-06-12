"""
AgentLoop — the ReAct (Reasoning + Acting) cycle.

Core algorithm:
  1. Initialize messages = [system_prompt, user_message(goal)]
  2. Loop:
     a. Call LLM.chat(messages, tools)
     b. If finish_reason == "stop" → return final answer
     c. If finish_reason == "tool_calls" → execute each tool, append results
     d. If steps > max_steps → raise MaxStepsExceeded
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from mini_core.agent.event import (
    AgentEventType,
    EventWriter,
    make_event,
)
from mini_core.events.types import (
    LLMRequestStartEvent,
    LLMResponseEvent,
    ToolCallStartEvent,
    ToolCallResultEvent,
    StepCompletedEvent,
)
from mini_core.events.bus import EventBus
from mini_core.llm.provider import LLMProvider, LLMError
from mini_core.tools.registry import ToolRegistry
from mini_core.tools.base import ToolResult

logger = logging.getLogger("mini-core.agent.loop")


class MaxStepsExceeded(Exception):
    """Raised when the agent exceeds the maximum number of steps."""

    def __init__(self, max_steps: int, steps_taken: int) -> None:
        super().__init__(f"Exceeded max steps: {max_steps} (took {steps_taken})")
        self.max_steps = max_steps
        self.steps_taken = steps_taken


class AgentLoop:
    """The ReAct loop: reason → act → observe → repeat.

    Each iteration is one "step".  Within a step the agent may make
    multiple tool calls (in sequence) before returning to the LLM.
    """

    def __init__(
        self,
        llm: LLMProvider,
        tools: ToolRegistry,
        events: EventWriter,
        run_id: str,
        max_steps: int = 20,
        event_bus: EventBus | None = None,
    ) -> None:
        self._llm = llm
        self._tools = tools
        self._events = events
        self._run_id = run_id
        self._max_steps = max_steps
        self._step = 0
        self._total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        self._event_bus = event_bus

    @property
    def step_number(self) -> int:
        return self._step

    @property
    def total_usage(self) -> dict[str, int]:
        return self._total_usage

    async def run(self, goal: str, system_prompt: str) -> str:
        """Run the ReAct loop and return the final answer.

        Args:
            goal: The user's goal / task description.
            system_prompt: The system prompt text.

        Returns:
            The final answer from the LLM.

        Raises:
            MaxStepsExceeded: If the loop exceeds max_steps.
        """
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": goal},
        ]

        tool_defs = self._tools.get_definitions()

        while self._step < self._max_steps:
            self._step += 1
            logger.info("Step %d: calling LLM with %d messages", self._step, len(messages))

            # Emit LLM_REQUEST_START (legacy)
            self._events.write(
                make_event(
                    AgentEventType.LLM_REQUEST_START,
                    run_id=self._run_id,
                    step_number=self._step,
                    data={"messages_count": len(messages)},
                )
            )
            # Emit S2 event
            if self._event_bus:
                self._event_bus.emit(LLMRequestStartEvent(
                    run_id=self._run_id,
                    step_number=self._step,
                    messages_count=len(messages),
                    tool_count=len(tool_defs),
                ))

            # Call LLM
            try:
                if tool_defs:
                    response = await self._llm.chat(messages, tools=tool_defs)
                else:
                    response = await self._llm.chat(messages, tools=None)
            except LLMError as exc:
                logger.error("LLM error at step %d: %s", self._step, exc)
                self._events.write(
                    make_event(
                        AgentEventType.RUN_ERROR,
                        run_id=self._run_id,
                        step_number=self._step,
                        data={"error": str(exc), "error_type": "llm_error"},
                    )
                )
                raise

            # Accumulate usage
            self._total_usage["prompt_tokens"] += response.usage.prompt_tokens
            self._total_usage["completion_tokens"] += response.usage.completion_tokens
            self._total_usage["total_tokens"] += response.usage.total_tokens

            # Emit LLM_RESPONSE (legacy)
            self._events.write(
                make_event(
                    AgentEventType.LLM_RESPONSE,
                    run_id=self._run_id,
                    step_number=self._step,
                    data={
                        "finish_reason": response.finish_reason,
                        "content": response.content[:500] if response.content else None,
                        "tool_calls": [tc.name for tc in response.tool_calls],
                        "usage": {
                            "prompt": response.usage.prompt_tokens,
                            "completion": response.usage.completion_tokens,
                        },
                    },
                )
            )
            # Emit S2 event
            if self._event_bus:
                self._event_bus.emit(LLMResponseEvent(
                    run_id=self._run_id,
                    step_number=self._step,
                    content=response.content,
                    tool_calls=[tc.name for tc in response.tool_calls],
                    finish_reason=response.finish_reason,
                    usage={
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                        "total_tokens": response.usage.total_tokens,
                    },
                ))

            # Case 1: Model is done — return final answer
            if response.finish_reason == "stop" or (response.content and not response.tool_calls):
                final_answer = response.content or ""
                self._events.write(
                    make_event(
                        AgentEventType.RUN_COMPLETED,
                        run_id=self._run_id,
                        step_number=self._step,
                        data={
                            "final_answer_preview": final_answer[:200],
                            "total_steps": self._step,
                            "token_usage": self._total_usage,
                        },
                    )
                )
                return final_answer

            # Case 2: Model wants to call tools
            if response.finish_reason == "tool_calls" or response.tool_calls:
                # Append assistant message with tool_calls
                assistant_msg: dict[str, Any] = {
                    "role": "assistant",
                    "content": response.content,
                }
                if response.tool_calls:
                    assistant_msg["tool_calls"] = [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
                        }
                        for tc in response.tool_calls
                    ]
                messages.append(assistant_msg)

                # Execute each tool call
                for tc in response.tool_calls:
                    # Emit TOOL_CALL_START (legacy)
                    self._events.write(
                        make_event(
                            AgentEventType.TOOL_CALL_START,
                            run_id=self._run_id,
                            step_number=self._step,
                            data={
                                "tool_name": tc.name,
                                "tool_call_id": tc.id,
                                "arguments": tc.arguments,
                            },
                        )
                    )
                    # Emit S2 event
                    _t_start = time.monotonic()
                    if self._event_bus:
                        self._event_bus.emit(ToolCallStartEvent(
                            run_id=self._run_id,
                            step_number=self._step,
                            tool_name=tc.name,
                            tool_args=tc.arguments,
                        ))

                    # Validate parameters
                    validation = self._tools.validate(tc.name, tc.arguments)
                    if not validation["valid"]:
                        error_msg = "; ".join(validation["errors"])
                        tool_result = ToolResult(
                            success=False,
                            content=f"Parameter validation error: {error_msg}",
                            raw_content=f"Validation errors: {error_msg}",
                            error=error_msg,
                        )
                    else:
                        # Look up and execute the tool
                        tool = self._tools.get(tc.name)
                        if tool is None:
                            tool_result = ToolResult(
                                success=False,
                                content=f"Error: Tool '{tc.name}' not found. Available tools: {', '.join(t.name for t in self._tools._tools.values())}",
                                raw_content="",
                                error=f"tool not found: {tc.name}",
                            )
                        else:
                            try:
                                tool_result = await tool.execute(**tc.arguments)
                            except Exception as exc:
                                logger.error("Tool '%s' execution failed: %s", tc.name, exc)
                                tool_result = ToolResult(
                                    success=False,
                                    content=f"Tool execution error: {exc}",
                                    raw_content=str(exc),
                                    error=str(exc),
                                )

                    # Emit TOOL_CALL_RESULT (legacy)
                    self._events.write(
                        make_event(
                            AgentEventType.TOOL_CALL_RESULT,
                            run_id=self._run_id,
                            step_number=self._step,
                            data={
                                "tool_name": tc.name,
                                "tool_call_id": tc.id,
                                "success": tool_result.success,
                                "content_preview": tool_result.content[:500],
                                "metadata": tool_result.metadata,
                            },
                        )
                    )
                    # Emit S2 event
                    _duration = (time.monotonic() - _t_start) * 1000
                    if self._event_bus:
                        self._event_bus.emit(ToolCallResultEvent(
                            run_id=self._run_id,
                            tool_name=tc.name,
                            success=tool_result.success,
                            content_length=len(tool_result.content),
                            duration_ms=round(_duration, 2),
                            error=tool_result.error,
                        ))

                    # Append tool result message
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": tool_result.content,
                    })

                # Emit STEP_COMPLETED (legacy + S2)
                action_type = "final" if response.tool_calls else "think"
                self._events.write(
                    make_event(
                        AgentEventType.STEP_COMPLETED,
                        run_id=self._run_id,
                        step_number=self._step,
                        data={"tools_executed": len(response.tool_calls)},
                    )
                )
                if self._event_bus:
                    self._event_bus.emit(StepCompletedEvent(
                        run_id=self._run_id,
                        step_number=self._step,
                        action_type=action_type,
                    ))

                # Loop back to step a
                continue

            # Unknown finish reason — treat as stop
            final = response.content or ""
            return final

        # Exceeded max steps
        err = MaxStepsExceeded(self._max_steps, self._step)
        self._events.write(
            make_event(
                AgentEventType.RUN_ERROR,
                run_id=self._run_id,
                step_number=self._step,
                data={"error": str(err), "error_type": "max_steps_exceeded"},
            )
        )
        raise err


