"""
S1 tests: Agent Loop — ReAct pattern + tool execution + event logging.

All tests use a MockLLMProvider to avoid depending on external APIs.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import pytest

from mini_core.agent.event import (
    AgentEventType,
    EventWriter,
    make_event,
    read_events_file,
)
from mini_core.agent.loop import AgentLoop, MaxStepsExceeded
from mini_core.agent.runner import AgentRunner, RunResult
from mini_core.llm.provider import (
    LLMProvider,
    LLMResponse,
    LLMError,
    ToolCall,
    TokenUsage,
)
from mini_core.tools.base import Tool, ToolResult
from mini_core.tools.registry import ToolRegistry
from mini_core.tools.builtin.read_file import ReadFileTool
from mini_core.tools.builtin.write_file import WriteFileTool
from mini_core.tools.builtin.run_shell import RunShellTool


# ── Mock LLM Provider ─────────────────────────────────────────────────────────


class MockLLMProvider(LLMProvider):
    """Returns canned responses from a queue, in order.

    Each call to chat() pops the next response from the queue.
    """

    def __init__(self, responses: list[LLMResponse] | None = None) -> None:
        self.responses: list[LLMResponse] = list(responses or [])
        self.calls: list[dict[str, Any]] = []  # Record all calls for assertions

    def add_response(self, resp: LLMResponse) -> None:
        self.responses.append(resp)

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        self.calls.append({"messages": messages, "tools": tools})
        if not self.responses:
            # Default: return a simple stop response
            return LLMResponse(
                content="I have completed the task.",
                finish_reason="stop",
                usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            )
        return self.responses.pop(0)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_text_response(text: str) -> LLMResponse:
    """Create a simple text (stop) response."""
    return LLMResponse(
        content=text,
        finish_reason="stop",
        usage=TokenUsage(prompt_tokens=10, completion_tokens=len(text.split()), total_tokens=10 + len(text.split())),
    )


def _make_tool_call_response(tool_calls: list[ToolCall], content: str | None = None) -> LLMResponse:
    """Create a tool_calls response."""
    return LLMResponse(
        content=content,
        tool_calls=tool_calls,
        finish_reason="tool_calls",
        usage=TokenUsage(prompt_tokens=20, completion_tokens=10, total_tokens=30),
    )


def _make_tool_call(name: str, arguments: dict[str, Any], call_id: str = "call_1") -> ToolCall:
    return ToolCall(id=call_id, name=name, arguments=arguments)


def _setup_registry(workdir: str) -> ToolRegistry:
    """Create a ToolRegistry with all 3 built-in tools."""
    reg = ToolRegistry()
    reg.register(ReadFileTool(workdir=workdir))
    reg.register(WriteFileTool(workdir=workdir))
    reg.register(RunShellTool(workdir=workdir))
    return reg


def _setup_event_writer(run_id: str, tmp_path: Path) -> EventWriter:
    """Create an EventWriter pointing to a temp file."""
    events_file = tmp_path / f"{run_id}.jsonl"
    writer = EventWriter(events_file)
    writer.open()
    return writer


# ── Test 1: Simple goal, no tools ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_simple_goal_no_tools(tmp_path):
    """Agent should answer a knowledge question without calling any tools."""
    mock_llm = MockLLMProvider([
        _make_text_response("The capital of France is Paris."),
    ])

    reg = _setup_registry(str(tmp_path))
    run_id = "test001"
    writer = _setup_event_writer(run_id, tmp_path)
    writer.write(make_event(AgentEventType.RUN_STARTED, run_id, 0, {"goal": "capital of France"}))

    loop = AgentLoop(llm=mock_llm, tools=reg, events=writer, run_id=run_id, max_steps=20)
    answer = await loop.run(goal="What is the capital of France?", system_prompt="You are helpful.")

    writer.close()

    assert "Paris" in answer
    assert loop.step_number == 1
    assert len(mock_llm.calls) == 1
    # Verify events file exists and has events
    events = read_events_file(writer.path)
    assert len(events) >= 2  # RUN_STARTED + LLM_RESPONSE + RUN_COMPLETED


# ── Test 2: Write and read file ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_write_and_read_file(tmp_path):
    """Agent writes a file, then reads it back to verify."""
    hello_path = tmp_path / "hello.txt"
    content = "Hello, World!"

    mock_llm = MockLLMProvider([
        # Step 1: Write the file
        _make_tool_call_response([
            _make_tool_call("write_file", {"path": "hello.txt", "content": content}, "tc1"),
        ]),
        # Step 2: Read it back to verify
        _make_tool_call_response([
            _make_tool_call("read_file", {"path": "hello.txt"}, "tc2"),
        ]),
        # Step 3: Final answer
        _make_text_response("File written and verified successfully."),
    ])

    reg = _setup_registry(str(tmp_path))
    run_id = "test002"
    writer = _setup_event_writer(run_id, tmp_path)
    writer.write(make_event(AgentEventType.RUN_STARTED, run_id, 0, {"goal": "write and verify"}))

    loop = AgentLoop(llm=mock_llm, tools=reg, events=writer, run_id=run_id, max_steps=20)
    answer = await loop.run(goal="Write hello.txt and verify", system_prompt="You are helpful.")
    writer.close()

    assert "success" in answer.lower() or "verified" in answer.lower()
    assert loop.step_number == 3
    # Verify the file was actually written
    assert hello_path.exists()
    assert hello_path.read_text() == content


# ── Test 3: Run shell command ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_run_shell_command(tmp_path):
    """Agent executes a shell command and gets the result."""
    mock_llm = MockLLMProvider([
        _make_tool_call_response([
            _make_tool_call("run_shell", {"command": "echo hello from shell"}, "tc1"),
        ]),
        _make_text_response("The shell command ran successfully."),
    ])

    reg = _setup_registry(str(tmp_path))
    run_id = "test003"
    writer = _setup_event_writer(run_id, tmp_path)
    writer.write(make_event(AgentEventType.RUN_STARTED, run_id, 0, {"goal": "run echo"}))

    loop = AgentLoop(llm=mock_llm, tools=reg, events=writer, run_id=run_id, max_steps=20)
    answer = await loop.run(goal="Run: echo hello from shell", system_prompt="You are helpful.")
    writer.close()

    assert loop.step_number == 2
    assert len(mock_llm.calls) == 2
    # The tool should have succeeded (echo always exits 0)
    events = read_events_file(writer.path)
    tool_results = [e for e in events if e["type"] == "tool_call_result"]
    assert len(tool_results) >= 1
    assert tool_results[0]["data"]["success"] is True


# ── Test 4: Multi-step task (3+ steps) ────────────────────────────────────────


@pytest.mark.asyncio
async def test_multi_step_task(tmp_path):
    """Agent executes a 3-step task: write file, read it, run shell on it."""
    mock_llm = MockLLMProvider([
        # Step 1: Write a Python script
        _make_tool_call_response([
            _make_tool_call("write_file", {"path": "greet.py", "content": "print('Hello from Python')"}, "tc1"),
        ]),
        # Step 2: Read it back
        _make_tool_call_response([
            _make_tool_call("read_file", {"path": "greet.py"}, "tc2"),
        ]),
        # Step 3: Run it
        _make_tool_call_response([
            _make_tool_call("run_shell", {"command": "python greet.py"}, "tc3"),
        ]),
        # Step 4: Final answer
        _make_text_response("Created greet.py, verified contents, and ran it. Output: Hello from Python"),
    ])

    reg = _setup_registry(str(tmp_path))
    run_id = "test004"
    writer = _setup_event_writer(run_id, tmp_path)
    writer.write(make_event(AgentEventType.RUN_STARTED, run_id, 0, {"goal": "multi-step"}))

    loop = AgentLoop(llm=mock_llm, tools=reg, events=writer, run_id=run_id, max_steps=20)
    answer = await loop.run(goal="Create and run greet.py", system_prompt="You are helpful.")
    writer.close()

    assert loop.step_number == 4
    assert "Hello from Python" in answer
    # File should exist
    assert (tmp_path / "greet.py").exists()


# ── Test 5: Events file created and valid ─────────────────────────────────────


@pytest.mark.asyncio
async def test_events_file_created(tmp_path):
    """Verify that the events.jsonl file is created with correct NDJSON format."""
    mock_llm = MockLLMProvider([
        _make_tool_call_response([
            _make_tool_call("write_file", {"path": "test.txt", "content": "data"}, "tc1"),
        ]),
        _make_text_response("Done."),
    ])

    reg = _setup_registry(str(tmp_path))
    run_id = "test005"
    writer = _setup_event_writer(run_id, tmp_path)
    writer.write(make_event(AgentEventType.RUN_STARTED, run_id, 0, {"goal": "create file"}))

    loop = AgentLoop(llm=mock_llm, tools=reg, events=writer, run_id=run_id, max_steps=20)
    await loop.run(goal="Create test.txt", system_prompt="You are helpful.")
    writer.close()

    # Read events file and validate
    events = read_events_file(writer.path)
    assert len(events) >= 4  # RUN_STARTED, LLM_REQUEST, LLM_RESPONSE, TOOL_CALL_START, TOOL_CALL_RESULT, STEP_COMPLETED, RUN_COMPLETED

    # Check event types are present
    event_types = [e["type"] for e in events]
    assert "run_started" in event_types
    assert "tool_call_start" in event_types
    assert "tool_call_result" in event_types
    assert "run_completed" in event_types

    # Each event must have required fields
    for evt in events:
        assert "type" in evt
        assert "timestamp" in evt
        assert "run_id" in evt
        assert evt["run_id"] == run_id
        assert "step_number" in evt
        assert "data" in evt
        # Verify it's valid JSON by re-serializing
        json.dumps(evt)


# ── Test 6: Max steps enforced ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_max_steps_enforced(tmp_path):
    """Agent should raise MaxStepsExceeded when it never gives a final answer."""
    # Create responses that always request tool calls (infinite loop)
    responses = []
    for i in range(25):
        responses.append(
            _make_tool_call_response([
                _make_tool_call("run_shell", {"command": f"echo step{i}"}, f"tc{i}"),
            ])
        )

    mock_llm = MockLLMProvider(responses)
    reg = _setup_registry(str(tmp_path))
    run_id = "test006"
    writer = _setup_event_writer(run_id, tmp_path)

    loop = AgentLoop(llm=mock_llm, tools=reg, events=writer, run_id=run_id, max_steps=3)

    with pytest.raises(MaxStepsExceeded) as exc_info:
        await loop.run(goal="Keep running forever", system_prompt="You are helpful.")

    writer.close()

    assert exc_info.value.max_steps == 3
    assert exc_info.value.steps_taken == 3

    # Verify run_error event was written
    events = read_events_file(writer.path)
    error_events = [e for e in events if e["type"] == "run_error"]
    assert len(error_events) >= 1
    assert "max_steps_exceeded" in error_events[0]["data"].get("error_type", "")


# ── Test 7: Tool not found error ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_tool_not_found_error(tmp_path):
    """When the LLM calls a tool that doesn't exist, the agent should handle it gracefully."""
    mock_llm = MockLLMProvider([
        # LLM calls a non-existent tool
        _make_tool_call_response([
            _make_tool_call("nonexistent_tool", {"arg": "value"}, "tc_bad"),
        ]),
        # After receiving the error, LLM adjusts and gives final answer
        _make_text_response("I tried to use a tool that doesn't exist. Let me try a different approach."),
    ])

    reg = _setup_registry(str(tmp_path))
    run_id = "test007"
    writer = _setup_event_writer(run_id, tmp_path)
    writer.write(make_event(AgentEventType.RUN_STARTED, run_id, 0, {"goal": "use bad tool"}))

    loop = AgentLoop(llm=mock_llm, tools=reg, events=writer, run_id=run_id, max_steps=20)
    answer = await loop.run(goal="Use a bad tool", system_prompt="You are helpful.")
    writer.close()

    # The agent should complete (not crash) and report the issue
    assert loop.step_number >= 1
    # Check that tool_call_result shows failure
    events = read_events_file(writer.path)
    tool_results = [e for e in events if e["type"] == "tool_call_result"]
    assert len(tool_results) >= 1
    # The result should indicate failure
    assert tool_results[0]["data"]["success"] is False
    assert "not found" in tool_results[0]["data"].get("content_preview", "").lower()


# ── Test 8: LLM error handling ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_llm_error_handling(tmp_path):
    """When the LLM API returns an error, the agent should surface it."""
    # Create a mock that raises LLMError on the second call
    class FailingLLM(LLMProvider):
        def __init__(self):
            self.call_count = 0

        async def chat(self, messages, tools=None):
            self.call_count += 1
            if self.call_count == 1:
                return _make_tool_call_response([
                    _make_tool_call("run_shell", {"command": "echo ok"}, "tc1"),
                ])
            # Second call fails
            raise LLMError("API rate limit exceeded")

    mock_llm = FailingLLM()
    reg = _setup_registry(str(tmp_path))
    run_id = "test008"
    writer = _setup_event_writer(run_id, tmp_path)
    writer.write(make_event(AgentEventType.RUN_STARTED, run_id, 0, {"goal": "test"}))

    loop = AgentLoop(llm=mock_llm, tools=reg, events=writer, run_id=run_id, max_steps=20)

    with pytest.raises(LLMError) as exc_info:
        await loop.run(goal="Do something then fail", system_prompt="You are helpful.")

    writer.close()

    assert "rate limit" in str(exc_info.value).lower()

    # Verify run_error event was written
    events = read_events_file(writer.path)
    error_events = [e for e in events if e["type"] == "run_error"]
    assert len(error_events) >= 1
    assert "llm_error" in error_events[0]["data"].get("error_type", "")


# ── Test: AgentRunner integration ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_agent_runner_integration(tmp_path, monkeypatch):
    """AgentRunner orchestrates a full run and returns a RunResult."""
    # Change to tmp_path so events/ is created there
    monkeypatch.chdir(tmp_path)

    mock_llm = MockLLMProvider([
        _make_text_response("Task completed: all good."),
    ])

    reg = _setup_registry(str(tmp_path))
    runner = AgentRunner(llm=mock_llm, tools=reg, max_steps=20)

    result = await runner.run(goal="Do a simple thing", workdir=str(tmp_path))

    assert isinstance(result, RunResult)
    assert result.success is True
    assert len(result.run_id) == 8  # hex uuid4[:8]
    assert result.steps == 1
    assert "all good" in result.final_answer
    assert result.events_file.endswith(".jsonl")

    # Verify events file exists
    events_path = Path(result.events_file)
    assert events_path.exists()
    events = read_events_file(events_path)
    assert len(events) >= 3  # RUN_STARTED, LLM_RESPONSE, RUN_COMPLETED
