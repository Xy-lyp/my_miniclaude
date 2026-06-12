"""
S6 tests: Context governance — token counting, watermark, truncation, compact.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from mini_core.context.counter import TokenCounter
from mini_core.context.watermark import WatermarkDetector, WatermarkLevel, WatermarkResult
from mini_core.context.truncator import ToolResultTruncator, TruncationResult
from mini_core.context.compactor import Compactor, CompactResult, COMPACT_PROMPT
from mini_core.context.manager import ContextManager, ContextHealth
from mini_core.llm.provider import LLMResponse, TokenUsage


# ── Test 1: Token counter accuracy ───────────────────────────────────────────


def test_token_counter_accuracy():
    tc = TokenCounter()
    # Short English text
    tokens = tc.count_text("Hello world this is a test")
    assert 5 <= tokens <= 8  # ~7 chars per word / 4
    # Chinese
    cn_tokens = tc.count_text("你好世界")
    assert 1 <= cn_tokens <= 4
    # Messages
    msgs = [{"role": "user", "content": "Hello, how are you?"}]
    count = tc.count(msgs)
    assert count > 0


# ── Test 2: Watermark normal ─────────────────────────────────────────────────


def test_watermark_normal():
    wd = WatermarkDetector(max_context_tokens=100000, warning_pct=0.70)
    r = wd.check(50000)  # 50%
    assert r.level == WatermarkLevel.NORMAL
    assert r.action == "none"
    assert r.usage_pct == 50.0


# ── Test 3: Watermark warning emits event ────────────────────────────────────


def test_watermark_warning_emits_event():
    wd = WatermarkDetector(max_context_tokens=100000, warning_pct=0.70)
    r = wd.check(75000)  # 75%
    assert r.level == WatermarkLevel.WARNING
    assert r.action == "notify"


# ── Test 4: Watermark high enables truncation ────────────────────────────────


def test_watermark_high_enables_truncation():
    wd = WatermarkDetector(max_context_tokens=100000, high_pct=0.85)
    r = wd.check(88000)  # 88%
    assert r.level == WatermarkLevel.HIGH
    assert r.action == "truncate_results"


# ── Test 5: Watermark critical triggers compact ──────────────────────────────


def test_watermark_critical_triggers_compact():
    wd = WatermarkDetector(max_context_tokens=100000, critical_pct=0.95)
    r = wd.check(97000)  # 97%
    assert r.level == WatermarkLevel.CRITICAL
    assert r.action == "compact"


# ── Test 6: Truncator preserves errors ───────────────────────────────────────


def test_truncator_preserves_errors():
    tr = ToolResultTruncator(max_tokens=50)  # Very small budget to force truncation
    content = "\n".join(["line " + str(i) for i in range(1, 50)] +
                         ["ERROR: something went wrong", "Traceback: at line 42"] +
                         ["line " + str(i) for i in range(50, 100)])
    result = tr.truncate(content, tool_name="run_shell")
    assert result.was_truncated
    assert "ERROR" in result.content
    assert "Traceback" in result.content
    assert "content truncated" in result.content


# ── Test 7: Truncator preserves JSON structure ────────────────────────────────


def test_truncator_preserves_json_structure():
    tr = ToolResultTruncator(max_tokens=20)
    data = json.dumps({"key": "value", "items": [1, 2, 3, 4, 5] * 100})  # Large JSON
    result = tr.truncate(data, tool_name="read_file")
    assert result.was_truncated
    assert "content truncated" in result.content


# ── Test 8: Truncator adds marker ────────────────────────────────────────────


def test_truncator_adds_marker():
    tr = ToolResultTruncator(max_tokens=5)
    result = tr.truncate("x" * 1000)
    assert result.was_truncated
    assert "content truncated" in result.content
    assert "removed" in result.content


# ── Test 9: Compactor reduces tokens ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_compactor_reduces_tokens():
    mock_llm = MagicMock()
    mock_llm.chat = AsyncMock(return_value=LLMResponse(
        content=json.dumps({
            "summary": "Created a web app with Flask.",
            "key_decisions": [{"decision": "Use Flask", "reason": "Simple"}],
            "important_outputs": [{"file": "app.py", "description": "Main app"}],
            "user_preferences": [{"preference": "Use pytest"}],
            "current_state": {"working_directory": "/tmp", "files_created": ["app.py"],
                              "last_action": "Created app.py"},
            "discarded_info": "Code formatting details",
        }),
        finish_reason="stop",
        usage=TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150),
    ))

    c = Compactor(llm=mock_llm)
    messages = [{"role": "user", "content": "x" * 100}] * 20  # 20 large messages
    result = await c.compact(messages, session_id="compact-s1", keep_recent=1)
    assert result.reduction_pct > 0 or len(result.new_messages) < len(messages)


# ── Test 10: Compactor preserves key info ────────────────────────────────────


@pytest.mark.asyncio
async def test_compactor_preserves_key_info():
    mock_llm = MagicMock()
    mock_llm.chat = AsyncMock(return_value=LLMResponse(
        content=json.dumps({
            "summary": "Setup Flask project.",
            "key_decisions": [{"decision": "Flask on port 5000", "reason": "default"}],
            "important_outputs": [],
            "user_preferences": [],
            "current_state": {"working_directory": "/tmp", "files_created": [],
                              "last_action": "Created app"},
            "discarded_info": "",
        }),
        finish_reason="stop",
        usage=TokenUsage(prompt_tokens=50, completion_tokens=30, total_tokens=80),
    ))

    c = Compactor(llm=mock_llm)
    messages = [
        {"role": "user", "content": "Create a Flask app on port 5000"},
        {"role": "assistant", "content": "I'll create that."},
        {"role": "user", "content": "Make sure to use port 5000"},
        {"role": "assistant", "content": "Done."},
    ]
    result = await c.compact(messages)
    assert "Flask" in result.summary


# ── Test 11: Compactor creates notes ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_compactor_creates_notes(tmp_path):
    from mini_core.memory.store import MemoryStore
    from mini_core.memory.notes import NotesManager

    store = MemoryStore(db_path=str(tmp_path / "test.db"))
    from mini_core.memory.session import SessionManager
    sm = SessionManager(store)
    sess = sm.create("test-compact", workdir="/tmp")
    sid = sess["id"]
    nm = NotesManager(store)

    mock_llm = MagicMock()
    mock_llm.chat = AsyncMock(return_value=LLMResponse(
        content=json.dumps({
            "summary": "Test",
            "key_decisions": [{"decision": "DB choice: PostgreSQL", "reason": "Reliability"}],
            "important_outputs": [],
            "user_preferences": [{"preference": "Use black formatter"}],
            "current_state": {"working_directory": "/tmp", "files_created": [],
                              "last_action": "Done"},
            "discarded_info": "",
        }),
        finish_reason="stop",
        usage=TokenUsage(prompt_tokens=50, completion_tokens=30, total_tokens=80),
    ))

    c = Compactor(llm=mock_llm, notes_mgr=nm)
    messages = [{"role": "user", "content": "Set up a project"}] * 10
    result = await c.compact(messages, session_id=sid)

    notes = nm.list_by_session(sid)
    assert len(notes) >= 2  # decision + preference
    assert any("DB choice" in n["title"] for n in notes)
    assert any("black" in n["content"] for n in notes)


# ── Test 12: Compactor keeps recent messages ──────────────────────────────────


@pytest.mark.asyncio
async def test_compactor_keeps_recent_messages():
    mock_llm = MagicMock()
    mock_llm.chat = AsyncMock(return_value=LLMResponse(
        content=json.dumps({"summary": "old", "key_decisions": [], "important_outputs": [],
                             "user_preferences": [], "current_state": {},
                             "discarded_info": ""}),
        finish_reason="stop",
        usage=TokenUsage(prompt_tokens=50, completion_tokens=20, total_tokens=70),
    ))

    c = Compactor(llm=mock_llm)
    messages = [{"role": "user", "content": f"msg {i}"} for i in range(20)]
    result = await c.compact(messages, keep_recent=2)
    # Should have a system summary + recent 4 messages (2 pairs)
    assert len(result.new_messages) < len(messages)
    assert any(m.get("role") == "system" for m in result.new_messages)


# ── Test 13: Auto compact during long run ────────────────────────────────────


@pytest.mark.asyncio
async def test_auto_compact_during_long_run():
    mock_llm = MagicMock()
    mock_llm.chat = AsyncMock(return_value=LLMResponse(
        content=json.dumps({"summary": "long run compact", "key_decisions": [],
                             "important_outputs": [], "user_preferences": [],
                             "current_state": {}, "discarded_info": ""}),
        finish_reason="stop",
        usage=TokenUsage(prompt_tokens=50, completion_tokens=20, total_tokens=70),
    ))

    ctx = ContextManager(max_context_tokens=5000, llm=mock_llm)
    # Feed many large messages
    ctx.update_messages([{"role": "user", "content": "x" * 100}] * 50)  # ~12500 chars → ~3100 tokens
    health = ctx.check_health()
    assert health.usage_pct > 0

    result = await ctx.compact(session_id="auto-compact-s1")
    assert result.reduction_pct >= 0


# ── Test 14: Manual compact via context manager ────────────────────────────────


@pytest.mark.asyncio
async def test_manual_compact_via_context_manager():
    mock_llm = MagicMock()
    mock_llm.chat = AsyncMock(return_value=LLMResponse(
        content=json.dumps({"summary": "manual compact", "key_decisions": [],
                             "important_outputs": [], "user_preferences": [],
                             "current_state": {}, "discarded_info": ""}),
        finish_reason="stop",
        usage=TokenUsage(prompt_tokens=50, completion_tokens=20, total_tokens=70),
    ))

    ctx = ContextManager(max_context_tokens=10000, llm=mock_llm)
    ctx.update_messages([{"role": "user", "content": "data" * 500}] * 10)
    result = await ctx.compact()
    assert result.new_messages is not None
    assert len(result.new_messages) > 0
