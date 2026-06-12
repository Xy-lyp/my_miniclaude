"""
S4 tests: Session → Thread → Notes three-layer memory system.

Tests CRUD, auto-extraction, recall, dedup, cascade delete, and isolation.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from mini_core.memory.store import MemoryStore
from mini_core.memory.session import SessionManager
from mini_core.memory.thread import ThreadManager
from mini_core.memory.notes import NotesManager, MemoryExtractor
from mini_core.memory.recall import MemoryRecall
from mini_core.llm.provider import LLMResponse, TokenUsage


@pytest.fixture
def store(tmp_path):
    db_path = tmp_path / "test.db"
    s = MemoryStore(db_path=str(db_path))
    yield s


@pytest.fixture
def session(store):
    sm = SessionManager(store)
    return sm.create("test-project", workdir="/tmp/test", system_prompt_override="Be helpful")


# ── Test 1: Session CRUD ──────────────────────────────────────────────────────


def test_session_crud(store):
    sm = SessionManager(store)
    # Create
    s = sm.create("my-project", workdir="/tmp/proj")
    assert s["name"] == "my-project"
    assert s["thread_count"] == 0
    # Read
    s2 = sm.get(s["id"])
    assert s2["name"] == "my-project"
    # Update
    s3 = sm.update(s["id"], name="renamed-project")
    assert s3["name"] == "renamed-project"
    # List
    sessions = sm.list_all()
    assert len(sessions) >= 1
    # Delete
    assert sm.delete(s["id"]) is True
    assert sm.get(s["id"]) is None


# ── Test 2: Thread CRUD ──────────────────────────────────────────────────────


def test_thread_crud(store, session):
    tm = ThreadManager(store)
    t = tm.create(session["id"], goal="Create a Flask app", title="flask-setup")
    assert t["session_id"] == session["id"]
    assert t["status"] == "running"
    # Get
    t2 = tm.get(t["id"])
    assert t2["goal"] == "Create a Flask app"
    # Complete
    t3 = tm.complete(t["id"], messages=[{"role": "user", "content": "hi"}],
                     steps=3, tool_calls=5, prompt_tokens=100, completion_tokens=50, run_id="run-1")
    assert t3["status"] == "completed"
    assert t3["step_count"] == 3
    # List by session
    threads = tm.list_by_session(session["id"])
    assert len(threads) == 1


# ── Test 3: Thread continue ──────────────────────────────────────────────────


def test_thread_continue(store, session):
    tm = ThreadManager(store)
    t1 = tm.create(session["id"], goal="Step 1", title="part1")
    t1 = tm.complete(t1["id"], messages=[{"role": "user", "content": "do X"}],
                     steps=1, tool_calls=1, prompt_tokens=10, completion_tokens=5)
    # Continue by loading messages
    loaded = tm.get(t1["id"])
    assert len(loaded["messages"]) == 1
    # Create child thread
    t2 = tm.create(session["id"], goal="Step 2", title="part2", parent_thread_id=t1["id"])
    assert t2["parent_thread_id"] == t1["id"]


# ── Test 4: Auto-extract user preference ─────────────────────────────────────


@pytest.mark.asyncio
async def test_auto_extract_preference(store, session):
    mock_llm = MagicMock()
    mock_llm.chat = AsyncMock(return_value=LLMResponse(
        content='[{"title": "使用 pytest", "content": "用户偏好使用 pytest 进行测试", "importance": 8}]',
        finish_reason="stop",
        usage=TokenUsage(prompt_tokens=10, completion_tokens=10, total_tokens=20),
    ))
    extractor = MemoryExtractor(store, llm=mock_llm)
    thread = {
        "id": "th-1", "session_id": session["id"], "goal": "test",
        "messages": [
            {"role": "user", "content": "请使用 pytest 写测试"},
            {"role": "assistant", "content": "好的，我会使用 pytest。"},
        ],
    }
    notes = await extractor.extract_from_thread(thread)
    # Should have at least one extracted note
    assert len(notes) >= 1
    assert notes[0]["note_type"] in ("user_preference", "project_context", "learning", "decision")
    nm = NotesManager(store)
    all_notes = nm.list_by_session(session["id"])
    assert len(all_notes) >= 1


# ── Test 5: Auto-extract learning ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_auto_extract_learning(store, session):
    class LearningMockLLM:
        def __init__(self):
            self._calls = 0
        async def chat(self, messages, tools=None):
            self._calls += 1
            # Only return learning notes; other extraction types get empty
            import json as _json
            if "经验教训" in str(messages):
                return LLMResponse(
                    content='[{"title": "ModuleNotFoundError fix", "content": "Need pip install aiohttp", "importance": 9}]',
                    finish_reason="stop",
                    usage=TokenUsage(prompt_tokens=10, completion_tokens=10, total_tokens=20),
                )
            return LLMResponse(content='[]', finish_reason="stop", usage=TokenUsage(prompt_tokens=5, completion_tokens=2, total_tokens=7))
    extractor = MemoryExtractor(store, llm=LearningMockLLM())
    thread = {
        "id": "th-2", "session_id": session["id"], "goal": "fix error",
        "messages": [
            {"role": "assistant", "content": "Error: ModuleNotFoundError: No module named 'aiohttp'"},
            {"role": "user", "content": "fix it"},
            {"role": "assistant", "content": "Running: pip install aiohttp"},
        ],
    }
    notes = await extractor.extract_from_thread(thread)
    assert len(notes) >= 1
    # Should be a learning note
    learn_types = [n["note_type"] for n in notes]
    assert "learning" in learn_types


# ── Test 6: Memory recall injects into prompt ─────────────────────────────────


def test_memory_recall_injects_into_prompt(store, session):
    nm = NotesManager(store)
    nm.create(session["id"], title="使用 pytest", content="用户偏好使用 pytest",
              note_type="user_preference", importance=8)
    nm.create(session["id"], title="Poetry 管理依赖", content="项目使用 Poetry",
              note_type="project_context", importance=7)
    nm.create(session["id"], title="数据库 PostgreSQL", content="使用 PostgreSQL + asyncpg",
              note_type="decision", importance=9)

    recall = MemoryRecall(store)
    notes = recall.recall(session["id"], goal="创建一个 Python 项目，需要数据库和测试")
    assert len(notes) >= 1

    formatted = recall.format_for_prompt(notes)
    assert "pytest" in formatted.lower()
    assert "## Relevant Context" in formatted


# ── Test 7: Duplicate note detection ──────────────────────────────────────────


def test_duplicate_note_detection(store, session):
    nm = NotesManager(store)
    n1 = nm.create(session["id"], title="使用 pytest", content="v1",
                   note_type="user_preference")
    # Upsert with same title should update
    n2 = nm.upsert(session["id"], title="使用 pytest", content="v2",
                   note_type="user_preference", source="auto")
    assert n2["id"] == n1["id"]  # Same ID
    assert n2["content"] == "v2"  # Updated content
    # Only one note exists
    notes = nm.list_by_session(session["id"])
    assert len(notes) == 1


# ── Test 8: Memory recall semantic/keyword search ─────────────────────────────


def test_memory_recall_semantic_search(store, session):
    nm = NotesManager(store)
    nm.create(session["id"], title="Python code style", content="Use black formatter, line-length=88",
              note_type="user_preference", importance=8, tags=["python", "style"])
    nm.create(session["id"], title="JavaScript config", content="Use prettier for formatting",
              note_type="user_preference", importance=5, tags=["js", "style"])
    nm.create(session["id"], title="Database choice", content="PostgreSQL + asyncpg",
              note_type="decision", importance=9, tags=["database"])

    # Keyword search
    results = nm.search(session["id"], "python formatter")
    assert len(results) >= 1
    assert "black" in results[0]["content"].lower()

    # Recall by goal
    recall = MemoryRecall(store)
    notes = recall.recall(session["id"], "Write some Python code with tests")
    assert len(notes) >= 1
    assert any("python" in n.get("title", "").lower() or "black" in n.get("content", "").lower() for n in notes)


# ── Test 9: Cross-session isolation ───────────────────────────────────────────


def test_cross_session_isolation(store):
    sm = SessionManager(store)
    s1 = sm.create("project-a")
    s2 = sm.create("project-b")

    nm = NotesManager(store)
    nm.create(s1["id"], title="A 的偏好", content="data-A", note_type="user_preference")
    nm.create(s2["id"], title="B 的偏好", content="data-B", note_type="user_preference")

    # Each session only sees its own notes
    notes_a = nm.list_by_session(s1["id"])
    notes_b = nm.list_by_session(s2["id"])
    assert len(notes_a) == 1
    assert len(notes_b) == 1
    assert notes_a[0]["title"] == "A 的偏好"
    assert notes_b[0]["title"] == "B 的偏好"

    # Recall should be isolated
    recall = MemoryRecall(store)
    ra = recall.recall(s1["id"], "偏好")
    rb = recall.recall(s2["id"], "偏好")
    assert ra[0]["content"] == "data-A"
    assert rb[0]["content"] == "data-B"


# ── Test 10: Session delete cascades ──────────────────────────────────────────


def test_session_delete_cascades(store):
    sm = SessionManager(store)
    s = sm.create("temp-session")

    tm = ThreadManager(store)
    t = tm.create(s["id"], goal="temp", title="temp-thread")

    nm = NotesManager(store)
    n = nm.create(s["id"], title="temp-note", content="temp", note_type="project_context")

    # Delete session
    sm.delete(s["id"])

    # All related data should be gone
    assert tm.get(t["id"]) is None
    assert nm.get(n["id"]) is None
    assert store.get_session(s["id"]) is None
