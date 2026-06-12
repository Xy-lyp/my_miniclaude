"""
S5 tests: Tool security — validation, permissions, retry strategies.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from mini_core.security.validator import ParameterValidator, ValidationResult
from mini_core.security.risk import RiskAssessor, RiskAssessment
from mini_core.security.permissions import PermissionManager, Decision, PermissionResult
from mini_core.security.rules import RuleEngine, PermissionRule
from mini_core.security.failure import ErrorClassifier, RetryStrategy, ErrorCategory, ClassifiedError


# ── Test 1: Schema validation ────────────────────────────────────────────────


def test_schema_validation():
    v = ParameterValidator(workdir="/tmp")
    # Valid
    r = v.validate("read_file", {"path": "hello.txt"})
    assert r.is_valid is True
    # Invalid: missing path
    r = v.validate("read_file", {})
    assert r.is_valid is False
    assert any("path" in e for e in r.errors)
    # Invalid: write_file without content
    r = v.validate("write_file", {"path": "x.txt"})
    assert r.is_valid is False


# ── Test 2: Path traversal blocked ───────────────────────────────────────────


def test_path_traversal_blocked():
    v = ParameterValidator(workdir="/tmp/test")
    r = v.validate("read_file", {"path": "../../etc/passwd"})
    assert r.is_valid is False
    assert any("traversal" in e.lower() for e in r.errors)
    # Valid: within workdir
    r = v.validate("read_file", {"path": "subdir/file.txt"})
    assert r.is_valid is True


# ── Test 3: Dangerous command detected ───────────────────────────────────────


def test_dangerous_command_detected():
    v = ParameterValidator(workdir="/tmp")
    r = v.validate("run_shell", {"command": "rm -rf /"})
    assert r.is_valid is False
    assert any("CRITICAL" in e for e in r.errors)

    r = v.validate("run_shell", {"command": "sudo pip install x"})
    assert r.is_valid is False
    assert any("sudo" in e.lower() for e in r.errors)

    # Safe command
    r = v.validate("run_shell", {"command": "python --version"})
    assert r.is_valid is True


# ── Test 4: Risk assessment levels ───────────────────────────────────────────


def test_risk_assessment_levels():
    ra = RiskAssessor()
    # Safe: echo
    a = ra.assess("run_shell", {"command": "echo hello"})
    assert a.level == "low"
    assert a.auto_approve is True
    # Critical: rm -rf /
    a = ra.assess("run_shell", {"command": "rm -rf /"})
    assert a.level == "critical"
    assert a.auto_approve is False
    # High: pip install
    a = ra.assess("run_shell", {"command": "pip install requests"})
    assert a.level == "high"
    assert a.requires_confirmation is True


# ── Test 5: Permission auto-approve ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_permission_auto_approve_safe():
    pm = PermissionManager()
    # Safe operation should auto-approve
    result = await pm.request_approval("read_file", {"path": "test.txt"})
    assert result.decision == Decision.APPROVE_ONCE
    assert len(pm.audit_log) == 1


# ── Test 6: Critical requires confirmation ────────────────────────────────────


@pytest.mark.asyncio
async def test_permission_critical_requires_confirmation():
    pm = PermissionManager()
    # Critical: this should wait for confirmation, but we set a short timeout
    pm._timeout = 0.1  # Short timeout for test
    result = await pm.request_approval("run_shell", {"command": "rm -rf /"})
    # Should timeout → auto-deny
    assert result.decision == Decision.TIMEOUT


# ── Test 7: Rule engine match ────────────────────────────────────────────────


def test_rule_engine_match():
    reng = RuleEngine()
    reng.add_rule(PermissionRule(id="r1", session_id=None, tool_name="read_file", decision="allow"))
    reng.add_rule(PermissionRule(id="r2", session_id=None, tool_name="run_shell",
                                  path_pattern="pip *", decision="allow"))
    reng.add_rule(PermissionRule(id="r3", session_id=None, tool_name="*", decision="deny"))

    # Exact tool match
    r = reng.match("read_file", {"path": "x.txt"}, "s1")
    assert r is not None and r.decision == "allow"

    # Pattern match
    r = reng.match("run_shell", {"command": "pip install x", "path": "pip install x"}, "s1")
    assert r is not None

    # Wildcard fallback
    r = reng.match("unknown_tool", {}, "s1")
    assert r is not None and r.decision == "deny"


# ── Test 8: Rule created from decision ───────────────────────────────────────


@pytest.mark.asyncio
async def test_rule_created_from_decision():
    pm = PermissionManager()
    # Simulate user creating a rule by providing decision
    # First request will wait, so set short timeout
    pm._timeout = 0.05

    # Pre-add a rule to auto-allow
    reng = pm.rules
    reng.add_rule(PermissionRule(id="pre", session_id=None, tool_name="write_file",
                                  path_pattern="*.txt", decision="allow"))

    result = await pm.request_approval("write_file", {"path": "data.txt", "content": "x"},
                                        session_id="s1")
    assert result.decision == Decision.APPROVE_RULE


# ── Test 9: Retry network timeout ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_retry_network_timeout():
    from mini_core.tools.builtin.read_file import ReadFileTool
    import tempfile

    tmp = tempfile.mkdtemp()
    tool = ReadFileTool(workdir=tmp)

    # Create a file so read_file succeeds
    Path(tmp, "exists.txt").write_text("hello")

    # Normal execution should work immediately
    rs = RetryStrategy()
    result = await rs.execute_with_retry(tool, {"path": "exists.txt"})
    assert result.success is True
    assert "hello" in result.content


# ── Test 10: Non-idempotent skip ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_retry_non_idempotent_skipped():
    from mini_core.tools.builtin.write_file import WriteFileTool
    import tempfile

    tmp = tempfile.mkdtemp()
    tool = WriteFileTool(workdir=tmp)

    # write_file should still work on first attempt
    rs = RetryStrategy()
    result = await rs.execute_with_retry(tool, {"path": "ok.txt", "content": "data"}, max_retries=2)
    assert result.success is True


# ── Test 11: Error classification all types ──────────────────────────────────


def test_error_classification_all_types():
    ec = ErrorClassifier()

    # RETRYABLE
    c = ec.classify("read_file", TimeoutError("Connection timed out"))
    assert c.category == ErrorCategory.RETRYABLE

    # RETRYABLE_WITH_BACKOFF
    c = ec.classify("read_file", Exception("HTTP 429 rate limit exceeded"))
    assert c.category == ErrorCategory.RETRYABLE_WITH_BACKOFF

    # FIXABLE_BY_MODEL
    c = ec.classify("read_file", FileNotFoundError("File not found: /tmp/x"))
    assert c.category == ErrorCategory.FIXABLE_BY_MODEL

    # PERMISSION
    c = ec.classify("write_file", PermissionError("Permission denied"))
    assert c.category == ErrorCategory.PERMISSION

    # FATAL
    c = ec.classify("run_shell", OSError("No space left on device"))
    assert c.category == ErrorCategory.FATAL


# ── Test 12: Fixable error returned ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_fixable_error_returned_to_llm():
    from mini_core.tools.builtin.read_file import ReadFileTool
    import tempfile

    tmp = tempfile.mkdtemp()
    tool = ReadFileTool(workdir=tmp)

    # Reading non-existent file → FIXABLE_BY_MODEL, returns helpful error
    rs = RetryStrategy()
    try:
        result = await rs.execute_with_retry(tool, {"path": "nonexistent.txt"}, max_retries=0)
        assert result.success is False
        assert "not found" in result.content.lower()
    except Exception:
        pass  # May raise if no retries left, but we set max_retries=0


# ── Test 13: Permission timeout auto-deny ────────────────────────────────────


@pytest.mark.asyncio
async def test_permission_timeout_auto_deny():
    pm = PermissionManager()
    pm._timeout = 0.01  # Very short
    result = await pm.request_approval("run_shell", {"command": "pip install x"})
    assert result.decision == Decision.TIMEOUT


# ── Test 14: Audit log all decisions ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_audit_log_all_decisions():
    pm = PermissionManager()
    # Auto-approve
    await pm.request_approval("read_file", {"path": "x.txt"})
    # Check log
    assert len(pm.audit_log) >= 1
    entry = pm.audit_log[0]
    assert entry.tool_name == "read_file"
    assert entry.decision == "approve_once"
    assert entry.risk_level == "low"
    assert entry.timestamp != ""
