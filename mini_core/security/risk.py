"""
RiskAssessor — determines risk level for tool calls.

Considers: tool type, arguments, context, history.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from mini_core.security.validator import DANGEROUS_COMMAND_PATTERNS
import re


@dataclass
class RiskAssessment:
    level: str  # "safe" | "low" | "medium" | "high" | "critical"
    auto_approve: bool = False
    reason: str = ""
    affected_paths: list[str] = field(default_factory=list)
    requires_confirmation: bool = False


# Static risk levels per tool
TOOL_RISK_LEVELS: dict[str, str] = {
    "read_file": "low",
    "write_file": "medium",
    "run_shell": "high",
    "task_planner": "low",
}


class RiskAssessor:
    def assess(self, tool_name: str, args: dict, context: dict | None = None) -> RiskAssessment:
        base_level = TOOL_RISK_LEVELS.get(tool_name, "medium")

        if tool_name == "read_file":
            return self._assess_read(args, base_level)
        elif tool_name == "write_file":
            return self._assess_write(args, base_level)
        elif tool_name == "run_shell":
            return self._assess_shell(args, base_level)
        else:
            return RiskAssessment(level=base_level, auto_approve=(base_level in ("safe", "low")),
                                  reason=f"Default risk for {tool_name}")

    def _assess_read(self, args: dict, base: str) -> RiskAssessment:
        path = args.get("path", "")
        if any(path.startswith(d) for d in ["/etc/", "/sys/", "C:\\Windows"]):
            return RiskAssessment(level="high", auto_approve=False,
                                  reason="Reading system files", affected_paths=[path],
                                  requires_confirmation=True)
        if ".." in path:
            return RiskAssessment(level="medium", auto_approve=False,
                                  reason="Path contains parent references", affected_paths=[path],
                                  requires_confirmation=True)
        return RiskAssessment(level="low", auto_approve=True, reason="Safe read within workdir")

    def _assess_write(self, args: dict, base: str) -> RiskAssessment:
        path = args.get("path", "")
        content = args.get("content", "")
        size_kb = len(content) / 1024 if content else 0
        if size_kb > 1000:
            return RiskAssessment(level="high", auto_approve=False,
                                  reason=f"Large file write ({size_kb:.0f}KB)", affected_paths=[path],
                                  requires_confirmation=True)
        if path.endswith((".env", ".secret", "credentials.json", "id_rsa")):
            return RiskAssessment(level="high", auto_approve=False,
                                  reason="Writing to sensitive filename pattern", affected_paths=[path],
                                  requires_confirmation=True)
        return RiskAssessment(level="medium", auto_approve=False,
                              reason="File write operation", affected_paths=[path],
                              requires_confirmation=True)

    def _assess_shell(self, args: dict, base: str) -> RiskAssessment:
        cmd = args.get("command", "")
        level = "high"
        reason = "Shell command execution"
        affected: list[str] = []

        for pattern, risk, desc in DANGEROUS_COMMAND_PATTERNS:
            if re.search(pattern, cmd):
                if risk == "CRITICAL":
                    return RiskAssessment(level="critical", auto_approve=False,
                                          reason=desc, requires_confirmation=True)
                elif risk == "HIGH":
                    level = "high"
                    reason = desc

        if any(kw in cmd for kw in ["pip install", "npm install", "cargo install", "brew install"]):
            level = "high"
            reason = "Package installation"

        if any(kw in cmd for kw in ["python --version", "python3 --version", "which ", "echo ",
                                      "ls ", "pwd", "date", "uname"]):
            return RiskAssessment(level="low", auto_approve=True, reason="Safe utility command")

        return RiskAssessment(level=level, auto_approve=(level == "low"),
                              reason=reason, affected_paths=affected,
                              requires_confirmation=(level in ("high", "critical")))
