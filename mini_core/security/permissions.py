"""
PermissionManager — approval flow with risk-check, rule-matching, and timeout.
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from mini_core.security.risk import RiskAssessor, RiskAssessment
from mini_core.security.rules import RuleEngine, PermissionRule


class Decision(Enum):
    APPROVE_ONCE = "approve_once"
    APPROVE_RULE = "approve_rule"
    APPROVE_SESSION = "approve_session"
    DENY = "deny"
    DENY_ALWAYS = "deny_always"
    TIMEOUT = "timeout"


@dataclass
class PermissionResult:
    decision: Decision
    request_id: str = ""
    rule: PermissionRule | None = None


@dataclass
class AuditEntry:
    timestamp: str
    tool_name: str
    args_summary: str
    risk_level: str
    decision: str
    session_id: str = ""
    request_id: str = ""


class PermissionManager:
    def __init__(self) -> None:
        self._risk = RiskAssessor()
        self._rules = RuleEngine()
        self._pending: dict[str, asyncio.Event] = {}
        self._pending_decisions: dict[str, str] = {}
        self._audit_log: list[AuditEntry] = []
        self._timeout = 120.0

    @property
    def rules(self) -> RuleEngine:
        return self._rules

    @property
    def audit_log(self) -> list[AuditEntry]:
        return self._audit_log

    def assess(self, tool_name: str, args: dict) -> RiskAssessment:
        return self._risk.assess(tool_name, args)

    async def request_approval(self, tool_name: str, args: dict,
                                session_id: str = "") -> PermissionResult:
        risk = self._risk.assess(tool_name, args)
        args_summary = str(args)[:200]

        # Check rules first
        existing = self._rules.match(tool_name, args, session_id)
        if existing:
            decision = Decision.APPROVE_RULE if existing.decision == "allow" else Decision.DENY_ALWAYS
            self._audit_log.append(AuditEntry(
                timestamp=self._now(), tool_name=tool_name, args_summary=args_summary,
                risk_level=risk.level, decision=decision.value, session_id=session_id,
            ))
            return PermissionResult(decision=decision, rule=existing)

        # Auto-approve safe operations
        if risk.auto_approve:
            self._audit_log.append(AuditEntry(
                timestamp=self._now(), tool_name=tool_name, args_summary=args_summary,
                risk_level=risk.level, decision="approve_once", session_id=session_id,
            ))
            return PermissionResult(decision=Decision.APPROVE_ONCE)

        # Requires confirmation — emit event and wait
        request_id = uuid.uuid4().hex[:8]
        event = asyncio.Event()
        self._pending[request_id] = event

        try:
            decision_str = await asyncio.wait_for(event.wait(), timeout=self._timeout)
        except asyncio.TimeoutError:
            decision_str = "timeout"

        self._pending.pop(request_id, None)

        decision = Decision(decision_str) if decision_str in Decision._value2member_map_ else Decision.TIMEOUT

        # If user chose to create a rule
        rule = None
        if decision == Decision.APPROVE_RULE:
            path = args.get("path", "")
            rule = PermissionRule(
                id=uuid.uuid4().hex[:8], session_id=session_id,
                tool_name=tool_name,
                path_pattern=path if path else None,
                arg_conditions=None, decision="allow",
            )
            self._rules.add_rule(rule)
        elif decision == Decision.APPROVE_SESSION:
            rule = PermissionRule(
                id=uuid.uuid4().hex[:8], session_id=session_id,
                tool_name=tool_name, decision="allow",
            )
            self._rules.add_rule(rule)
        elif decision == Decision.DENY_ALWAYS:
            rule = PermissionRule(
                id=uuid.uuid4().hex[:8], session_id=session_id,
                tool_name=tool_name, decision="deny",
            )
            self._rules.add_rule(rule)

        self._audit_log.append(AuditEntry(
            timestamp=self._now(), tool_name=tool_name, args_summary=args_summary,
            risk_level=risk.level, decision=decision.value, session_id=session_id,
            request_id=request_id,
        ))

        return PermissionResult(decision=decision, request_id=request_id, rule=rule)

    def get_pending_requests(self) -> list[dict]:
        return [
            {"request_id": rid, "tool_name": "", "risk_level": "", "args": "", "timeout": self._timeout}
            for rid in self._pending
        ]

    def provide_decision(self, request_id: str, decision: str) -> bool:
        if request_id in self._pending:
            self._pending[request_id].set()
            self._pending_decisions[request_id] = decision
            return True
        return False

    def _now(self) -> str:
        from datetime import datetime, timezone
        return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
