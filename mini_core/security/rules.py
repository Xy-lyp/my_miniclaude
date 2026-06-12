"""
RuleEngine — remembers user permission decisions and auto-applies them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from fnmatch import fnmatch
from typing import Any


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass
class PermissionRule:
    id: str
    session_id: str | None  # None = global
    tool_name: str
    path_pattern: str | None = None
    arg_conditions: dict | None = None
    decision: str = "allow"  # "allow" | "deny"
    created_at: str = ""
    usage_count: int = 0
    last_used_at: str | None = None

    def __post_init__(self):
        if not self.created_at:
            self.created_at = _now()


class RuleEngine:
    """Matches tool calls against stored permission rules."""

    def __init__(self) -> None:
        self._rules: list[PermissionRule] = []

    def match(self, tool_name: str, args: dict, session_id: str) -> PermissionRule | None:
        candidates: list[tuple[PermissionRule, int]] = []

        for rule in self._rules:
            # Session filter: global rules match always, session rules match session_id
            if rule.session_id and rule.session_id != session_id:
                continue

            score = 0
            # Tool name match
            if rule.tool_name == tool_name:
                score += 10
            elif rule.tool_name == "*":
                score += 5
            else:
                continue

            # Path pattern match
            path = args.get("path", "")
            if rule.path_pattern:
                if fnmatch(path, rule.path_pattern):
                    score += 8
            elif path:
                score += 1

            # Arg conditions match
            if rule.arg_conditions:
                all_match = True
                for k, v in rule.arg_conditions.items():
                    if str(args.get(k, "")) != str(v):
                        all_match = False
                        break
                if all_match:
                    score += 6

            if score > 0:
                candidates.append((rule, score))

        if candidates:
            candidates.sort(key=lambda x: -x[1])
            best = candidates[0][0]
            best.usage_count += 1
            best.last_used_at = _now()
            return best

        return None

    def add_rule(self, rule: PermissionRule) -> None:
        self._rules.append(rule)

    def remove_rule(self, rule_id: str) -> None:
        self._rules = [r for r in self._rules if r.id != rule_id]

    def list_rules(self, session_id: str | None = None) -> list[PermissionRule]:
        if session_id is None:
            return list(self._rules)
        return [r for r in self._rules if r.session_id is None or r.session_id == session_id]
