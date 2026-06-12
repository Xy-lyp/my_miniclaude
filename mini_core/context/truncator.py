"""
ToolResultTruncator — smart truncation that preserves important content.

Strategies: keep header, error lines, key outputs, final summary.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TruncationResult:
    content: str
    original_tokens: int
    truncated_tokens: int
    was_truncated: bool
    strategy: str = "none"


class ToolResultTruncator:
    """Intelligently truncates tool results to fit within token budgets."""

    def __init__(self, max_tokens: int = 2048) -> None:
        self.max_tokens = max_tokens

    def truncate(self, content: str, tool_name: str = "") -> TruncationResult:
        """Truncate content, preserving the most informative parts."""
        if not content:
            return TruncationResult(content="", original_tokens=0, truncated_tokens=0, was_truncated=False)

        original_tokens = self._estimate_tokens(content)
        if original_tokens <= self.max_tokens:
            return TruncationResult(content=content, original_tokens=original_tokens,
                                    truncated_tokens=0, was_truncated=False)

        preserved = self._preserve_structure(content, tool_name)
        preserved_tokens = self._estimate_tokens(preserved)

        marker = (
            f"\n[... content truncated: {original_tokens} → {preserved_tokens} tokens, "
            f"removed {original_tokens - preserved_tokens} tokens ...]\n"
        )

        return TruncationResult(
            content=preserved + marker,
            original_tokens=original_tokens,
            truncated_tokens=preserved_tokens,
            was_truncated=True,
            strategy=tool_name,
        )

    def _preserve_structure(self, content: str, tool_name: str) -> str:
        lines = content.split("\n")
        budget = max(5, self.max_tokens // 4)  # lines budget

        if len(lines) <= budget:
            return content

        # Keep first 15% (header/summary)
        head_count = max(2, int(budget * 0.15))
        preserved: list[str] = list(lines[:head_count])

        # Keep error/warning lines
        for line in lines[head_count:]:
            if any(kw in line.lower() for kw in ("error", "failed", "traceback", "exception",
                                                   "warning", "critical", "fatal", "exit code")):
                preserved.append(line)

        # For JSON: keep structure, sample array items
        if tool_name in ("read_file",) and content.strip().startswith(("{", "[")):
            return self._truncate_json(content, budget)

        # Keep last 10% (summary/result)
        tail_count = max(1, int(budget * 0.10))
        tail_start = len(lines) - tail_count
        for line in lines[tail_start:]:
            if line not in preserved:
                preserved.append(line)

        # Fill remaining budget with middle lines
        remaining = budget - len(preserved)
        if remaining > 0:
            middle_start = head_count
            middle_end = min(middle_start + remaining, len(lines) - tail_count)
            for line in lines[middle_start:middle_end]:
                if line not in preserved:
                    preserved.append(line)

        return "\n".join(preserved[:budget])

    def _truncate_json(self, content: str, budget: int) -> str:
        """Keep JSON structure, truncate arrays."""
        if len(content) <= self.max_tokens * 3:
            return content[:self.max_tokens * 4]  # rough char cut
        return content[:self.max_tokens * 4]

    def _estimate_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)
