"""
ContextManager — unified entry point for context governance.

Coordinates: TokenCounter, WatermarkDetector, Truncator, Compactor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mini_core.context.counter import TokenCounter
from mini_core.context.watermark import WatermarkDetector, WatermarkLevel, WatermarkResult
from mini_core.context.truncator import ToolResultTruncator, TruncationResult
from mini_core.context.compactor import Compactor, CompactResult


@dataclass
class ContextHealth:
    level: WatermarkLevel
    usage_pct: float
    current_tokens: int
    max_tokens: int
    available_tokens: int
    truncation_enabled: bool
    compact_recommended: bool
    message_stats: dict[str, int]  # role → message count
    token_stats: dict[str, int]  # role → estimated tokens


class ContextManager:
    """Unified context governance for agent loops."""

    def __init__(self, max_context_tokens: int = 200000, llm=None, notes_mgr=None) -> None:
        self.counter = TokenCounter()
        self.watermark = WatermarkDetector(max_context_tokens=max_context_tokens)
        self.truncator = ToolResultTruncator()
        self.compactor = Compactor(llm=llm, notes_mgr=notes_mgr)
        self.enable_truncation = False
        self._current_messages: list[dict] = []

    def update_messages(self, messages: list[dict]) -> None:
        self._current_messages = list(messages)

    def check_health(self) -> ContextHealth:
        current_tokens = self.counter.count(self._current_messages)
        wr = self.watermark.check(current_tokens)

        if wr.level == WatermarkLevel.CRITICAL:
            self.enable_truncation = True

        # Compute message stats
        msg_stats: dict[str, int] = {}
        token_stats: dict[str, int] = {}
        for m in self._current_messages:
            role = m.get("role", "unknown")
            msg_stats[role] = msg_stats.get(role, 0) + 1
            content = str(m.get("content", ""))
            token_stats[role] = token_stats.get(role, 0) + self.counter.count_text(content)

        return ContextHealth(
            level=wr.level,
            usage_pct=wr.usage_pct,
            current_tokens=current_tokens,
            max_tokens=wr.max_tokens,
            available_tokens=wr.available_tokens,
            truncation_enabled=self.enable_truncation,
            compact_recommended=(wr.action == "compact"),
            message_stats=msg_stats,
            token_stats=token_stats,
        )

    def truncate_result(self, content: str, tool_name: str = "") -> TruncationResult:
        if not self.enable_truncation:
            return TruncationResult(content=content, original_tokens=len(content)//4,
                                    truncated_tokens=0, was_truncated=False)
        return self.truncator.truncate(content, tool_name=tool_name)

    async def compact(self, session_id: str = "", keep_recent: int = 3) -> CompactResult:
        result = await self.compactor.compact(
            self._current_messages, session_id=session_id, keep_recent=keep_recent,
        )
        if result.new_messages:
            self._current_messages = result.new_messages
            self.enable_truncation = False
        return result
