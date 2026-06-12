"""
WatermarkDetector — monitors context usage and emits level-based alerts.

Levels: NORMAL (<70%) → WARNING (70-85%) → HIGH (85-95%) → CRITICAL (>95%)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class WatermarkLevel(Enum):
    NORMAL = "normal"
    WARNING = "warning"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class WatermarkResult:
    level: WatermarkLevel
    current_tokens: int
    max_tokens: int
    usage_pct: float
    available_tokens: int
    action: str  # "none" | "notify" | "truncate_results" | "compact"


class WatermarkDetector:
    """Checks token usage against thresholds and recommends actions."""

    def __init__(self, max_context_tokens: int = 200000, reserved_tokens: int = 4096,
                 warning_pct: float = 0.70, high_pct: float = 0.85, critical_pct: float = 0.95) -> None:
        self.max_tokens = max_context_tokens
        self.reserved_tokens = reserved_tokens
        self.warning_pct = warning_pct
        self.high_pct = high_pct
        self.critical_pct = critical_pct

    def check(self, current_tokens: int) -> WatermarkResult:
        usage_pct = current_tokens / self.max_tokens if self.max_tokens > 0 else 0
        available = max(0, self.max_tokens - current_tokens - self.reserved_tokens)

        if usage_pct < self.warning_pct:
            level = WatermarkLevel.NORMAL
        elif usage_pct < self.high_pct:
            level = WatermarkLevel.WARNING
        elif usage_pct < self.critical_pct:
            level = WatermarkLevel.HIGH
        else:
            level = WatermarkLevel.CRITICAL

        return WatermarkResult(
            level=level,
            current_tokens=current_tokens,
            max_tokens=self.max_tokens,
            usage_pct=round(usage_pct * 100, 1),
            available_tokens=available,
            action=self._get_action(level),
        )

    def _get_action(self, level: WatermarkLevel) -> str:
        return {
            WatermarkLevel.NORMAL: "none",
            WatermarkLevel.WARNING: "notify",
            WatermarkLevel.HIGH: "truncate_results",
            WatermarkLevel.CRITICAL: "compact",
        }[level]
