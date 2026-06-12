"""
Failure classifier + retry strategy for tool execution errors.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any

from mini_core.tools.base import Tool, ToolResult


class ErrorCategory(Enum):
    RETRYABLE = "retryable"
    RETRYABLE_WITH_BACKOFF = "retryable_with_backoff"
    FIXABLE_BY_MODEL = "fixable_by_model"
    PERMISSION = "permission"
    FATAL = "fatal"


@dataclass
class ClassifiedError:
    category: ErrorCategory
    original_error: Exception
    message: str
    should_retry: bool = False
    retry_delay: float = 0.0
    max_retries: int = 0


class ErrorClassifier:
    def classify(self, tool_name: str, error: Exception, result: ToolResult | None = None) -> ClassifiedError:
        msg = str(error).lower()

        # Permission errors
        if any(kw in msg for kw in ("permission denied", "access denied", "not allowed",
                                      "path traversal blocked", "forbidden")):
            return ClassifiedError(ErrorCategory.PERMISSION, error, str(error))

        # Network / timeout errors
        if any(kw in msg for kw in ("timeout", "timed out", "connection refused",
                                      "connection reset", "network")):
            return ClassifiedError(
                ErrorCategory.RETRYABLE, error, str(error),
                should_retry=True, max_retries=2,
            )

        # Rate limit
        if any(kw in msg for kw in ("429", "rate limit", "too many requests")):
            return ClassifiedError(
                ErrorCategory.RETRYABLE_WITH_BACKOFF, error, str(error),
                should_retry=True, retry_delay=1.0, max_retries=3,
            )

        # File in use
        if any(kw in msg for kw in ("being used", "in use", "locked")):
            return ClassifiedError(
                ErrorCategory.RETRYABLE_WITH_BACKOFF, error, str(error),
                should_retry=True, retry_delay=0.5, max_retries=3,
            )

        # Model-fixable errors (parameter issues)
        if any(kw in msg for kw in ("file not found", "not found", "invalid parameter",
                                      "syntax error", "invalid argument", "command not found")):
            return ClassifiedError(ErrorCategory.FIXABLE_BY_MODEL, error, str(error))

        # Fatal
        if any(kw in msg for kw in ("disk full", "out of memory", "no space",
                                      "sigkill", "killed", "bus error")):
            return ClassifiedError(ErrorCategory.FATAL, error, str(error))

        return ClassifiedError(ErrorCategory.FATAL, error, str(error))


class RetryStrategy:
    def __init__(self, classifier: ErrorClassifier | None = None) -> None:
        self._classifier = classifier or ErrorClassifier()
        # Track non-idempotent operations
        self._non_idempotent_tools = {"write_file", "run_shell"}

    async def execute_with_retry(self, tool: Tool, args: dict,
                                  max_retries: int = 3) -> ToolResult:
        last_result: ToolResult | None = None
        delay = 0.0

        for attempt in range(max_retries + 1):
            try:
                result = await tool.execute(**args)
                return result
            except Exception as exc:
                classified = self._classifier.classify(tool.name, exc)

                if not classified.should_retry or attempt >= classified.max_retries:
                    if classified.category == ErrorCategory.FIXABLE_BY_MODEL:
                        return ToolResult(success=False, content=classified.message,
                                          raw_content=str(exc), error=classified.message)
                    raise

                # Non-idempotent: skip retry unless safe
                if tool.name in self._non_idempotent_tools and attempt > 0:
                    return ToolResult(success=False, content=f"Non-idempotent tool failed: {classified.message}",
                                      raw_content=str(exc), error=classified.message)

                # Wait with backoff
                delay = classified.retry_delay * (2 ** attempt) if classified.retry_delay > 0 else 0.5 * (2 ** attempt)
                await asyncio.sleep(delay)

        return ToolResult(success=False, content="Max retries exceeded",
                          raw_content=str(last_result) if last_result else "", error="max retries exceeded")
