"""
LLM Provider — abstract base class + OpenAI-compatible implementation.

Reads configuration from environment variables:
  LLM_API_KEY   — API key (required)
  LLM_BASE_URL  — API base URL (default: https://api.openai.com/v1)
  LLM_MODEL     — model name (default: deepseek-v4-flash)
"""

from __future__ import annotations

import json
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("mini-core.llm")

# ── Response types ────────────────────────────────────────────────────────────


@dataclass
class TokenUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMResponse:
    content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str = "stop"
    usage: TokenUsage = field(default_factory=TokenUsage)


# ── Abstract provider ─────────────────────────────────────────────────────────


class LLMProvider(ABC):
    """Abstract interface for LLM backends."""

    @abstractmethod
    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        """Send messages to the LLM and return a structured response."""
        ...


# ── OpenAI-compatible implementation ──────────────────────────────────────────


class OpenAICompatibleProvider(LLMProvider):
    """LLM provider for any OpenAI-compatible API.

    Configuration (env vars):
      LLM_API_KEY  — required
      LLM_BASE_URL — defaults to https://api.openai.com/v1
      LLM_MODEL    — defaults to deepseek-v4-flash
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
    ) -> None:
        self._api_key = api_key or os.environ.get("LLM_API_KEY", "")
        self._model = model or os.environ.get("LLM_MODEL", "deepseek-v4-pro")
        # Auto-detect base URL from model name
        default_base = "https://api.openai.com/v1"
        if "deepseek" in self._model.lower():
            default_base = "https://api.deepseek.com/v1"
        elif "claude" in self._model.lower() or "anthropic" in self._model.lower():
            default_base = "https://api.anthropic.com/v1"
        self._base_url = (base_url or os.environ.get("LLM_BASE_URL", default_base)).rstrip("/")

        if not self._api_key:
            logger.warning("LLM_API_KEY not set — LLM calls will fail")

    @property
    def model(self) -> str:
        return self._model

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        """Send a chat completion request to the OpenAI-compatible endpoint."""
        import aiohttp

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}",
        }

        body: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
        }

        if tools:
            body["tools"] = tools

        url = f"{self._base_url}/chat/completions"

        logger.debug("LLM request: %d messages, %d tools", len(messages), len(tools or []))

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=body, headers=headers, timeout=aiohttp.ClientTimeout(total=120)) as resp:
                    if resp.status != 200:
                        error_text = await resp.text()
                        logger.error("LLM API error %d: %s", resp.status, error_text)
                        raise LLMError(f"API returned {resp.status}: {error_text[:500]}")

                    data = await resp.json()
        except Exception as exc:
            if isinstance(exc, LLMError):
                raise
            logger.error("LLM request failed: %s", exc)
            raise LLMError(f"LLM request failed: {exc}") from exc

        return self._parse_response(data)

    def _parse_response(self, data: dict[str, Any]) -> LLMResponse:
        """Parse an OpenAI-format chat completion response."""
        choice = data.get("choices", [{}])[0]
        message = choice.get("message", {})

        finish_reason = choice.get("finish_reason", "stop")
        content = message.get("content")

        # Parse tool calls
        tool_calls: list[ToolCall] = []
        raw_tool_calls = message.get("tool_calls", [])
        for tc in raw_tool_calls:
            tc_id = tc.get("id", "")
            func = tc.get("function", {})
            name = func.get("name", "")
            args_str = func.get("arguments", "{}")
            try:
                arguments = json.loads(args_str) if isinstance(args_str, str) else args_str
            except json.JSONDecodeError:
                arguments = {}

            tool_calls.append(ToolCall(id=tc_id, name=name, arguments=arguments))

        # Parse usage
        usage_raw = data.get("usage", {})
        usage = TokenUsage(
            prompt_tokens=usage_raw.get("prompt_tokens", 0),
            completion_tokens=usage_raw.get("completion_tokens", 0),
            total_tokens=usage_raw.get("total_tokens", 0),
        )

        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            usage=usage,
        )


# ── Errors ────────────────────────────────────────────────────────────────────


class LLMError(Exception):
    """Raised when the LLM API call fails."""
    pass
