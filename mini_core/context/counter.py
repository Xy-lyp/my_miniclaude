"""
TokenCounter — estimates token counts for messages and text.

Uses a simple character-based heuristic (4 chars ≈ 1 token) with
optional tiktoken support for greater accuracy.
"""

from __future__ import annotations


class TokenCounter:
    """Token counting with character-based fallback."""

    # Rough mapping: chars per token varies by language
    CHARS_PER_TOKEN = 4  # English average
    CHARS_PER_TOKEN_CN = 1.5  # Chinese

    def __init__(self, system_prompt_tokens: int = 0, tool_def_tokens: int = 0) -> None:
        self.system_prompt_tokens = system_prompt_tokens
        self.tool_definitions_tokens = tool_def_tokens

    def count(self, messages: list[dict]) -> int:
        """Estimate total tokens for a message list."""
        total = self.system_prompt_tokens + self.tool_definitions_tokens
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                total += self.count_text(content)
            # Tool calls
            for tc in msg.get("tool_calls", []):
                func = tc.get("function", {})
                total += self.count_text(func.get("name", ""))
                total += self.count_text(func.get("arguments", ""))
        return total

    def count_text(self, text: str) -> int:
        """Estimate tokens for a plain text string."""
        if not text:
            return 0
        # Count Chinese chars differently
        cn_chars = sum(1 for c in text if '一' <= c <= '鿿')
        other_chars = len(text) - cn_chars
        return int(cn_chars / self.CHARS_PER_TOKEN_CN + other_chars / self.CHARS_PER_TOKEN)

    def calibrate(self, estimated: int, api_usage: dict) -> int:
        """Calibrate estimate with actual API usage. Returns the calibrated count."""
        api_total = api_usage.get("prompt_tokens", estimated)
        return api_total
