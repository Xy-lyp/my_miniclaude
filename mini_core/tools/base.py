"""
Tool base class and ToolResult type.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolResult:
    """Result from a single tool execution."""

    success: bool
    content: str  # Returned to the LLM (may be truncated)
    raw_content: str  # Full content (for event records)
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class Tool(ABC):
    """Abstract base class for all tools."""

    name: str
    description: str
    parameters: dict[str, Any]

    @abstractmethod
    async def execute(self, **kwargs: Any) -> ToolResult:
        """Execute the tool and return a result."""
        ...

    def to_openai_definition(self) -> dict[str, Any]:
        """Return the OpenAI-compatible function definition for this tool."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }
