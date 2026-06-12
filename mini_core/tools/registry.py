"""
ToolRegistry — register, lookup, and validate tools.
"""

from __future__ import annotations

from typing import Any

from mini_core.tools.base import Tool


class ToolRegistry:
    """Holds all available tools and provides lookup + OpenAI-format definitions."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Register a tool instance."""
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        """Get a tool by name, or None if not found."""
        return self._tools.get(name)

    def has(self, name: str) -> bool:
        """Check whether a tool with the given name is registered."""
        return name in self._tools

    def get_definitions(self) -> list[dict[str, Any]]:
        """Return all tools as OpenAI-compatible function definitions."""
        return [t.to_openai_definition() for t in self._tools.values()]

    def validate(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        """Validate arguments against the tool's JSON Schema parameters.

        Returns a dict with keys:
          - valid: bool
          - errors: list[str]  (if any)
        """
        tool = self.get(name)
        if tool is None:
            return {"valid": False, "errors": [f"Tool not found: {name}"]}

        errors: list[str] = []
        schema = tool.parameters

        # Validate required fields
        required: list[str] = schema.get("required", [])
        for req_field in required:
            if req_field not in args:
                errors.append(f"Missing required parameter: '{req_field}'")

        # Validate types for each property
        properties: dict[str, Any] = schema.get("properties", {})
        for prop_name, prop_schema in properties.items():
            if prop_name not in args:
                continue  # optional param

            value = args[prop_name]
            expected_type = prop_schema.get("type")

            if expected_type == "string" and not isinstance(value, str):
                errors.append(f"'{prop_name}' must be a string, got {type(value).__name__}")
            elif expected_type == "integer" and not isinstance(value, int):
                errors.append(f"'{prop_name}' must be an integer, got {type(value).__name__}")
            elif expected_type == "number" and not isinstance(value, (int, float)):
                errors.append(f"'{prop_name}' must be a number, got {type(value).__name__}")
            elif expected_type == "boolean" and not isinstance(value, bool):
                errors.append(f"'{prop_name}' must be a boolean, got {type(value).__name__}")
            elif expected_type == "array" and not isinstance(value, list):
                errors.append(f"'{prop_name}' must be an array, got {type(value).__name__}")
            elif expected_type == "object" and not isinstance(value, dict):
                errors.append(f"'{prop_name}' must be an object, got {type(value).__name__}")

        return {"valid": len(errors) == 0, "errors": errors}
