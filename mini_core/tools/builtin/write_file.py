"""
write_file tool — writes content to a file within the working directory.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from mini_core.tools.base import Tool, ToolResult


class WriteFileTool(Tool):
    name = "write_file"
    description = "Write content to a file at the given path (relative to the working directory). Creates parent directories if needed."
    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to the file to write, relative to the working directory.",
            },
            "content": {
                "type": "string",
                "description": "The text content to write to the file.",
            },
        },
        "required": ["path", "content"],
    }

    def __init__(self, workdir: str = ".") -> None:
        self._workdir = Path(workdir).resolve()

    async def execute(self, **kwargs: Any) -> ToolResult:
        rel_path = kwargs.get("path", "")
        content = kwargs.get("content", "")
        full_path = (self._workdir / rel_path).resolve()

        # Security: ensure the path is within workdir
        if not str(full_path).startswith(str(self._workdir)):
            return ToolResult(
                success=False,
                content=f"Error: Access denied — path '{rel_path}' is outside the working directory.",
                raw_content="",
                error="path outside workdir",
            )

        old_content: str | None = None
        if full_path.exists():
            try:
                old_content = full_path.read_text("utf-8")
            except Exception:
                old_content = "<binary or unreadable>"

        try:
            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.write_text(content, encoding="utf-8")
        except PermissionError:
            return ToolResult(
                success=False,
                content=f"Error: Permission denied writing to '{rel_path}'.",
                raw_content="",
                error="permission denied",
            )
        except Exception as exc:
            return ToolResult(
                success=False,
                content=f"Error writing '{rel_path}': {exc}",
                raw_content="",
                error=str(exc),
            )

        written_size = len(content.encode("utf-8"))
        return ToolResult(
            success=True,
            content=f"Successfully wrote {written_size} bytes to '{rel_path}'.",
            raw_content=content,
            metadata={
                "path": str(full_path),
                "bytes_written": written_size,
                "old_content": old_content,
            },
        )
