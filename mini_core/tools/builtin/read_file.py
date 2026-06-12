"""
read_file tool — reads file content relative to the working directory.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from mini_core.tools.base import Tool, ToolResult

MAX_FILE_SIZE = 1 * 1024 * 1024  # 1 MB


class ReadFileTool(Tool):
    name = "read_file"
    description = "Read the contents of a file at the given path (relative to the working directory)."
    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to the file, relative to the working directory.",
            },
        },
        "required": ["path"],
    }

    def __init__(self, workdir: str = ".") -> None:
        self._workdir = Path(workdir).resolve()

    async def execute(self, **kwargs: Any) -> ToolResult:
        rel_path = kwargs.get("path", "")
        full_path = (self._workdir / rel_path).resolve()

        # Security: ensure the path is within workdir
        if not str(full_path).startswith(str(self._workdir)):
            return ToolResult(
                success=False,
                content=f"Error: Access denied — path '{rel_path}' is outside the working directory.",
                raw_content="",
                error="path outside workdir",
            )

        if not full_path.exists():
            return ToolResult(
                success=False,
                content=f"Error: File not found: {rel_path}",
                raw_content="",
                error="file not found",
            )

        if full_path.is_dir():
            return ToolResult(
                success=False,
                content=f"Error: '{rel_path}' is a directory, not a file.",
                raw_content="",
                error="is a directory",
            )

        # Detect binary files by trying to read as UTF-8
        try:
            file_size = full_path.stat().st_size
            if file_size > MAX_FILE_SIZE:
                # Read only first 1MB
                raw = full_path.read_bytes()[:MAX_FILE_SIZE]
                try:
                    content = raw.decode("utf-8")
                except UnicodeDecodeError:
                    return ToolResult(
                        success=False,
                        content=f"Error: File '{rel_path}' appears to be binary (cannot decode as UTF-8). Size: {file_size} bytes.",
                        raw_content="",
                        error="binary file",
                        metadata={"file_size": file_size},
                    )
                return ToolResult(
                    success=True,
                    content=f"File is {file_size} bytes — showing first 1MB:\n\n{content}\n\n[TRUNCATED — {file_size - MAX_FILE_SIZE} bytes omitted]",
                    raw_content=raw.decode("utf-8", errors="replace"),
                    metadata={"file_size": file_size, "truncated": True},
                )

            # Read entire file
            try:
                content = full_path.read_text("utf-8")
            except UnicodeDecodeError:
                return ToolResult(
                    success=False,
                    content=f"Error: File '{rel_path}' appears to be binary (cannot decode as UTF-8). Size: {file_size} bytes.",
                    raw_content="",
                    error="binary file",
                    metadata={"file_size": file_size},
                )

            return ToolResult(
                success=True,
                content=content,
                raw_content=content,
                metadata={"file_size": file_size, "truncated": False},
            )

        except PermissionError:
            return ToolResult(
                success=False,
                content=f"Error: Permission denied reading '{rel_path}'.",
                raw_content="",
                error="permission denied",
            )
        except Exception as exc:
            return ToolResult(
                success=False,
                content=f"Error reading '{rel_path}': {exc}",
                raw_content="",
                error=str(exc),
            )
