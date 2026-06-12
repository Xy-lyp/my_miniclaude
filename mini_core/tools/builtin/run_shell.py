"""
run_shell tool — executes a shell command within the working directory.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from mini_core.tools.base import Tool, ToolResult

# Commands that are outright blocked
BLOCKED_COMMANDS = [
    "rm -rf /",
    "mkfs.",
    "dd if=",
    ":(){ :|:& };:",  # fork bomb
]


class RunShellTool(Tool):
    name = "run_shell"
    description = "Execute a shell command in the working directory. Returns stdout, stderr, and exit code."
    parameters = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "The shell command to execute.",
            },
            "timeout": {
                "type": "integer",
                "description": "Timeout in seconds (default: 30, max: 120).",
                "default": 30,
            },
        },
        "required": ["command"],
    }

    def __init__(self, workdir: str = ".") -> None:
        self._workdir = Path(workdir).resolve()

    async def execute(self, **kwargs: Any) -> ToolResult:
        command = kwargs.get("command", "")
        timeout = kwargs.get("timeout", 30)

        # Sanity checks
        if not command or not command.strip():
            return ToolResult(
                success=False,
                content="Error: Empty command.",
                raw_content="",
                error="empty command",
            )

        # Check against blocked commands
        for blocked in BLOCKED_COMMANDS:
            if blocked in command:
                return ToolResult(
                    success=False,
                    content=f"Error: Command contains blocked pattern: '{blocked}'",
                    raw_content="",
                    error="blocked command",
                )

        # Cap timeout
        timeout = max(1, min(timeout, 120))

        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self._workdir),
            )

            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(), timeout=timeout
            )

            stdout = stdout_bytes.decode("utf-8", errors="replace")
            stderr = stderr_bytes.decode("utf-8", errors="replace")
            exit_code = proc.returncode if proc.returncode is not None else -1

            # Build a structured output for the LLM
            parts: list[str] = []
            if stdout:
                parts.append(f"STDOUT:\n{stdout}")
            if stderr:
                parts.append(f"STDERR:\n{stderr}")
            parts.append(f"EXIT_CODE: {exit_code}")

            output = "\n\n".join(parts) if parts else "(no output)"

            return ToolResult(
                success=exit_code == 0,
                content=output,
                raw_content=output,
                metadata={
                    "exit_code": exit_code,
                    "stdout_length": len(stdout),
                    "stderr_length": len(stderr),
                    "timeout": timeout,
                },
            )

        except asyncio.TimeoutError:
            return ToolResult(
                success=False,
                content=f"Error: Command timed out after {timeout}s.",
                raw_content="",
                error="timeout",
                metadata={"timeout": timeout},
            )
        except Exception as exc:
            return ToolResult(
                success=False,
                content=f"Error executing command: {exc}",
                raw_content="",
                error=str(exc),
            )
