"""
ParameterValidator — three-layer validation: Schema + Semantics + Path Safety.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class SemanticResult:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class PathResult:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class ValidationResult:
    is_valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    sanitized_args: dict[str, Any] = field(default_factory=dict)


# ── Dangerous command patterns ───────────────────────────────────────────────

DANGEROUS_COMMAND_PATTERNS: list[tuple[str, str, str]] = [
    (r"\brm\s+(-rf?|--recursive)\s+/", "CRITICAL", "rm -rf /"),
    (r">\s*/dev/sda", "CRITICAL", "write to block device"),
    (r"\bdd\s+if=", "CRITICAL", "dd disk operation"),
    (r"\bmkfs\.", "CRITICAL", "format filesystem"),
    (r"\bsudo\b", "HIGH", "sudo privilege escalation"),
    (r"\bchmod\s+777", "HIGH", "chmod 777"),
    (r"\bcurl\b.*\|\s*(ba)?sh", "HIGH", "curl pipe bash"),
    (r"\bwget\b.*\|\s*(ba)?sh", "HIGH", "wget pipe bash"),
    (r"\bcat\s+/etc/(passwd|shadow)", "MEDIUM", "read sensitive system file"),
    (r"\benv\b", "LOW", "print environment variables"),
    (r":\(\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;\s*:", "CRITICAL", "fork bomb"),
]

FORBIDDEN_DIRS = [
    "/etc", "/sys", "/proc", "/dev", "/boot", "/root",
    "C:\\Windows", "C:\\Windows\\System32", "C:\\Program Files",
]


class ParameterValidator:
    def __init__(self, workdir: str = ".") -> None:
        self._workdir = Path(workdir).resolve()

    def validate(self, tool_name: str, args: dict) -> ValidationResult:
        errors: list[str] = []
        warnings: list[str] = []

        # 1. Schema check
        errors.extend(self._validate_schema(tool_name, args))

        # 2. Semantic check
        sem = self._validate_semantics(tool_name, args)
        errors.extend(sem.errors)
        warnings.extend(sem.warnings)

        # 3. Path safety
        if tool_name in ("read_file", "write_file", "run_shell"):
            path_res = self._validate_path_safety(tool_name, args)
            errors.extend(path_res.errors)
            warnings.extend(path_res.warnings)

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors, warnings=warnings,
            sanitized_args=self._sanitize_args(args),
        )

    def _validate_schema(self, tool_name: str, args: dict) -> list[str]:
        errors: list[str] = []
        if tool_name == "read_file":
            if "path" not in args or not isinstance(args.get("path"), str):
                errors.append("read_file: 'path' is required and must be a string")
        elif tool_name == "write_file":
            if "path" not in args or not isinstance(args.get("path"), str):
                errors.append("write_file: 'path' is required and must be a string")
            if "content" not in args:
                errors.append("write_file: 'content' is required")
        elif tool_name == "run_shell":
            cmd = args.get("command", "")
            if not cmd or not isinstance(cmd, str):
                errors.append("run_shell: 'command' is required and must be a string")
            if len(cmd) > 10000:
                errors.append("run_shell: command too long (max 10000 chars)")
        return errors

    def _validate_semantics(self, tool_name: str, args: dict) -> SemanticResult:
        errors: list[str] = []
        warnings: list[str] = []

        if tool_name == "write_file":
            content = args.get("content", "")
            if len(content) > 10 * 1024 * 1024:  # 10MB
                warnings.append("File content exceeds 10MB")
            if len(content) > 50 * 1024 * 1024:  # 50MB
                errors.append("File content exceeds maximum size of 50MB")

        if tool_name == "run_shell":
            cmd = args.get("command", "")
            for pattern, level, desc in DANGEROUS_COMMAND_PATTERNS:
                if re.search(pattern, cmd):
                    if level == "CRITICAL":
                        errors.append(f"CRITICAL: {desc}")
                    elif level == "HIGH":
                        errors.append(f"Dangerous command: {desc}")
                    elif level == "MEDIUM":
                        warnings.append(f"Warning: {desc}")

        return SemanticResult(errors=errors, warnings=warnings)

    def _validate_path_safety(self, tool_name: str, args: dict) -> PathResult:
        errors: list[str] = []
        warnings: list[str] = []

        path_key = "path" if tool_name in ("read_file", "write_file") else None
        if path_key and path_key in args:
            raw = args[path_key]
            if "\x00" in raw:
                errors.append("Path contains null byte — rejected")
                return PathResult(errors=errors, warnings=warnings)

            try:
                full = (self._workdir / raw).resolve()
            except Exception:
                errors.append(f"Invalid path: {raw}")
                return PathResult(errors=errors, warnings=warnings)

            rel = str(full)
            workdir_str = str(self._workdir)

            # Path traversal check
            if not rel.startswith(workdir_str) and not rel.startswith(workdir_str + os.sep):
                errors.append(f"Path traversal blocked: {raw} → {rel}")
                return PathResult(errors=errors, warnings=warnings)

            # Forbidden directories
            for forbidden in FORBIDDEN_DIRS:
                if rel.startswith(forbidden):
                    errors.append(f"Access to forbidden directory blocked: {forbidden}")
                    return PathResult(errors=errors, warnings=warnings)

            # Hidden file / directory
            parts = raw.replace("\\", "/").split("/")
            for p in parts:
                if p.startswith(".") and p not in (".", ".."):
                    warnings.append(f"Hidden file/directory: {p}")

            # For write_file: protect key directories
            if tool_name == "write_file" and full.exists():
                for protected in [".git", "node_modules", ".venv", "__pycache__"]:
                    if protected in parts:
                        errors.append(f"Writing to protected directory: {protected}")
                        return PathResult(errors=errors, warnings=warnings)

        return PathResult(errors=errors, warnings=warnings)

    def _sanitize_args(self, args: dict) -> dict:
        """Remove null bytes and trim whitespace from string args."""
        clean = {}
        for k, v in args.items():
            if isinstance(v, str):
                clean[k] = v.replace("\x00", "").strip()
            else:
                clean[k] = v
        return clean
