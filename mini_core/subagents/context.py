"""SubAgent context builder — extracts key context from parent conversation."""

from __future__ import annotations

import json
from typing import Any


def build_subagent_context(parent_messages: list[dict], task: str,
                            working_directory: str = ".") -> list[dict]:
    """Build a minimal context for a sub-agent from parent messages."""
    files_created = _extract_created_files(parent_messages)
    key_decisions = _extract_key_decisions(parent_messages)
    parent_goal = _extract_goal(parent_messages)

    context = {
        "parent_goal": parent_goal,
        "working_directory": working_directory,
        "files_created": files_created,
        "key_decisions": key_decisions,
        "your_task": task,
    }

    return [
        {
            "role": "system",
            "content": f"""You are a sub-agent working on a specific task.

## Context from Parent Agent
{json.dumps(context, indent=2, ensure_ascii=False)}

## Instructions
- Focus ONLY on your assigned task
- Report your results clearly when done
- Do not deviate from your task
""",
        },
        {"role": "user", "content": task},
    ]


def _extract_goal(messages: list[dict]) -> str:
    for m in messages:
        if m.get("role") == "user":
            content = str(m.get("content", ""))[:100]
            return content
    return "unknown"


def _extract_created_files(messages: list[dict]) -> list[str]:
    files: list[str] = []
    for m in messages:
        content = str(m.get("content", ""))
        import re
        for match in re.finditer(r'(?:created|wrote?|saved)\s+(?:to\s+)?["\']?([^\s"\']+)', content, re.I):
            files.append(match.group(1))
    return list(set(files))[-10:]  # Last 10 unique


def _extract_key_decisions(messages: list[dict]) -> list[str]:
    decisions: list[str] = []
    for m in messages:
        if m.get("role") == "assistant":
            content = str(m.get("content", ""))
            if "decision" in content.lower() or "decided" in content.lower():
                decisions.append(content[:200])
    return decisions[-5:]  # Last 5
