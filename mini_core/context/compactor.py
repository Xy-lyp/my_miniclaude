"""
Compactor — LLM-powered conversation compression.

Compresses old conversation history into a structured summary,
keeping recent messages intact.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from mini_core.memory.notes import NotesManager


COMPACT_PROMPT = """You are a conversation summarizer. Compress the following conversation history into a structured JSON summary.

Requirements:
1. Preserve all key decisions and their reasons
2. Preserve important tool call results (file paths, command outputs, errors)
3. Preserve user preferences and explicit instructions
4. Discard redundant information already superseded by later actions
5. Note the source round for each item

Output strict JSON only (no extra text):
{
  "summary": "Concise text summary...",
  "key_decisions": [
    {"decision": "Use Flask over FastAPI", "reason": "User explicitly requested"}
  ],
  "important_outputs": [
    {"file": "/path/to/file", "description": "Created configuration file"}
  ],
  "user_preferences": [
    {"preference": "Use black for code formatting", "source_round": 3}
  ],
  "current_state": {
    "working_directory": "/tmp/myapp",
    "files_created": ["app.py", "requirements.txt"],
    "last_action": "Successfully ran Flask app on port 5000"
  },
  "discarded_info": "Detailed code formatting discussions (already applied)"
}"""


@dataclass
class CompactResult:
    summary: str = ""
    key_decisions: list[dict] = field(default_factory=list)
    important_outputs: list[dict] = field(default_factory=list)
    user_preferences: list[dict] = field(default_factory=list)
    current_state: dict = field(default_factory=dict)
    discarded_info: str = ""
    tokens_before: int = 0
    tokens_after: int = 0
    reduction_pct: float = 0.0
    new_messages: list[dict] = field(default_factory=list)


class Compactor:
    """Compress conversation history using LLM summarization."""

    def __init__(self, llm=None, notes_mgr: NotesManager | None = None) -> None:
        self._llm = llm
        self._notes_mgr = notes_mgr

    async def compact(self, messages: list[dict], session_id: str = "",
                       keep_recent: int = 3) -> CompactResult:
        """Compress old messages, keeping recent ones intact."""
        if not self._llm:
            return CompactResult(tokens_before=0, tokens_after=0, reduction_pct=0.0)

        # Estimate tokens before (simple estimate)
        tokens_before = sum(len(str(m)) // 4 for m in messages)

        # Split: old messages to compress, recent ones to keep
        recent_count = keep_recent * 2  # N user+assistant pairs
        old_messages = messages[:-recent_count] if len(messages) > recent_count else messages[:len(messages)//2]
        recent_messages = messages[-recent_count:] if len(messages) > recent_count else []

        if len(old_messages) < 2:
            return CompactResult(tokens_before=tokens_before, tokens_after=tokens_before,
                                 reduction_pct=0.0, new_messages=messages)

        # Build conversation text for the LLM
        convo_text = ""
        for m in old_messages:
            role = m.get("role", "?")
            content = str(m.get("content", ""))[:500]
            convo_text += f"[{role}]: {content}\n"

        # Call LLM to generate summary
        result = CompactResult(tokens_before=tokens_before)
        try:
            resp = await self._llm.chat(
                [{"role": "user", "content": f"{COMPACT_PROMPT}\n\nConversation:\n{convo_text}"}],
                tools=None,
            )
            text = (resp.content or "").strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1]
                if text.rstrip().endswith("```"):
                    text = text.rsplit("```", 1)[0]
            parsed = json.loads(text)
            result = CompactResult(
                summary=parsed.get("summary", ""),
                key_decisions=parsed.get("key_decisions", []),
                important_outputs=parsed.get("important_outputs", []),
                user_preferences=parsed.get("user_preferences", []),
                current_state=parsed.get("current_state", {}),
                discarded_info=parsed.get("discarded_info", ""),
                tokens_before=tokens_before,
            )
        except Exception:
            # Fallback: just use a simple summary
            result = CompactResult(
                summary=f"Compressed {len(old_messages)} messages.",
                tokens_before=tokens_before,
            )

        # Build new message list
        summary_text = self._build_compact_system_message(result)
        new_messages = [{"role": "system", "content": summary_text}]
        new_messages.extend(recent_messages)

        tokens_after = sum(len(str(m)) // 4 for m in new_messages)
        reduction_pct = round((1 - tokens_after / max(tokens_before, 1)) * 100, 1)

        result.tokens_after = tokens_after
        result.reduction_pct = reduction_pct
        result.new_messages = new_messages

        # Auto-create notes from key findings
        if self._notes_mgr and session_id:
            for d in result.key_decisions:
                self._notes_mgr.upsert(session_id, title=d["decision"],
                                       content=d.get("reason", ""), note_type="decision",
                                       source="compact", importance=7)
            for p in result.user_preferences:
                self._notes_mgr.upsert(session_id, title=p["preference"],
                                       content=p.get("preference", ""), note_type="user_preference",
                                       source="compact", importance=6)
            for o in result.important_outputs:
                self._notes_mgr.upsert(session_id, title=o.get("file", o.get("description", "")),
                                       content=o.get("description", ""), note_type="project_context",
                                       source="compact", importance=5)

        return result

    def _build_compact_system_message(self, cr: CompactResult) -> str:
        parts = [
            "## Conversation History Summary (Compacted)",
            f"\n{cr.summary}",
        ]
        if cr.key_decisions:
            parts.append("\n### Key Decisions Made")
            for d in cr.key_decisions:
                parts.append(f"- {d['decision']}: {d.get('reason', '')}")
        if cr.important_outputs:
            parts.append("\n### Important Outputs")
            for o in cr.important_outputs:
                parts.append(f"- {o.get('file', '?')}: {o.get('description', '')}")
        if cr.user_preferences:
            parts.append("\n### User Preferences")
            for p in cr.user_preferences:
                parts.append(f"- {p['preference']}")
        if cr.current_state:
            parts.append(f"\n### Current State\n{json.dumps(cr.current_state, indent=2)}")
        return "\n".join(parts)
