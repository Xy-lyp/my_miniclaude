"""SkillMatcher — matches user input to appropriate skills."""

from __future__ import annotations

from dataclasses import dataclass

from mini_core.skills.loader import Skill


@dataclass
class MatchResult:
    skill: Skill | None
    confidence: float = 0.0
    method: str = "none"  # "explicit" | "keyword" | "llm"


class SkillMatcher:
    def match(self, user_input: str, available_skills: list[Skill]) -> MatchResult:
        # 1. Explicit slash command
        if user_input.strip().startswith("/"):
            name = user_input.strip()[1:].split()[0]
            for skill in available_skills:
                if skill.name == name:
                    return MatchResult(skill=skill, confidence=1.0, method="explicit")
            return MatchResult(skill=None, confidence=0.0, method="explicit")

        # 2. Keyword matching
        best = None
        best_score = 0.0
        lower_input = user_input.lower()

        for skill in available_skills:
            score = 0.0
            desc_words = set(skill.description.lower().split())
            name_words = set(skill.name.lower().replace("-", " ").split())

            # Count keyword matches
            for word in name_words:
                if word in lower_input and len(word) > 2:
                    score += 0.3
            for word in desc_words:
                if word in lower_input and len(word) > 3:
                    score += 0.15

            if score > best_score:
                best_score = score
                best = skill

        if best and best_score > 0.2:
            return MatchResult(skill=best, confidence=min(best_score, 0.9), method="keyword")

        return MatchResult(skill=None, confidence=0.0, method="keyword")

    def get_triggered_skills(self, user_input: str, available_skills: list[Skill]) -> list[Skill]:
        """Return all skills that match (for TUI suggestions)."""
        triggered: list[Skill] = []
        lower = user_input.lower()
        for s in available_skills:
            words = set(s.name.replace("-", " ").split())
            if any(w in lower for w in words if len(w) > 2):
                triggered.append(s)
        return triggered
