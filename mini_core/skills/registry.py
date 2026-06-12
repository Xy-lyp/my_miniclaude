"""SkillRegistry — manages loaded skills and activation state."""

from __future__ import annotations

from mini_core.skills.loader import Skill, SkillLoader


class SkillRegistry:
    def __init__(self) -> None:
        self._loader = SkillLoader()
        self._skills: dict[str, Skill] = {}
        self._active: set[str] = set()
        self.reload()

    def reload(self) -> None:
        self._skills.clear()
        for skill in self._loader.load_all():
            self._skills[skill.name] = skill

    def list_all(self) -> list[Skill]:
        return list(self._skills.values())

    def get(self, name: str) -> Skill | None:
        return self._skills.get(name)

    def activate(self, name: str) -> bool:
        if name in self._skills:
            self._active.add(name)
            return True
        return False

    def deactivate(self, name: str) -> None:
        self._active.discard(name)

    @property
    def active_skills(self) -> list[Skill]:
        return [self._skills[n] for n in self._active if n in self._skills]

    @property
    def active_names(self) -> list[str]:
        return list(self._active)

    def get_active_prompt(self) -> str:
        if not self._active:
            return ""
        parts: list[str] = []
        for skill in self.active_skills:
            parts.append(f"\n## Active Skill: {skill.name}\n{skill.prompt}")
        return "\n".join(parts)

    def get_active_tool_whitelist(self) -> list[str] | None:
        """Return combined allowed_tools from all active skills, or None for no restriction."""
        if not self._active:
            return None
        allowed: set[str] = set()
        for skill in self.active_skills:
            allowed.update(skill.allowed_tools)
        return list(allowed) if allowed else None
