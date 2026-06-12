"""
SkillLoader — parses SKILL.md files and builds Skill objects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Skill:
    name: str
    description: str
    version: str = "1.0.0"
    prompt: str = ""
    allowed_tools: list[str] = field(default_factory=list)
    model: str | None = None
    requires_approval: bool = False
    source_path: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class SkillLoader:
    """Loads Skill definitions from SKILL.md files."""

    def __init__(self) -> None:
        self._search_paths: list[Path] = []

    def add_search_path(self, path: Path) -> None:
        self._search_paths.append(path)

    def _default_search_paths(self) -> list[Path]:
        paths: list[Path] = []
        # Project skills
        paths.append(Path("skills"))
        paths.append(Path(".claude/skills"))
        # Home directory
        home = Path.home() / ".kama" / "skills"
        if home.exists():
            paths.append(home)
        return paths

    def load(self, skill_dir: Path) -> Skill | None:
        skill_file = skill_dir / "SKILL.md"
        if not skill_file.exists():
            return None

        text = skill_file.read_text(encoding="utf-8")
        frontmatter, body = self._parse_frontmatter(text)

        if "name" not in frontmatter:
            return None

        return Skill(
            name=frontmatter["name"],
            description=frontmatter.get("description", ""),
            version=frontmatter.get("version", "1.0.0"),
            prompt=body.strip(),
            allowed_tools=frontmatter.get("allowed_tools", []),
            model=frontmatter.get("model"),
            requires_approval=frontmatter.get("requires_approval", False),
            source_path=str(skill_file),
            metadata={k: v for k, v in frontmatter.items()
                      if k not in ("name", "description", "version", "allowed_tools", "model", "requires_approval")},
        )

    def load_all(self) -> list[Skill]:
        skills: list[Skill] = []
        seen: set[str] = set()

        search_paths = self._default_search_paths() + self._search_paths
        for base in search_paths:
            if not base.exists():
                continue
            for skill_dir in base.iterdir():
                if skill_dir.is_dir():
                    skill = self.load(skill_dir)
                    if skill and skill.name not in seen:
                        skills.append(skill)
                        seen.add(skill.name)
        return skills

    def _parse_frontmatter(self, text: str) -> tuple[dict[str, Any], str]:
        """Parse YAML frontmatter from markdown. Returns (metadata, body)."""
        lines = text.split("\n")
        if not lines or lines[0].strip() != "---":
            return {}, text

        end_idx = -1
        for i in range(1, len(lines)):
            if lines[i].strip() == "---":
                end_idx = i
                break
        if end_idx == -1:
            return {}, text

        frontmatter_lines = lines[1:end_idx]
        body = "\n".join(lines[end_idx + 1:])

        metadata: dict[str, Any] = {}
        current_key: str | None = None
        current_list: list[str] = []

        for line in frontmatter_lines:
            stripped = line.strip()
            if not stripped:
                continue

            # Check if this is a new key: value or key:
            if ":" in stripped and not stripped.startswith("- "):
                # Save previous list
                if current_key is not None and current_list:
                    metadata[current_key] = current_list
                    current_list = []
                    current_key = None

                # Check if it's "key: value" or "key:"
                if ": " in stripped or stripped.endswith(":"):
                    key_part, _, value_part = stripped.partition(":")
                    key = key_part.strip()
                    value = value_part.strip()
                    if value:
                        # Single-line value
                        value = value.strip('"').strip("'")
                        if value in ("true", "false"):
                            metadata[key] = (value == "true")
                        elif value == "":
                            metadata[key] = []
                            current_key = key
                        else:
                            metadata[key] = value
                    else:
                        # Key with no value — start a list
                        metadata[key] = []
                        current_key = key
                        current_list = []
            elif stripped.startswith("- ") and current_key:
                val = stripped[2:].strip().strip('"').strip("'")
                current_list.append(val)

        if current_key and current_list:
            metadata[current_key] = current_list

        return metadata, body
