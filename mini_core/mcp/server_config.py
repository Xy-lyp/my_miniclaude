"""MCP Server configuration management."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class MCPServerConfig:
    name: str
    transport: str = "stdio"  # "stdio" | "sse"
    command: str = ""
    args: list[str] = field(default_factory=list)
    url: str = ""
    env: dict[str, str] = field(default_factory=dict)
    enabled: bool = True

    @classmethod
    def from_dict(cls, d: dict) -> MCPServerConfig:
        return cls(
            name=d["name"], transport=d.get("transport", "stdio"),
            command=d.get("command", ""), args=d.get("args", []),
            url=d.get("url", ""), env=d.get("env", {}),
            enabled=d.get("enabled", True),
        )

    def to_dict(self) -> dict:
        return {
            "name": self.name, "transport": self.transport,
            "command": self.command, "args": self.args,
            "url": self.url, "env": self.env,
            "enabled": self.enabled,
        }


class MCPServerConfigManager:
    DEFAULT_CONFIG_PATH = Path.home() / ".kama" / "mcp_servers.json"

    def __init__(self, config_path: Path | None = None) -> None:
        self._path = config_path or self.DEFAULT_CONFIG_PATH
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._servers: dict[str, MCPServerConfig] = {}
        self._load()

    def _load(self) -> None:
        if not self._path.exists():
            self._save_default()
            return
        try:
            data = json.loads(self._path.read_text())
            for s in data.get("servers", []):
                cfg = MCPServerConfig.from_dict(s)
                self._servers[cfg.name] = cfg
        except Exception:
            self._servers = {}

    def _save_default(self) -> None:
        default = {"servers": []}
        self._path.write_text(json.dumps(default, indent=2))

    def save(self) -> None:
        data = {"servers": [s.to_dict() for s in self._servers.values()]}
        self._path.write_text(json.dumps(data, indent=2))

    def list_all(self) -> list[MCPServerConfig]:
        return list(self._servers.values())

    def list_enabled(self) -> list[MCPServerConfig]:
        return [s for s in self._servers.values() if s.enabled]

    def get(self, name: str) -> MCPServerConfig | None:
        return self._servers.get(name)

    def add(self, config: MCPServerConfig) -> None:
        self._servers[config.name] = config
        self.save()

    def remove(self, name: str) -> bool:
        if name in self._servers:
            del self._servers[name]
            self.save()
            return True
        return False
