"""
MCP Client — stdio transport for Model Context Protocol.

Implements initialize, tools/list, and tools/call.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any

from mini_core.mcp.server_config import MCPServerConfig

logger = logging.getLogger("mini-core.mcp.client")


@dataclass
class MCPToolDef:
    name: str
    description: str = ""
    inputSchema: dict = field(default_factory=dict)


@dataclass
class MCPToolResult:
    content: str = ""
    isError: bool = False


@dataclass
class ServerCapabilities:
    tools: list[MCPToolDef] = field(default_factory=list)
    resources: list[dict] = field(default_factory=list)


class MCPConnection:
    """An active connection to an MCP server via stdio."""

    def __init__(self, config: MCPServerConfig) -> None:
        self.config = config
        self._proc: asyncio.subprocess.Process | None = None
        self._next_id = 0
        self._capabilities: ServerCapabilities | None = None

    async def connect(self) -> None:
        env = os.environ.copy()
        for k, v in self.config.env.items():
            if v.startswith("${env:") and v.endswith("}"):
                env_var = v[6:-1]
                env[k] = os.environ.get(env_var, "")
            else:
                env[k] = v

        cmd = [self.config.command] + self.config.args
        self._proc = await asyncio.create_subprocess_exec(
            *cmd, stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE, env=env,
        )
        logger.info("MCP client connected to %s", self.config.name)

    async def disconnect(self) -> None:
        if self._proc:
            self._proc.terminate()
            await self._proc.wait()
            self._proc = None

    async def _send_request(self, method: str, params: dict | None = None) -> dict:
        if not self._proc or not self._proc.stdin:
            raise ConnectionError("MCP not connected")
        self._next_id += 1
        req = {"jsonrpc": "2.0", "id": self._next_id, "method": method, "params": params or {}}
        line = json.dumps(req) + "\n"
        self._proc.stdin.write(line.encode("utf-8"))
        await self._proc.stdin.drain()

        if not self._proc.stdout:
            raise ConnectionError("MCP stdout not available")
        resp_line = await self._proc.stdout.readline()
        if not resp_line:
            raise ConnectionError("MCP server closed connection")
        return json.loads(resp_line.decode("utf-8"))

    async def initialize(self) -> ServerCapabilities:
        result = await self._send_request("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "miniclaude", "version": "0.1.0"},
        })
        self._capabilities = ServerCapabilities()
        return self._capabilities

    async def list_tools(self) -> list[MCPToolDef]:
        result = await self._send_request("tools/list")
        tools_data = result.get("result", {}).get("tools", [])
        return [MCPToolDef(name=t["name"], description=t.get("description", ""),
                           inputSchema=t.get("inputSchema", {})) for t in tools_data]

    async def call_tool(self, name: str, args: dict) -> MCPToolResult:
        result = await self._send_request("tools/call", {"name": name, "arguments": args})
        r = result.get("result", {})
        content = ""
        for c in r.get("content", []):
            if isinstance(c, dict) and "text" in c:
                content += c["text"]
        return MCPToolResult(content=content, isError=r.get("isError", False))
