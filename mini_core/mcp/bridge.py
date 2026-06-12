"""
MCPBridge — connects MCP tools into the ToolRegistry.

Wraps external MCP tools as native KamaClaude Tool objects with
naming convention: mcp:{server_name}:{tool_name}
"""

from __future__ import annotations

import logging
from typing import Any

from mini_core.mcp.client import MCPConnection, MCPToolDef
from mini_core.mcp.server_config import MCPServerConfig, MCPServerConfigManager
from mini_core.tools.base import Tool, ToolResult
from mini_core.tools.registry import ToolRegistry

logger = logging.getLogger("mini-core.mcp.bridge")


class MCPToolWrapper(Tool):
    """Wraps an MCP tool as a KamaClaude Tool."""

    def __init__(self, mcp_name: str, tool_def: MCPToolDef, connection: MCPConnection) -> None:
        self.name = f"mcp:{mcp_name}:{tool_def.name}"
        self.description = f"[MCP:{mcp_name}] {tool_def.description}"
        self.parameters = tool_def.inputSchema or {
            "type": "object", "properties": {}, "required": [],
        }
        self._connection = connection
        self._mcp_tool_name = tool_def.name

    async def execute(self, **kwargs: Any) -> ToolResult:
        try:
            result = await self._connection.call_tool(self._mcp_tool_name, kwargs)
            if result.isError:
                return ToolResult(success=False, content=result.content,
                                  raw_content=result.content, error="MCP tool returned error")
            return ToolResult(success=True, content=result.content[:4096],
                              raw_content=result.content,
                              metadata={"source": "mcp", "tool": self._mcp_tool_name})
        except Exception as exc:
            return ToolResult(success=False, content=f"MCP error: {exc}",
                              raw_content=str(exc), error=str(exc))


class MCPBridge:
    """Bridges MCP servers to the ToolRegistry."""

    def __init__(self, config_manager: MCPServerConfigManager | None = None) -> None:
        self._config_mgr = config_manager or MCPServerConfigManager()
        self._connections: dict[str, MCPConnection] = {}
        self._tools: dict[str, MCPToolWrapper] = {}

    @property
    def connections(self) -> dict[str, MCPConnection]:
        return self._connections

    async def connect_all(self) -> list[MCPConnection]:
        connections: list[MCPConnection] = []
        for cfg in self._config_mgr.list_enabled():
            try:
                conn = MCPConnection(cfg)
                await conn.connect()
                self._connections[cfg.name] = conn
                connections.append(conn)
            except Exception as exc:
                logger.warning("Failed to connect MCP server '%s': %s", cfg.name, exc)
        return connections

    async def refresh_tools(self, registry: ToolRegistry) -> int:
        """Connect all servers and register their tools. Returns tool count."""
        for name, conn in list(self._connections.items()):
            try:
                await conn.disconnect()
            except Exception:
                pass
        self._connections.clear()

        for name, wrapper in list(self._tools.items()):
            # Remove old wrappers from registry
            pass
        self._tools.clear()

        await self.connect_all()
        count = 0
        for srv_name, conn in self._connections.items():
            try:
                tools = await conn.list_tools()
                for td in tools:
                    wrapper = MCPToolWrapper(srv_name, td, conn)
                    self._tools[wrapper.name] = wrapper
                    registry.register(wrapper)
                    count += 1
            except Exception as exc:
                logger.warning("Failed to list tools from '%s': %s", srv_name, exc)
        return count

    def disconnect_all(self) -> None:
        for conn in self._connections.values():
            try:
                asyncio = __import__("asyncio")
                asyncio.get_event_loop().run_until_complete(conn.disconnect())
            except Exception:
                pass
        self._connections.clear()
