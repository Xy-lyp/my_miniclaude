"""
IPC client for the TUI — connects to mini-core daemon, sends JSON-RPC
requests, and receives event pushes.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from mini_core.protocol import JSONRPC_VERSION

logger = logging.getLogger("mini-tui.client")


class TuiIPCClient:
    """Async IPC client for the Textual TUI."""

    def __init__(self, host: str = "127.0.0.1", port: int = 9527):
        self._host = host
        self._port = port
        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None
        self._event_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._connected = False
        self._next_id = 0
        self._pending: dict[int, asyncio.Future] = {}

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def event_queue(self) -> asyncio.Queue:
        return self._event_queue

    async def connect(self) -> None:
        """Connect to the daemon and start the reader loop."""
        try:
            self._reader, self._writer = await asyncio.open_connection(self._host, self._port)
            self._connected = True
            asyncio.create_task(self._reader_loop())
        except Exception as exc:
            logger.error("TUI failed to connect: %s", exc)
            self._connected = False

    async def disconnect(self) -> None:
        """Disconnect from the daemon."""
        self._connected = False
        if self._writer:
            self._writer.close()
            await self._writer.wait_closed()

    async def call(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """Send a JSON-RPC request and wait for the response."""
        if not self._writer:
            raise ConnectionError("Not connected")

        self._next_id += 1
        req_id = self._next_id
        req = {
            "jsonrpc": JSONRPC_VERSION,
            "id": req_id,
            "method": method,
            "params": params or {},
        }
        line = json.dumps(req, ensure_ascii=False) + "\n"
        self._writer.write(line.encode("utf-8"))
        await self._writer.drain()

        future: asyncio.Future[dict[str, Any]] = asyncio.get_event_loop().create_future()
        self._pending[req_id] = future
        return await asyncio.wait_for(future, timeout=300)

    async def subscribe_events(self, event_types: list[str] | None = None) -> None:
        """Subscribe to event pushes from the daemon."""
        await self.call("events.subscribe", {"event_types": event_types or ["*"]})

    async def _reader_loop(self) -> None:
        """Read lines from the daemon, routing responses and events."""
        while self._connected and self._reader:
            try:
                line_bytes = await self._reader.readline()
                if not line_bytes:
                    break
                msg = json.loads(line_bytes.decode("utf-8"))

                msg_type = msg.get("type")
                if msg_type == "event":
                    # Event push from daemon
                    await self._event_queue.put(msg)
                elif "id" in msg:
                    # JSON-RPC response
                    req_id = msg["id"]
                    future = self._pending.pop(req_id, None)
                    if future and not future.done():
                        future.set_result(msg)
            except asyncio.CancelledError:
                break
            except Exception:
                if self._connected:
                    continue
                break
