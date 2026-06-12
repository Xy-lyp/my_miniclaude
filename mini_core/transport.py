"""
NDJSON codec + TCP transport layer.

Encoding rule: each line is one complete JSON object, terminated by \\n.
No single data frame may contain more than one JSON object.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, AsyncIterator

logger = logging.getLogger("mini-core.transport")

# ── NDJSON codec ──────────────────────────────────────────────────────────────


def encode_message(obj: dict[str, Any]) -> bytes:
    """Encode a dict as a single NDJSON line (JSON + newline, UTF-8 bytes)."""
    json_str = json.dumps(obj, ensure_ascii=False)
    return (json_str + "\n").encode("utf-8")


def decode_line(line: str) -> dict[str, Any]:
    """Decode a single NDJSON line string into a dict.

    Raises json.JSONDecodeError on malformed JSON.
    """
    return json.loads(line)


# ── TCP transport ─────────────────────────────────────────────────────────────


class JsonRpcStreamWriter:
    """Wraps an asyncio.StreamWriter for sending NDJSON messages."""

    def __init__(self, writer: asyncio.StreamWriter) -> None:
        self._writer = writer

    async def send(self, message: dict[str, Any]) -> None:
        """Encode and send a JSON-RPC message over the wire."""
        data = encode_message(message)
        self._writer.write(data)
        await self._writer.drain()

    async def close(self) -> None:
        """Close the writer gracefully."""
        try:
            self._writer.close()
            await self._writer.wait_closed()
        except Exception:
            pass

    def is_closing(self) -> bool:
        return self._writer.is_closing()


class JsonRpcStreamReader:
    """Reads NDJSON lines from an asyncio.StreamReader and decodes them."""

    def __init__(self, reader: asyncio.StreamReader) -> None:
        self._reader = reader

    async def receive(self) -> dict[str, Any] | None:
        """Read and decode one NDJSON line.

        Returns None when the connection is closed (EOF).
        """
        try:
            line_bytes = await self._reader.readline()
        except Exception:
            return None

        if not line_bytes:
            # EOF — connection closed
            return None

        line_str = line_bytes.decode("utf-8").rstrip("\n").rstrip("\r")

        if not line_str:
            # Empty line — skip and try next
            return await self.receive()

        return decode_line(line_str)

    async def close(self) -> None:
        """Close the reader (best-effort)."""
        pass  # StreamReader has no explicit close; the transport handles it


class JsonRpcConnection:
    """Bidirectional JSON-RPC connection over TCP NDJSON.

    Holds a reader and writer for a single client connection.
    """

    def __init__(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        self.reader = JsonRpcStreamReader(reader)
        self.writer = JsonRpcStreamWriter(writer)
        self._transport_writer = writer

    async def receive(self) -> dict[str, Any] | None:
        """Receive a single JSON-RPC message (or None on EOF)."""
        return await self.reader.receive()

    async def send(self, message: dict[str, Any]) -> None:
        """Send a JSON-RPC message."""
        await self.writer.send(message)

    async def close(self) -> None:
        """Gracefully close the connection."""
        await self.writer.close()

    def peer_info(self) -> str:
        """Return a human-readable peer address string."""
        peername = self._transport_writer.get_extra_info("peername")
        if peername:
            return f"{peername[0]}:{peername[1]}"
        return "unknown"
