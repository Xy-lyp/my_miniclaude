"""
JSON-RPC 2.0 protocol type definitions.

Strictly follows the JSON-RPC 2.0 specification:
https://www.jsonrpc.org/specification
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

# ── JSON-RPC 2.0 standard error codes ──────────────────────────────────────────

#: Invalid JSON was received by the server.
PARSE_ERROR = -32700
#: The JSON sent is not a valid Request object.
INVALID_REQUEST = -32600
#: The method does not exist / is not available.
METHOD_NOT_FOUND = -32601
#: Invalid method parameter(s).
INVALID_PARAMS = -32602
#: Internal JSON-RPC error.
INTERNAL_ERROR = -32603

# Server error range: -32000 to -32099 (reserved for implementation-defined server errors)

JSONRPC_VERSION = "2.0"


@dataclass
class JsonRpcRequest:
    """A JSON-RPC 2.0 request object."""

    method: str
    params: dict[str, Any] = field(default_factory=dict)
    id: int | str | None = None
    jsonrpc: Literal["2.0"] = "2.0"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> JsonRpcRequest:
        """Parse a dict into a JsonRpcRequest, raising ValueError on invalid format."""
        if not isinstance(data, dict):
            raise ValueError("Request must be a JSON object")

        jsonrpc = data.get("jsonrpc")
        if jsonrpc != "2.0":
            raise ValueError(f"Invalid jsonrpc version: {jsonrpc}")

        if "method" not in data:
            raise ValueError("Missing required field 'method'")

        method = data["method"]
        if not isinstance(method, str):
            raise ValueError("'method' must be a string")

        req_id = data.get("id")
        # id may be a number, a string, or null (notification)
        if req_id is not None and not isinstance(req_id, (int, str)):
            raise ValueError("'id' must be an int, string, or null")

        params = data.get("params", {})
        if not isinstance(params, dict):
            raise ValueError("'params' must be a dict")

        return cls(method=method, params=params, id=req_id, jsonrpc=jsonrpc)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict for JSON encoding."""
        d: dict[str, Any] = {
            "jsonrpc": self.jsonrpc,
            "method": self.method,
            "params": self.params,
        }
        if self.id is not None:
            d["id"] = self.id
        return d

    def is_notification(self) -> bool:
        """A Request with no 'id' is a Notification (no response expected)."""
        return self.id is None


@dataclass
class JsonRpcResponse:
    """A JSON-RPC 2.0 response object (success)."""

    id: int | str | None
    result: Any
    jsonrpc: Literal["2.0"] = "2.0"

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "jsonrpc": self.jsonrpc,
            "id": self.id,
            "result": self.result,
        }
        return d


@dataclass
class JsonRpcError:
    """A JSON-RPC 2.0 error object."""

    code: int
    message: str
    data: Any = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "code": self.code,
            "message": self.message,
        }
        if self.data is not None:
            d["data"] = self.data
        return d


@dataclass
class JsonRpcErrorResponse:
    """A JSON-RPC 2.0 response object (error)."""

    id: int | str | None
    error: JsonRpcError
    jsonrpc: Literal["2.0"] = "2.0"

    def to_dict(self) -> dict[str, Any]:
        return {
            "jsonrpc": self.jsonrpc,
            "id": self.id,
            "error": self.error.to_dict(),
        }


def make_response(req_id: int | str | None, result: Any) -> dict[str, Any]:
    """Build a success response dict for the given request id."""
    return JsonRpcResponse(id=req_id, result=result).to_dict()


def make_error_response(
    req_id: int | str | None,
    code: int,
    message: str,
    data: Any = None,
) -> dict[str, Any]:
    """Build an error response dict for the given request id."""
    return JsonRpcErrorResponse(
        id=req_id,
        error=JsonRpcError(code=code, message=message, data=data),
    ).to_dict()
