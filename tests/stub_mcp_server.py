"""A minimal MCP server, used as a real child process by the stdio tests.

Kept to the three methods this platform calls, and deliberately dependency-free
so the transport is exercised against a separate interpreter rather than an
in-process fake. It is not imported by the test suite; it is executed.
"""

from __future__ import annotations

import json
import sys

TOOLS = [
    {
        "name": "echo",
        "description": "Echo a message back.",
        "inputSchema": {
            "type": "object",
            "properties": {"message": {"type": "string"}},
            "required": ["message"],
            "additionalProperties": False,
        },
        # Self-asserted safety metadata, which this platform must ignore.
        "annotations": {"readOnlyHint": True},
    }
]


def handle(request: dict[str, object]) -> dict[str, object] | None:
    method = request.get("method")
    if method == "initialize":
        result: dict[str, object] = {
            "protocolVersion": "2025-06-18",
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "stub", "version": "9.9.9"},
        }
    elif method == "tools/list":
        result = {"tools": TOOLS}
    elif method == "tools/call":
        params = request.get("params")
        assert isinstance(params, dict)
        arguments = params.get("arguments") or {}
        assert isinstance(arguments, dict)
        if params.get("name") != "echo":
            return {
                "jsonrpc": "2.0",
                "id": request.get("id"),
                "error": {"code": -32602, "message": "unknown tool"},
            }
        result = {"content": [{"type": "text", "text": f"echo: {arguments.get('message')}"}]}
    elif method is not None and str(method).startswith("notifications/"):
        return None
    else:
        return {
            "jsonrpc": "2.0",
            "id": request.get("id"),
            "error": {"code": -32601, "message": "method not found"},
        }
    return {"jsonrpc": "2.0", "id": request.get("id"), "result": result}


def main() -> None:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        response = handle(json.loads(line))
        if response is None:
            continue
        sys.stdout.write(json.dumps(response) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
