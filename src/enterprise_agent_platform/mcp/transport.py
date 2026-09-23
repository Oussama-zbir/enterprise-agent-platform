"""How JSON-RPC messages reach an MCP server.

The port is narrow on purpose — request, notify, close — so the client owns the
protocol and the adapter owns only framing and process lifetime. That is what
lets the whole client be tested against an in-process stub with no subprocess,
no sockets, and no external server in CI, which is the same trick the LLM port
already uses for model calls.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from collections.abc import Mapping, Sequence
from typing import Any, Protocol, runtime_checkable

from enterprise_agent_platform.mcp.protocol import (
    JSONRPC_VERSION,
    MCPProtocolError,
    MCPRemoteError,
    MCPTransportError,
)

logger = logging.getLogger(__name__)

MAX_FRAME_BYTES = 4 * 1024 * 1024
"""Largest single JSON-RPC line accepted from a server.

A bound is required rather than optional: ``StreamReader`` buffers a line in
memory before it can be parsed, so without one a server that never writes a
newline is an out-of-memory condition in this process.
"""

TERMINATE_GRACE_SECONDS = 5.0


@runtime_checkable
class MCPTransport(Protocol):
    """A bidirectional JSON-RPC channel to one MCP server."""

    async def request(self, method: str, params: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """Send a request and return its ``result`` object.

        The deadline belongs to the caller, which cancels this coroutine — the
        same split ``LLMClient`` uses with the model port, and the reason a
        transport has no policy of its own to disagree with.

        Raises:
            MCPRemoteError: the server answered with a JSON-RPC error.
            MCPTransportError: the channel failed or the server went away.
        """
        ...

    async def notify(self, method: str, params: Mapping[str, Any] | None = None) -> None:
        """Send a notification, which by definition has no reply."""
        ...

    async def aclose(self) -> None:
        """Release the channel. Safe to call more than once."""
        ...


class StdioTransport:
    """Runs an MCP server as a child process and speaks newline-delimited JSON.

    A single reader task owns stdout and hands each reply to the future waiting
    on its id, so concurrent tool calls multiplex over the one pipe instead of
    serialising behind a lock. Writes are serialised, because two coroutines
    interleaving partial lines on stdin would corrupt the stream.

    The child's environment is *not* inherited by default. An MCP server is
    third-party code with a shell on this machine; handing it the platform's
    ``EAP_ANTHROPIC_API_KEY`` and database URL because it happened to be started
    from the same process is an avoidable credential leak. A deployment passes
    exactly what the server needs.
    """

    def __init__(
        self,
        command: str,
        args: Sequence[str] = (),
        *,
        env: Mapping[str, str] | None = None,
        cwd: str | None = None,
    ) -> None:
        self._command = command
        self._args = tuple(args)
        self._env = dict(env or {})
        self._cwd = cwd
        self._process: asyncio.subprocess.Process | None = None
        self._pending: dict[int, asyncio.Future[dict[str, Any]]] = {}
        self._reader_task: asyncio.Task[None] | None = None
        self._stderr_task: asyncio.Task[None] | None = None
        self._write_lock = asyncio.Lock()
        self._next_id = 0
        self._closed = False

    async def start(self) -> None:
        if self._process is not None:
            raise RuntimeError("transport already started")
        try:
            process = await asyncio.create_subprocess_exec(
                self._command,
                *self._args,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env={**self._env, "PATH": os.environ.get("PATH", "")},
                cwd=self._cwd,
                limit=MAX_FRAME_BYTES,
            )
        except OSError as exc:
            raise MCPTransportError(f"could not start MCP server '{self._command}'") from exc
        self._process = process
        self._reader_task = asyncio.create_task(self._read_stdout())
        self._stderr_task = asyncio.create_task(self._drain_stderr())

    async def request(self, method: str, params: Mapping[str, Any] | None = None) -> dict[str, Any]:
        self._next_id += 1
        request_id = self._next_id
        future: asyncio.Future[dict[str, Any]] = asyncio.get_running_loop().create_future()
        self._pending[request_id] = future
        message: dict[str, Any] = {
            "jsonrpc": JSONRPC_VERSION,
            "id": request_id,
            "method": method,
        }
        if params is not None:
            message["params"] = dict(params)
        try:
            await self._write(message)
            return await future
        finally:
            # Also the abandonment path: when the caller's deadline cancels this
            # coroutine the id is retired here, so a reply that arrives later
            # finds nothing waiting for it and is dropped by the reader rather
            # than resolving a future nobody holds.
            self._pending.pop(request_id, None)

    async def notify(self, method: str, params: Mapping[str, Any] | None = None) -> None:
        message: dict[str, Any] = {"jsonrpc": JSONRPC_VERSION, "method": method}
        if params is not None:
            message["params"] = dict(params)
        await self._write(message)

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        process = self._process
        for task in (self._reader_task, self._stderr_task):
            if task is not None:
                task.cancel()
        if process is None:
            return
        if process.returncode is None:
            # Closing stdin is how an MCP server is asked to exit; terminate and
            # kill are the escalation for one that ignores it, so a shutdown
            # cannot hang on a misbehaving child.
            if process.stdin is not None and not process.stdin.is_closing():
                process.stdin.close()
            try:
                async with asyncio.timeout(TERMINATE_GRACE_SECONDS):
                    await process.wait()
            except TimeoutError:
                process.terminate()
                try:
                    async with asyncio.timeout(TERMINATE_GRACE_SECONDS):
                        await process.wait()
                except TimeoutError:
                    process.kill()
                    await process.wait()
        self._fail_pending(MCPTransportError("MCP transport closed"))

    async def _write(self, message: Mapping[str, Any]) -> None:
        process = self._process
        if self._closed or process is None or process.stdin is None:
            raise MCPTransportError("MCP transport is not connected")
        payload = json.dumps(message, separators=(",", ":")).encode() + b"\n"
        async with self._write_lock:
            try:
                process.stdin.write(payload)
                await process.stdin.drain()
            except (BrokenPipeError, ConnectionResetError) as exc:
                raise MCPTransportError("MCP server closed its input stream") from exc

    async def _read_stdout(self) -> None:
        process = self._process
        assert process is not None and process.stdout is not None
        try:
            while True:
                try:
                    line = await process.stdout.readline()
                except ValueError as exc:
                    raise MCPTransportError(
                        f"MCP server sent a frame larger than {MAX_FRAME_BYTES} bytes"
                    ) from exc
                if not line:
                    raise MCPTransportError("MCP server closed its output stream")
                stripped = line.strip()
                if stripped:
                    self._dispatch(stripped)
        except asyncio.CancelledError:
            raise
        except MCPTransportError as exc:
            self._fail_pending(exc)
        except Exception as exc:  # pragma: no cover - defensive
            self._fail_pending(MCPTransportError(f"MCP transport failed: {exc!r}"))

    def _dispatch(self, raw: bytes) -> None:
        try:
            message = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("mcp.frame.invalid", extra={"command": self._command})
            return
        if not isinstance(message, dict):
            logger.warning("mcp.frame.invalid", extra={"command": self._command})
            return
        message_id = message.get("id")
        if not isinstance(message_id, int):
            # A server-initiated request or notification. This client advertises
            # no capabilities, so anything arriving here is unsolicited and is
            # ignored rather than answered.
            return
        future = self._pending.get(message_id)
        if future is None or future.done():
            return
        error = message.get("error")
        if isinstance(error, dict):
            code = error.get("code")
            detail = error.get("message")
            future.set_exception(
                MCPRemoteError(
                    code if isinstance(code, int) else -32000,
                    str(detail) if detail is not None else "unspecified",
                )
            )
            return
        result = message.get("result")
        if not isinstance(result, dict):
            future.set_exception(MCPProtocolError("MCP response carried no result object"))
            return
        future.set_result(result)

    def _fail_pending(self, exc: MCPTransportError) -> None:
        for future in list(self._pending.values()):
            if not future.done():
                future.set_exception(exc)

    async def _drain_stderr(self) -> None:
        """Log the server's stderr so a failing child is diagnosable.

        Unread stderr fills its pipe buffer and blocks the child, so this task
        has to exist even if the output were of no interest.
        """
        process = self._process
        assert process is not None and process.stderr is not None
        while True:
            try:
                line = await process.stderr.readline()
            except (ValueError, OSError):
                return
            if not line:
                return
            logger.warning(
                "mcp.server.stderr",
                extra={
                    "command": self._command,
                    "output": line.decode(errors="replace").rstrip(),
                },
            )
