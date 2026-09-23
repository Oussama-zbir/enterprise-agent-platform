"""Adapt discovered MCP tools into the platform's own ``Tool`` type.

This is the seam the whole integration exists for. Once a remote tool is a
``Tool`` in the ``ToolRegistry``, the agent runner, the risk gate, the approval
workflow, the per-call timeout, and the result truncation all apply to it
unchanged — none of them can tell it apart from a tool this repository ships,
and none of them imports anything from ``mcp``.

Two translations happen here.

*Schema.* The server's JSON Schema is what the model is shown, because its
per-property descriptions and constraints are what make tool calls land. A
Pydantic model derived from that schema is what the platform validates against,
so a remote tool gets the same "the handler never sees a raw dictionary from a
language model" guarantee a local one does. The derived model is deliberately a
coarsening — it enforces required fields and broad types and nothing else — so
it can never reject arguments the schema the model was shown would allow.

*Text.* Descriptions come from a remote party and end up in the system prompt,
so they are bounded in length and are the only server-supplied text that gets
that far.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, ConfigDict, create_model

from enterprise_agent_platform.mcp.client import MCPClient
from enterprise_agent_platform.mcp.policy import MCPServerConfig
from enterprise_agent_platform.mcp.protocol import MCPError, MCPToolDeclaration
from enterprise_agent_platform.tools.models import Tool, ToolExecutionError
from enterprise_agent_platform.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

MAX_DESCRIPTION_CHARS = 1_024

_JSON_TYPES: dict[str, Any] = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "array": list[Any],
    "object": dict[str, Any],
}


def build_tool(
    config: MCPServerConfig, client: MCPClient, declaration: MCPToolDeclaration
) -> Tool[Any]:
    """Wrap one discovered MCP tool as a platform tool.

    The risk level comes from ``config`` and the tool's name alone; nothing in
    ``declaration`` influences it.
    """
    local_name = config.local_name(declaration.name)
    arguments = arguments_model(local_name, declaration.input_schema)
    remote_name = declaration.name

    async def handler(args: BaseModel) -> str:
        try:
            outcome = await client.call_tool(remote_name, args.model_dump(exclude_unset=True))
        except MCPError as exc:
            # Protocol and transport failures are defects in the deployment, not
            # something the model can correct, and their text can name hosts and
            # commands. They are logged here and re-raised for the registry to
            # replace with a generic message.
            logger.warning(
                "mcp.tool.call.failed",
                extra={
                    "mcp_server": config.name,
                    "tool": local_name,
                    "error": type(exc).__name__,
                },
            )
            raise
        if outcome.is_error:
            # The server is reporting that the tool failed. That is an outcome
            # the model should see, so it becomes a tool result rather than an
            # exception — the same contract local tools have.
            raise ToolExecutionError(
                outcome.content or f"MCP tool '{remote_name}' reported a failure."
            )
        return outcome.content

    return Tool(
        name=local_name,
        description=_describe(config, declaration),
        arguments=arguments,
        risk=config.risk_for(declaration.name),
        handler=handler,
        input_schema=dict(declaration.input_schema) or {"type": "object", "properties": {}},
    )


def register_mcp_tools(
    registry: ToolRegistry,
    client: MCPClient,
    config: MCPServerConfig,
    declarations: tuple[MCPToolDeclaration, ...],
) -> tuple[str, ...]:
    """Register every usable tool from one server; return the names registered.

    A single unusable tool — a name no provider would accept — is skipped with a
    warning rather than failing startup. The server is third-party, its tool
    list can change without this deployment changing, and losing one capability
    is a better outcome than a service that will not boot.
    """
    registered: list[str] = []
    for declaration in declarations:
        if not config.is_registrable(declaration.name):
            logger.warning(
                "mcp.tool.skipped",
                extra={
                    "mcp_server": config.name,
                    "remote_tool": declaration.name,
                    "reason": "unusable_name",
                },
            )
            continue
        tool = build_tool(config, client, declaration)
        registry.register(tool)
        registered.append(tool.name)
        logger.info(
            "mcp.tool.registered",
            extra={
                "mcp_server": config.name,
                "tool": tool.name,
                "remote_tool": declaration.name,
                "risk": tool.risk.value,
            },
        )
    return tuple(registered)


def arguments_model(local_name: str, schema: Mapping[str, Any]) -> type[BaseModel]:
    """Derive a validation model from an MCP tool's input schema.

    Only the top level is translated. Nested structures stay ``dict``/``list``
    of ``Any`` and are passed through: the provider already constrains decoding
    to the server's full schema, and re-implementing JSON Schema here to check
    it a second time would add a dialect of its own to disagree with.
    """
    properties = schema.get("properties")
    required = set(schema.get("required") or ())
    fields: dict[str, Any] = {}
    all_names_usable = True
    if isinstance(properties, Mapping):
        for prop, subschema in properties.items():
            if not isinstance(prop, str) or not prop.isidentifier() or prop.startswith("_"):
                # Legal in JSON Schema, not expressible as a model field. It
                # still reaches the server, via the extras below.
                all_names_usable = False
                continue
            annotation = _annotation(subschema)
            if prop in required:
                fields[prop] = (annotation, ...)
            else:
                fields[prop] = (annotation | None, None)

    # Extras are forbidden only when the server closed the object *and* every
    # property survived translation, so this model never rejects what the
    # server's own schema accepts.
    closed = schema.get("additionalProperties") is False and all_names_usable
    config = ConfigDict(extra="forbid" if closed else "allow")
    return create_model(f"{_model_name(local_name)}Arguments", __config__=config, **fields)


def _annotation(subschema: object) -> Any:
    if not isinstance(subschema, Mapping):
        return Any
    declared = subschema.get("type")
    if not isinstance(declared, str):
        # Absent, or a union of types: anything narrower risks rejecting a value
        # the schema shown to the model allows.
        return Any
    return _JSON_TYPES.get(declared, Any)


def _describe(config: MCPServerConfig, declaration: MCPToolDeclaration) -> str:
    description = declaration.description.strip()
    if not description:
        # Tool requires a description, and the model needs one to choose well.
        description = f"Tool '{declaration.name}' published by MCP server '{config.name}'."
    if len(description) > MAX_DESCRIPTION_CHARS:
        description = description[:MAX_DESCRIPTION_CHARS].rstrip() + "…"
    return description


def _model_name(local_name: str) -> str:
    return "".join(part.title() for part in local_name.replace("-", "_").split("_") if part)
