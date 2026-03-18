"""MCP client manager — connects to external MCP servers and proxies tools.

Each configured MCP server is launched as a subprocess (stdio transport).
Its tools are discovered, namespaced as ``servername.toolname``, and
registered in the shared ToolRegistry.
"""

from __future__ import annotations

import logging
from contextlib import AsyncExitStack
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from spare_paw.mcp.schema import extract_mcp_result

logger = logging.getLogger(__name__)


class MCPClientManager:
    """Manages connections to multiple MCP servers."""

    def __init__(self) -> None:
        self._exit_stack = AsyncExitStack()
        self._sessions: dict[str, ClientSession] = {}
        self._server_tools: dict[str, list[str]] = {}  # server_name -> [tool_names]

    async def connect_all(
        self,
        servers_config: list[dict[str, Any]],
        tool_registry: Any,
    ) -> None:
        """Connect to all configured MCP servers and register their tools.

        Parameters
        ----------
        servers_config:
            List of server dicts with ``name``, ``command``, ``args``, and
            optional ``env`` keys.
        tool_registry:
            The ToolRegistry to register discovered tools into.
        """
        for server_cfg in servers_config:
            name = server_cfg.get("name", "unknown")
            try:
                await self._connect_one(server_cfg, tool_registry)
            except Exception:
                logger.exception("Failed to connect to MCP server %r — skipping", name)

    async def _connect_one(
        self,
        server_cfg: dict[str, Any],
        tool_registry: Any,
    ) -> None:
        """Connect to a single MCP server."""
        name = server_cfg["name"]
        command = server_cfg["command"]
        args = server_cfg.get("args", [])
        env = server_cfg.get("env")

        params = StdioServerParameters(command=command, args=args, env=env)

        stdio_transport = await self._exit_stack.enter_async_context(
            stdio_client(params)
        )
        read_stream, write_stream = stdio_transport
        session = await self._exit_stack.enter_async_context(
            ClientSession(read_stream, write_stream)
        )
        await session.initialize()

        self._sessions[name] = session

        # Discover tools
        tools_result = await session.list_tools()
        tool_names: list[str] = []

        for mcp_tool in tools_result.tools:
            namespaced = f"{name}.{mcp_tool.name}"
            tool_names.append(namespaced)

            # Create a proxy handler bound to this session and tool name
            handler = self._make_proxy(session, mcp_tool.name)

            tool_registry.register(
                name=namespaced,
                description=mcp_tool.description or "",
                parameters_schema=mcp_tool.inputSchema or {"type": "object", "properties": {}},
                handler=handler,
                run_in_executor=False,
            )

        self._server_tools[name] = tool_names
        logger.info(
            "Connected to MCP server %r: %d tools",
            name,
            len(tool_names),
        )

    @staticmethod
    def _make_proxy(session: ClientSession, tool_name: str):
        """Create an async proxy handler for a remote MCP tool."""

        async def _proxy(**kwargs: Any) -> str:
            try:
                result = await session.call_tool(tool_name, arguments=kwargs)
                return extract_mcp_result(result)
            except Exception as exc:
                return f"MCP tool error ({tool_name}): {type(exc).__name__}: {exc}"

        return _proxy

    def get_status(self) -> dict[str, Any]:
        """Return connection status for the /mcp command."""
        servers: list[dict[str, Any]] = []
        for name, tools in self._server_tools.items():
            servers.append({
                "name": name,
                "connected": name in self._sessions,
                "tools": len(tools),
                "tool_names": tools,
            })
        return {"servers": servers, "total_tools": sum(len(t) for t in self._server_tools.values())}

    async def close(self) -> None:
        """Tear down all MCP sessions and transports."""
        try:
            await self._exit_stack.aclose()
        except Exception:
            logger.exception("Error closing MCP client sessions")
        self._sessions.clear()
        self._server_tools.clear()
