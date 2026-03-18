"""Tests for MCP support — schema conversion and client proxy."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from spare_paw.mcp.schema import extract_mcp_result, mcp_to_openai_schema
from spare_paw.tools.registry import ToolRegistry


# ===========================================================================
# Schema conversion
# ===========================================================================


class TestMCPToOpenAISchema:
    def test_basic_conversion(self):
        mcp_tool = MagicMock()
        mcp_tool.name = "create_issue"
        mcp_tool.description = "Create a GitHub issue"
        mcp_tool.inputSchema = {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "body": {"type": "string"},
            },
            "required": ["title"],
        }

        result = mcp_to_openai_schema(mcp_tool)

        assert result["type"] == "function"
        assert result["function"]["name"] == "create_issue"
        assert result["function"]["description"] == "Create a GitHub issue"
        assert result["function"]["parameters"] == mcp_tool.inputSchema

    def test_empty_description(self):
        mcp_tool = MagicMock()
        mcp_tool.name = "noop"
        mcp_tool.description = None
        mcp_tool.inputSchema = {"type": "object", "properties": {}}

        result = mcp_to_openai_schema(mcp_tool)
        assert result["function"]["description"] == ""

    def test_none_input_schema(self):
        mcp_tool = MagicMock()
        mcp_tool.name = "simple"
        mcp_tool.description = "A simple tool"
        mcp_tool.inputSchema = None

        result = mcp_to_openai_schema(mcp_tool)
        assert result["function"]["parameters"] == {"type": "object", "properties": {}}


# ===========================================================================
# extract_mcp_result
# ===========================================================================


class TestExtractMCPResult:
    def test_single_text_block(self):
        block = MagicMock()
        block.text = "Hello, world!"
        result = MagicMock()
        result.content = [block]
        result.isError = False

        assert extract_mcp_result(result) == "Hello, world!"

    def test_multiple_text_blocks(self):
        block1 = MagicMock()
        block1.text = "Line 1"
        block2 = MagicMock()
        block2.text = "Line 2"
        result = MagicMock()
        result.content = [block1, block2]
        result.isError = False

        assert extract_mcp_result(result) == "Line 1\nLine 2"

    def test_error_result(self):
        block = MagicMock()
        block.text = "something went wrong"
        result = MagicMock()
        result.content = [block]
        result.isError = True

        text = extract_mcp_result(result)
        assert text.startswith("MCP error:")
        assert "something went wrong" in text

    def test_binary_block(self):
        block = MagicMock(spec=[])  # no .text attribute
        block.data = b"binary"
        block.mimeType = "image/png"
        result = MagicMock()
        result.content = [block]
        result.isError = False

        text = extract_mcp_result(result)
        assert "binary" in text
        assert "image/png" in text


# ===========================================================================
# MCPClientManager — proxy registration
# ===========================================================================


class TestMCPClientProxy:
    @pytest.mark.asyncio
    async def test_proxy_registers_tools(self):
        """Verify that connecting a mock MCP server registers namespaced tools."""
        from spare_paw.mcp.client import MCPClientManager

        registry = ToolRegistry()

        # Mock the MCP tool discovery
        mock_tool = MagicMock()
        mock_tool.name = "list_files"
        mock_tool.description = "List files in a directory"
        mock_tool.inputSchema = {
            "type": "object",
            "properties": {"path": {"type": "string"}},
        }

        mock_session = AsyncMock()
        mock_session.initialize = AsyncMock()
        mock_tools_result = MagicMock()
        mock_tools_result.tools = [mock_tool]
        mock_session.list_tools = AsyncMock(return_value=mock_tools_result)

        manager = MCPClientManager()

        with patch("spare_paw.mcp.client.stdio_client") as mock_stdio, \
             patch("spare_paw.mcp.client.ClientSession") as mock_cs_cls:

            # stdio_client returns (read, write) streams via async context manager
            mock_transport_ctx = AsyncMock()
            mock_transport_ctx.__aenter__ = AsyncMock(return_value=(MagicMock(), MagicMock()))
            mock_transport_ctx.__aexit__ = AsyncMock(return_value=False)
            mock_stdio.return_value = mock_transport_ctx

            # ClientSession is also an async context manager
            mock_session_ctx = AsyncMock()
            mock_session_ctx.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session_ctx.__aexit__ = AsyncMock(return_value=False)
            mock_cs_cls.return_value = mock_session_ctx

            await manager.connect_all(
                [{"name": "fs", "command": "echo", "args": []}],
                registry,
            )

        assert "fs.list_files" in registry
        assert len(registry) == 1

        status = manager.get_status()
        assert status["total_tools"] == 1
        assert status["servers"][0]["name"] == "fs"

    @pytest.mark.asyncio
    async def test_proxy_calls_session(self):
        """Verify that invoking a proxied tool calls session.call_tool."""
        from spare_paw.mcp.client import MCPClientManager

        mock_session = AsyncMock()
        mock_result = MagicMock()
        block = MagicMock()
        block.text = '{"files": ["a.txt"]}'
        mock_result.content = [block]
        mock_result.isError = False
        mock_session.call_tool = AsyncMock(return_value=mock_result)

        proxy = MCPClientManager._make_proxy(mock_session, "list_files")
        result = await proxy(path="/home")

        mock_session.call_tool.assert_called_once_with("list_files", arguments={"path": "/home"})
        assert '{"files": ["a.txt"]}' in result

    @pytest.mark.asyncio
    async def test_proxy_handles_error(self):
        """Verify that proxy returns error string on exception."""
        from spare_paw.mcp.client import MCPClientManager

        mock_session = AsyncMock()
        mock_session.call_tool = AsyncMock(side_effect=ConnectionError("lost"))

        proxy = MCPClientManager._make_proxy(mock_session, "broken")
        result = await proxy()

        assert "ConnectionError" in result
        assert "lost" in result

    @pytest.mark.asyncio
    async def test_connect_skips_failing_server(self):
        """Verify that a failing server is skipped without crashing."""
        from spare_paw.mcp.client import MCPClientManager

        manager = MCPClientManager()

        with patch("spare_paw.mcp.client.stdio_client", side_effect=FileNotFoundError("no such command")):
            registry = ToolRegistry()
            await manager.connect_all(
                [{"name": "bad", "command": "nonexistent"}],
                registry,
            )

        assert len(registry) == 0
        assert manager.get_status()["total_tools"] == 0
