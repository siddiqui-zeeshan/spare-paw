"""Tests for the coding tool."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from spare_paw.tools.registry import ToolRegistry


def _make_app_state() -> MagicMock:
    app_state = MagicMock()
    app_state.config = MagicMock()
    app_state.config.get = MagicMock(side_effect=lambda key, default=None: {
        "models.coder": "z-ai/glm-5",
        "models.main_agent": "google/gemini-2.0-flash",
        "agent.max_tool_iterations": 10,
    }.get(key, default))
    app_state.router_client = AsyncMock()
    app_state.executor = None
    app_state.tool_registry = ToolRegistry()
    return app_state


class TestCodeTool:
    def test_registration(self):
        """Verify the code tool registers correctly."""
        from spare_paw.tools.code import register

        registry = ToolRegistry()
        app_state = _make_app_state()
        register(registry, {}, app_state)

        assert "code" in registry
        schemas = registry.get_schemas()
        func = schemas[0]["function"]
        assert func["name"] == "code"
        assert "task" in func["parameters"]["properties"]

    @pytest.mark.asyncio
    @patch("spare_paw.tools.code.run_tool_loop", new_callable=AsyncMock)
    async def test_uses_coder_model(self, mock_loop):
        """Verify the coding tool uses models.coder."""
        from spare_paw.tools.code import _handle_code

        mock_loop.return_value = "Fixed the bug"
        app_state = _make_app_state()

        result = json.loads(await _handle_code(app_state, "fix the login bug"))

        assert result["model"] == "z-ai/glm-5"
        assert "Fixed the bug" in result["result"]
        mock_loop.assert_awaited_once()

    @pytest.mark.asyncio
    @patch("spare_paw.tools.code.run_tool_loop", new_callable=AsyncMock)
    async def test_falls_back_to_main_agent_model(self, mock_loop):
        """Verify fallback to main_agent model when coder is not set."""
        from spare_paw.tools.code import _handle_code

        mock_loop.return_value = "Done"
        app_state = _make_app_state()
        app_state.config.get = MagicMock(side_effect=lambda key, default=None: {
            "models.coder": None,
            "models.main_agent": "google/gemini-2.0-flash",
            "agent.max_tool_iterations": 10,
        }.get(key, default))

        result = json.loads(await _handle_code(app_state, "write a script"))

        assert result["model"] == "google/gemini-2.0-flash"

    @pytest.mark.asyncio
    @patch("spare_paw.tools.code.run_tool_loop", new_callable=AsyncMock)
    async def test_handles_error(self, mock_loop):
        """Verify error handling when the tool loop fails."""
        from spare_paw.tools.code import _handle_code

        mock_loop.side_effect = RuntimeError("model down")
        app_state = _make_app_state()

        result = json.loads(await _handle_code(app_state, "fix it"))

        assert "error" in result
        assert "RuntimeError" in result["error"]

    @pytest.mark.asyncio
    @patch("spare_paw.tools.code.run_tool_loop", new_callable=AsyncMock)
    async def test_filters_tools(self, mock_loop):
        """Verify only shell, files, and MCP tools are passed to the sub-agent."""
        from spare_paw.tools.code import _handle_code

        mock_loop.return_value = "Done"
        app_state = _make_app_state()

        # Register various tools
        registry = app_state.tool_registry

        async def noop(**kw):
            return "ok"

        registry.register("execute_shell", "Shell", {}, noop)
        registry.register("execute_files", "Files", {}, noop)
        registry.register("send_message", "Send msg", {}, noop)
        registry.register("github.create_issue", "GH issue", {}, noop)

        await _handle_code(app_state, "do something")

        # Check what tools were passed to run_tool_loop
        call_kwargs = mock_loop.call_args
        tools = call_kwargs.kwargs.get("tools") or call_kwargs[1].get("tools") or call_kwargs[0][3]
        tool_names = [t["function"]["name"] for t in tools]

        assert "execute_shell" in tool_names
        assert "execute_files" in tool_names
        assert "github.create_issue" in tool_names  # MCP tool (has '.')
        assert "send_message" not in tool_names  # excluded
