"""Tests for core/commands.py — platform-agnostic command functions."""

from __future__ import annotations

import ast
import inspect
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from spare_paw.core.commands import (
    cmd_config_show,
    cmd_forget,
    cmd_model,
    cmd_search,
    cmd_status,
)


def _make_app_state():
    app_state = MagicMock()
    app_state.config.get = lambda key, default=None: {
        "models.default": "test-model",
        "models.smart": "smart-model",
        "models.cron_default": "cron-model",
    }.get(key, default)
    app_state.config._overrides = {}
    app_state.start_time = datetime(2026, 3, 20, 0, 0, 0, tzinfo=timezone.utc)
    return app_state


class TestCmdStatus:
    @pytest.mark.asyncio
    async def test_returns_string_with_uptime(self):
        app_state = _make_app_state()

        mock_cursor = MagicMock()
        mock_cursor.fetchone = AsyncMock(return_value={"cnt": 2})
        mock_cursor.__aenter__ = AsyncMock(return_value=mock_cursor)
        mock_cursor.__aexit__ = AsyncMock(return_value=False)

        mock_db = MagicMock()
        mock_db.execute = MagicMock(return_value=mock_cursor)

        with patch("spare_paw.core.commands.get_db", new_callable=AsyncMock, return_value=mock_db):
            result = await cmd_status(app_state)

        assert isinstance(result, str)
        assert "Uptime" in result or "uptime" in result.lower()


class TestCmdForget:
    @pytest.mark.asyncio
    async def test_returns_confirmation(self):
        app_state = _make_app_state()
        with patch("spare_paw.core.commands.ctx_module") as mock_ctx:
            mock_ctx.new_conversation = AsyncMock(return_value="new-conv-id")
            result = await cmd_forget(app_state)

        assert isinstance(result, str)
        assert "new" in result.lower() or "conversation" in result.lower()


class TestCmdSearch:
    @pytest.mark.asyncio
    async def test_returns_results(self):
        app_state = _make_app_state()
        with patch("spare_paw.core.commands.ctx_module") as mock_ctx:
            mock_ctx.search = AsyncMock(return_value=[
                {"role": "user", "content": "hello world", "created_at": "2026-03-20T00:00:00"},
            ])
            result = await cmd_search(app_state, "hello")

        assert isinstance(result, str)
        assert "hello" in result.lower()

    @pytest.mark.asyncio
    async def test_no_results(self):
        app_state = _make_app_state()
        with patch("spare_paw.core.commands.ctx_module") as mock_ctx:
            mock_ctx.search = AsyncMock(return_value=[])
            result = await cmd_search(app_state, "nonexistent")

        assert isinstance(result, str)
        assert "no result" in result.lower()


class TestCmdModel:
    @pytest.mark.asyncio
    async def test_sets_model(self):
        app_state = _make_app_state()
        result = await cmd_model(app_state, "new-model")

        assert isinstance(result, str)
        assert "new-model" in result
        app_state.config.set_override.assert_called_once_with("models.default", "new-model")

    @pytest.mark.asyncio
    async def test_shows_current_when_no_arg(self):
        app_state = _make_app_state()
        result = await cmd_model(app_state, None)

        assert isinstance(result, str)
        assert "test-model" in result


class TestCmdConfigShow:
    @pytest.mark.asyncio
    async def test_returns_model_info(self):
        app_state = _make_app_state()
        result = await cmd_config_show(app_state)

        assert isinstance(result, str)
        assert "test-model" in result


class TestNoTelegramImport:
    def test_no_telegram_import(self):
        import spare_paw.core.commands as mod

        source = inspect.getsource(mod)
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert not alias.name.startswith("telegram"), (
                        f"Found 'import {alias.name}' in core/commands.py"
                    )
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    assert not node.module.startswith("telegram"), (
                        f"Found 'from {node.module}' in core/commands.py"
                    )
