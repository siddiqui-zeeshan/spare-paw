"""Tests for core/prompt.py — system prompt builder (leaf module)."""

from __future__ import annotations

import ast
import inspect
from unittest.mock import AsyncMock, patch

import pytest

from spare_paw.core.prompt import build_system_prompt

# Patch target for the lazy import inside build_system_prompt
_MEMORY_PATCH = "spare_paw.tools.memory.get_all_memories"


class TestBuildSystemPrompt:
    @pytest.mark.asyncio
    async def test_returns_base_prompt(self, tmp_path):
        """Returns at least the base system prompt from config."""
        mock_config = AsyncMock()
        mock_config.get = lambda key, default="": {
            "agent.system_prompt": "You are a helpful bot.",
        }.get(key, default)

        with patch("spare_paw.core.prompt._PROMPT_DIR", tmp_path), \
             patch(_MEMORY_PATCH, new_callable=AsyncMock, return_value=[]):
            result = await build_system_prompt(mock_config)

        assert "You are a helpful bot." in result

    @pytest.mark.asyncio
    async def test_current_time_replacement(self, tmp_path):
        """Replaces {current_time} in the base prompt."""
        mock_config = AsyncMock()
        mock_config.get = lambda key, default="": {
            "agent.system_prompt": "Time is {current_time}.",
        }.get(key, default)

        with patch("spare_paw.core.prompt._PROMPT_DIR", tmp_path), \
             patch(_MEMORY_PATCH, new_callable=AsyncMock, return_value=[]):
            result = await build_system_prompt(mock_config)

        assert "{current_time}" not in result
        assert "Time is " in result

    @pytest.mark.asyncio
    async def test_includes_prompt_files(self, tmp_path):
        """Loads IDENTITY.md, USER.md, SYSTEM.md from prompt dir."""
        (tmp_path / "IDENTITY.md").write_text("I am Claw.")
        (tmp_path / "USER.md").write_text("User likes cats.")

        mock_config = AsyncMock()
        mock_config.get = lambda key, default="": {
            "agent.system_prompt": "Base.",
        }.get(key, default)

        with patch("spare_paw.core.prompt._PROMPT_DIR", tmp_path), \
             patch(_MEMORY_PATCH, new_callable=AsyncMock, return_value=[]):
            result = await build_system_prompt(mock_config)

        assert "I am Claw." in result
        assert "User likes cats." in result

    @pytest.mark.asyncio
    async def test_includes_memories(self, tmp_path):
        """Injects persistent memories into the prompt."""
        mock_config = AsyncMock()
        mock_config.get = lambda key, default="": {
            "agent.system_prompt": "Base.",
        }.get(key, default)

        memories = [
            {"key": "name", "value": "Zeeshan"},
            {"key": "pref", "value": "likes Python"},
        ]

        with patch("spare_paw.core.prompt._PROMPT_DIR", tmp_path), \
             patch(_MEMORY_PATCH, new_callable=AsyncMock, return_value=memories):
            result = await build_system_prompt(mock_config)

        assert "name: Zeeshan" in result
        assert "pref: likes Python" in result

    @pytest.mark.asyncio
    async def test_includes_skills(self, tmp_path):
        """Loads skill files from skills/ subdirectory."""
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        (skills_dir / "research.md").write_text("Search the web thoroughly.")

        mock_config = AsyncMock()
        mock_config.get = lambda key, default="": {
            "agent.system_prompt": "Base.",
        }.get(key, default)

        with patch("spare_paw.core.prompt._PROMPT_DIR", tmp_path), \
             patch(_MEMORY_PATCH, new_callable=AsyncMock, return_value=[]):
            result = await build_system_prompt(mock_config)

        assert "Search the web thoroughly." in result


class TestLeafModuleInvariant:
    """core/prompt.py must be a leaf module — no imports from engine, router, or tools/subagent."""

    def test_no_forbidden_imports(self):
        import spare_paw.core.prompt as mod

        source = inspect.getsource(mod)
        tree = ast.parse(source)

        forbidden_prefixes = [
            "spare_paw.core.engine",
            "spare_paw.router",
            "spare_paw.tools.subagent",
            "telegram",
        ]

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    for prefix in forbidden_prefixes:
                        assert not alias.name.startswith(prefix), (
                            f"Forbidden import '{alias.name}' in core/prompt.py"
                        )
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    for prefix in forbidden_prefixes:
                        assert not node.module.startswith(prefix), (
                            f"Forbidden import 'from {node.module}' in core/prompt.py"
                        )
