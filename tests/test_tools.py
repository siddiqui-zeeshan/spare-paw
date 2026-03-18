"""Tests for spare_paw tools — registry, shell, and files."""

from __future__ import annotations

import json
import os

import pytest

from spare_paw.tools.registry import ToolRegistry
from spare_paw.tools.shell import execute_shell
from spare_paw.tools.files import execute_files, _check_path


# ===========================================================================
# ToolRegistry
# ===========================================================================


class TestToolRegistry:
    def test_register_and_contains(self):
        reg = ToolRegistry()
        reg.register(
            name="echo",
            description="Echoes input",
            parameters_schema={"type": "object", "properties": {"text": {"type": "string"}}},
            handler=self._dummy_handler,
        )
        assert "echo" in reg
        assert len(reg) == 1

    def test_get_schemas_openai_format(self):
        reg = ToolRegistry()
        schema = {"type": "object", "properties": {"x": {"type": "integer"}}}
        reg.register(name="add", description="Adds numbers", parameters_schema=schema, handler=self._dummy_handler)

        schemas = reg.get_schemas()
        assert len(schemas) == 1
        assert schemas[0]["type"] == "function"
        assert schemas[0]["function"]["name"] == "add"
        assert schemas[0]["function"]["description"] == "Adds numbers"
        assert schemas[0]["function"]["parameters"] == schema

    @pytest.mark.asyncio
    async def test_execute_async_handler(self):
        reg = ToolRegistry()

        async def greet(name: str) -> str:
            return f"Hello, {name}!"

        reg.register(name="greet", description="Greet", parameters_schema={}, handler=greet)
        result = await reg.execute("greet", {"name": "World"})
        assert result == "Hello, World!"

    @pytest.mark.asyncio
    async def test_execute_unknown_tool(self):
        reg = ToolRegistry()
        result = await reg.execute("nonexistent", {})
        assert "Unknown tool" in result

    @pytest.mark.asyncio
    async def test_execute_handles_exception(self):
        reg = ToolRegistry()

        async def broken(**kwargs):
            raise ValueError("boom")

        reg.register(name="broken", description="Breaks", parameters_schema={}, handler=broken)
        result = await reg.execute("broken", {})
        assert "ValueError" in result
        assert "boom" in result

    @staticmethod
    async def _dummy_handler(**kwargs):
        return "ok"


# ===========================================================================
# Shell tool
# ===========================================================================


class TestShellTool:
    def test_execute_simple_command(self):
        result_str = execute_shell("echo hello", timeout=5)
        result = json.loads(result_str)
        assert result["stdout"].strip() == "hello"
        assert result["exit_code"] == 0

    def test_execute_captures_stderr(self):
        result_str = execute_shell("echo err >&2", timeout=5)
        result = json.loads(result_str)
        assert "err" in result["stderr"]

    def test_execute_nonzero_exit_code(self):
        result_str = execute_shell("exit 42", timeout=5)
        result = json.loads(result_str)
        assert result["exit_code"] == 42

    def test_timeout_handling(self):
        result_str = execute_shell("sleep 10", timeout=1)
        result = json.loads(result_str)
        assert "timed out" in result.get("error", "").lower()
        assert result["exit_code"] == -1

    def test_output_truncation(self):
        # Generate output larger than max_output_chars
        result_str = execute_shell("python3 -c \"print('A' * 200)\"", timeout=5, max_output_chars=50)
        result = json.loads(result_str)
        assert "truncated" in result["stdout"]


# ===========================================================================
# Files tool
# ===========================================================================


class TestFilesTool:
    @pytest.mark.asyncio
    async def test_write_and_read(self, tmp_path):
        fpath = str(tmp_path / "test.txt")
        allowed = [str(tmp_path)]

        # Write
        res = json.loads(await execute_files("write", fpath, content="hello world", allowed_paths=allowed))
        assert res["success"] is True
        assert res["bytes_written"] == 11

        # Read
        res = json.loads(await execute_files("read", fpath, allowed_paths=allowed))
        assert res["content"] == "hello world"
        assert res["truncated"] is False

    @pytest.mark.asyncio
    async def test_exists(self, tmp_path):
        fpath = str(tmp_path / "exists.txt")
        allowed = [str(tmp_path)]

        assert await execute_files("exists", fpath, allowed_paths=allowed) == "false"

        (tmp_path / "exists.txt").write_text("x")
        assert await execute_files("exists", fpath, allowed_paths=allowed) == "true"

    @pytest.mark.asyncio
    async def test_list(self, tmp_path):
        (tmp_path / "a.txt").write_text("a")
        (tmp_path / "b.txt").write_text("b")
        allowed = [str(tmp_path)]

        res = json.loads(await execute_files("list", str(tmp_path), allowed_paths=allowed))
        assert "a.txt" in res["entries"]
        assert "b.txt" in res["entries"]

    @pytest.mark.asyncio
    async def test_delete(self, tmp_path):
        fpath = str(tmp_path / "todelete.txt")
        (tmp_path / "todelete.txt").write_text("bye")
        allowed = [str(tmp_path)]

        res = json.loads(await execute_files("delete", fpath, allowed_paths=allowed))
        assert res["success"] is True
        assert not os.path.exists(fpath)

    @pytest.mark.asyncio
    async def test_append(self, tmp_path):
        fpath = str(tmp_path / "append.txt")
        allowed = [str(tmp_path)]

        await execute_files("write", fpath, content="first", allowed_paths=allowed)
        await execute_files("append", fpath, content=" second", allowed_paths=allowed)

        res = json.loads(await execute_files("read", fpath, allowed_paths=allowed))
        assert res["content"] == "first second"

    # -- Path traversal prevention --

    @pytest.mark.asyncio
    async def test_path_traversal_rejected(self, tmp_path):
        allowed = [str(tmp_path)]
        evil_path = str(tmp_path / ".." / ".." / "etc" / "passwd")

        res = json.loads(await execute_files("read", evil_path, allowed_paths=allowed))
        assert "error" in res
        assert "outside allowed" in res["error"].lower()

    @pytest.mark.asyncio
    async def test_absolute_path_outside_allowed_rejected(self, tmp_path):
        allowed = [str(tmp_path)]

        res = json.loads(await execute_files("read", "/etc/hostname", allowed_paths=allowed))
        assert "error" in res

    def test_check_path_allows_valid(self, tmp_path):
        allowed = [str(tmp_path)]
        fpath = str(tmp_path / "sub" / "file.txt")
        assert _check_path(fpath, allowed) is None

    def test_check_path_rejects_escape(self, tmp_path):
        allowed = [str(tmp_path)]
        err = _check_path("/tmp/evil.txt", allowed)
        assert err is not None
        assert "outside" in err.lower()

    def test_check_path_no_restriction(self):
        # When allowed_paths is empty, anything goes
        assert _check_path("/anywhere/file.txt", []) is None
        assert _check_path("/anywhere/file.txt", None) is None

    @pytest.mark.asyncio
    async def test_write_requires_content(self, tmp_path):
        allowed = [str(tmp_path)]
        fpath = str(tmp_path / "no_content.txt")
        res = json.loads(await execute_files("write", fpath, content=None, allowed_paths=allowed))
        assert "error" in res
        assert "content" in res["error"].lower()
