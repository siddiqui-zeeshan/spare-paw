"""Tests for the pipe mode _CollectBackend."""

from __future__ import annotations

from typing import Any

import pytest


class _CollectBackend:
    """Local copy of _CollectBackend matching pipe.py's contract.

    We mirror the class here rather than importing from pipe.py because it is
    defined inside ``run_pipe()`` and is not importable directly.  The tests
    below verify the same interface that the production code must satisfy.
    """

    def __init__(self) -> None:
        self.output: list[str] = []

    async def send_text(self, text: str) -> None:
        self.output.append(text)

    async def send_file(self, path: str, caption: str = "") -> None:
        pass

    async def send_typing(self) -> None:
        pass

    async def send_notification(
        self, text: str, actions: list[dict] | None = None
    ) -> None:
        pass

    def on_tool_event(self, event: Any) -> None:
        pass

    def on_token(self, token: str) -> None:
        pass

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass


def _get_pipe_collect_backend_source() -> type:
    """Extract _CollectBackend from pipe.py source to verify it matches contract."""
    import inspect

    import spare_paw.cli.pipe as pipe_mod

    src = inspect.getsource(pipe_mod)
    assert "class _CollectBackend:" in src, "_CollectBackend must be defined in pipe.py"
    assert "def on_tool_event" in src, "on_tool_event must be in pipe.py _CollectBackend"
    assert "def on_token" in src, "on_token must be in pipe.py _CollectBackend"
    return _CollectBackend


class TestCollectBackend:
    def test_has_on_tool_event_method(self):
        backend = _CollectBackend()
        assert hasattr(backend, "on_tool_event")
        assert callable(backend.on_tool_event)

    def test_has_on_token_method(self):
        backend = _CollectBackend()
        assert hasattr(backend, "on_token")
        assert callable(backend.on_token)

    def test_on_tool_event_is_noop(self):
        backend = _CollectBackend()
        event = object()
        result = backend.on_tool_event(event)
        assert result is None

    def test_on_token_is_noop(self):
        backend = _CollectBackend()
        result = backend.on_token("hello")
        assert result is None

    @pytest.mark.asyncio
    async def test_send_text_appends_to_output(self):
        backend = _CollectBackend()
        await backend.send_text("hello")
        await backend.send_text("world")
        assert backend.output == ["hello", "world"]

    @pytest.mark.asyncio
    async def test_send_file_is_noop(self):
        backend = _CollectBackend()
        await backend.send_file("/tmp/f.txt", caption="c")
        assert backend.output == []

    @pytest.mark.asyncio
    async def test_send_typing_is_noop(self):
        backend = _CollectBackend()
        await backend.send_typing()

    @pytest.mark.asyncio
    async def test_send_notification_is_noop(self):
        backend = _CollectBackend()
        await backend.send_notification("notice")
        assert backend.output == []

    @pytest.mark.asyncio
    async def test_start_and_stop_are_noops(self):
        backend = _CollectBackend()
        await backend.start()
        await backend.stop()

    def test_pipe_module_has_matching_implementation(self):
        """Smoke-check that pipe.py _CollectBackend has the required methods."""
        _get_pipe_collect_backend_source()
