"""Tests for browser automation tools (CDP)."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest

from spare_paw.tools.browser import (
    BrowserSession,
    _handle_click,
    _handle_eval_js,
    _handle_get_elements,
    _handle_get_text,
    _handle_navigate,
    _handle_screenshot,
    _handle_type,
    _handle_wait,
    register,
    shutdown,
)


# -- Helpers ---------------------------------------------------------------


def _make_session_mock() -> AsyncMock:
    """Create a mock BrowserSession with a working send()."""
    session = AsyncMock(spec=BrowserSession)
    session.ensure_connected = AsyncMock()
    session.send = AsyncMock()
    session.wait_for_event = AsyncMock()
    session.close = AsyncMock()
    return session


# -- BrowserSession unit tests --------------------------------------------


class TestBrowserSession:
    def test_singleton(self):
        BrowserSession._instance = None
        s1 = BrowserSession.get()
        s2 = BrowserSession.get()
        assert s1 is s2
        BrowserSession._instance = None

    @pytest.mark.asyncio
    async def test_close_clears_state(self):
        session = BrowserSession()
        session._process = None
        session._ws = None
        session._http_session = None
        session._recv_task = None
        await session.close()
        assert session._pending == {}
        assert session._events == {}


# -- Navigate --------------------------------------------------------------


class TestNavigate:
    @pytest.mark.asyncio
    async def test_navigate_success(self):
        mock_session = _make_session_mock()
        mock_session.send.side_effect = [
            {},  # Page.navigate
            {"result": {"value": "Test Page"}},  # title
            {"result": {"value": "Hello world"}},  # text
        ]

        with patch("spare_paw.tools.browser._ensure_session", return_value=mock_session):
            result = json.loads(await _handle_navigate("https://example.com"))

        assert result["status"] == "loaded"
        assert result["title"] == "Test Page"
        assert result["text_snippet"] == "Hello world"
        assert result["url"] == "https://example.com"

    @pytest.mark.asyncio
    async def test_navigate_error(self):
        with patch("spare_paw.tools.browser._ensure_session", side_effect=RuntimeError("no chrome")):
            result = json.loads(await _handle_navigate("https://example.com"))
        assert "error" in result


# -- Click -----------------------------------------------------------------


class TestClick:
    @pytest.mark.asyncio
    async def test_click_found(self):
        mock_session = _make_session_mock()
        mock_session.send.return_value = {
            "result": {"value": {"found": True, "tag": "BUTTON", "text": "Submit"}}
        }

        with patch("spare_paw.tools.browser._ensure_session", return_value=mock_session):
            result = json.loads(await _handle_click("button.submit"))

        assert result["clicked"] is True
        assert result["tag"] == "BUTTON"

    @pytest.mark.asyncio
    async def test_click_not_found(self):
        mock_session = _make_session_mock()
        mock_session.send.return_value = {
            "result": {"value": {"found": False}}
        }

        with patch("spare_paw.tools.browser._ensure_session", return_value=mock_session):
            result = json.loads(await _handle_click("button.missing"))

        assert "error" in result
        assert "not found" in result["error"].lower()


# -- Type ------------------------------------------------------------------


class TestType:
    @pytest.mark.asyncio
    async def test_type_success(self):
        mock_session = _make_session_mock()
        mock_session.send.return_value = {
            "result": {"value": {"found": True, "tag": "INPUT"}}
        }

        with patch("spare_paw.tools.browser._ensure_session", return_value=mock_session):
            result = json.loads(await _handle_type("input#email", "test@test.com"))

        assert result["typed"] is True
        assert result["text"] == "test@test.com"

    @pytest.mark.asyncio
    async def test_type_not_found(self):
        mock_session = _make_session_mock()
        mock_session.send.return_value = {
            "result": {"value": {"found": False}}
        }

        with patch("spare_paw.tools.browser._ensure_session", return_value=mock_session):
            result = json.loads(await _handle_type("input.missing", "text"))

        assert "error" in result


# -- Screenshot ------------------------------------------------------------


class TestScreenshot:
    @pytest.mark.asyncio
    async def test_screenshot_success(self, tmp_path):
        mock_session = _make_session_mock()
        # Small 1x1 red PNG
        import base64

        png_data = base64.b64encode(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100).decode()
        mock_session.send.return_value = {"data": png_data}

        with (
            patch("spare_paw.tools.browser._ensure_session", return_value=mock_session),
            patch("spare_paw.tools.browser.SCREENSHOT_DIR", tmp_path),
        ):
            result = json.loads(await _handle_screenshot())

        assert "path" in result
        assert result["size_bytes"] > 0


# -- GetText ---------------------------------------------------------------


class TestGetText:
    @pytest.mark.asyncio
    async def test_get_text_full_page(self):
        mock_session = _make_session_mock()
        mock_session.send.return_value = {
            "result": {"value": "Page content here"}
        }

        with patch("spare_paw.tools.browser._ensure_session", return_value=mock_session):
            result = json.loads(await _handle_get_text())

        assert result["text"] == "Page content here"
        assert result["truncated"] is False

    @pytest.mark.asyncio
    async def test_get_text_with_selector(self):
        mock_session = _make_session_mock()
        mock_session.send.return_value = {
            "result": {"value": "Section text"}
        }

        with patch("spare_paw.tools.browser._ensure_session", return_value=mock_session):
            result = json.loads(await _handle_get_text(selector="div.content"))

        assert result["text"] == "Section text"


# -- EvalJs ----------------------------------------------------------------


class TestEvalJs:
    @pytest.mark.asyncio
    async def test_eval_success(self):
        mock_session = _make_session_mock()
        mock_session.send.return_value = {
            "result": {"value": 42, "type": "number"}
        }

        with patch("spare_paw.tools.browser._ensure_session", return_value=mock_session):
            result = json.loads(await _handle_eval_js("1 + 41"))

        assert result["result"] == 42
        assert result["type"] == "number"

    @pytest.mark.asyncio
    async def test_eval_js_error(self):
        mock_session = _make_session_mock()
        mock_session.send.return_value = {
            "result": {"subtype": "error", "description": "ReferenceError: x is not defined"}
        }

        with patch("spare_paw.tools.browser._ensure_session", return_value=mock_session):
            result = json.loads(await _handle_eval_js("x"))

        assert "error" in result


# -- GetElements -----------------------------------------------------------


class TestGetElements:
    @pytest.mark.asyncio
    async def test_get_elements(self):
        mock_session = _make_session_mock()
        mock_session.send.return_value = {
            "result": {
                "value": [
                    {"index": 0, "tag": "a", "text": "Link 1", "href": "https://a.com"},
                    {"index": 1, "tag": "a", "text": "Link 2", "href": "https://b.com"},
                ]
            }
        }

        with patch("spare_paw.tools.browser._ensure_session", return_value=mock_session):
            result = json.loads(await _handle_get_elements("a"))

        assert result["count"] == 2
        assert result["elements"][0]["tag"] == "a"


# -- Wait ------------------------------------------------------------------


class TestWait:
    @pytest.mark.asyncio
    async def test_wait_found_immediately(self):
        mock_session = _make_session_mock()
        mock_session.send.return_value = {
            "result": {"value": True}
        }

        with patch("spare_paw.tools.browser._ensure_session", return_value=mock_session):
            result = json.loads(await _handle_wait("div.loaded"))

        assert result["found"] is True
        assert "elapsed_ms" in result

    @pytest.mark.asyncio
    async def test_wait_timeout(self):
        mock_session = _make_session_mock()
        mock_session.send.return_value = {
            "result": {"value": False}
        }

        with patch("spare_paw.tools.browser._ensure_session", return_value=mock_session):
            result = json.loads(await _handle_wait("div.never", timeout=1))

        assert "error" in result
        assert "Timeout" in result["error"]


# -- Registration ----------------------------------------------------------


class TestRegistration:
    def test_register_all_tools(self):
        from spare_paw.tools.registry import ToolRegistry

        registry = ToolRegistry()
        register(registry, {"tools": {"browser": {"enabled": True}}})

        expected = {
            "browser_navigate",
            "browser_click",
            "browser_type",
            "browser_screenshot",
            "browser_get_text",
            "browser_eval_js",
            "browser_get_elements",
            "browser_wait",
        }
        registered = {name for name in expected if name in registry}
        assert registered == expected

    def test_register_disabled(self):
        from spare_paw.tools.registry import ToolRegistry

        registry = ToolRegistry()
        register(registry, {"tools": {"browser": {"enabled": False}}})
        assert len(registry) == 0

    def test_config_key_mapping(self):
        from spare_paw.tools.registry import ToolRegistry

        registry = ToolRegistry()
        register(registry, {})
        enabled = registry.get_enabled_tools({
            "tools": {"browser": {"enabled": False}}
        })
        assert not any(n.startswith("browser_") for n in enabled)


# -- Shutdown --------------------------------------------------------------


class TestShutdown:
    @pytest.mark.asyncio
    async def test_shutdown_no_session(self):
        BrowserSession._instance = None
        await shutdown()  # Should not raise

    @pytest.mark.asyncio
    async def test_shutdown_with_session(self):
        mock = _make_session_mock()
        BrowserSession._instance = mock
        await shutdown()
        mock.close.assert_awaited_once()
        assert BrowserSession._instance is None
