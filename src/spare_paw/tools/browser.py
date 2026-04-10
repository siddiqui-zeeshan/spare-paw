"""Browser automation tools via Chrome DevTools Protocol (CDP).

Provides composable browser tools the agent can chain for any web task:
navigate, click, type, screenshot, get text, evaluate JS, list elements, wait.

Uses raw WebSocket + CDP over aiohttp (no Playwright dependency).
Chromium is launched lazily on first use and persists across tool calls.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import shutil
import subprocess
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import aiohttp

if TYPE_CHECKING:
    from spare_paw.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

SCREENSHOT_DIR = Path("/tmp/spare-paw-screenshots")


# -- CDP Session -----------------------------------------------------------


class BrowserSession:
    """Persistent CDP connection to a headless Chromium instance."""

    _instance: BrowserSession | None = None

    def __init__(self) -> None:
        self._process: subprocess.Popen | None = None
        self._http_session: aiohttp.ClientSession | None = None
        self._ws: aiohttp.ClientWebSocketResponse | None = None
        self._msg_id = 0
        self._pending: dict[int, asyncio.Future] = {}
        self._events: dict[str, asyncio.Event] = {}
        self._recv_task: asyncio.Task | None = None
        self._ws_url: str | None = None
        self._chromium_cmd: str | None = None

    @classmethod
    def get(cls) -> BrowserSession:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    async def ensure_connected(self) -> None:
        """Launch Chromium and connect if not already running."""
        if self._ws is not None and not self._ws.closed:
            return

        # Find Chromium binary
        if self._chromium_cmd is None:
            for cmd in ("chromium-browser", "chromium", "google-chrome", "google-chrome-stable"):
                if shutil.which(cmd):
                    self._chromium_cmd = cmd
                    break
            if self._chromium_cmd is None:
                raise RuntimeError(
                    "Chromium not found. Install it: sudo apt install chromium-browser"
                )

        # Kill any stale process
        if self._process is not None:
            try:
                self._process.terminate()
                self._process.wait(timeout=5)
            except Exception:
                self._process.kill()
            self._process = None

        # Launch headless Chromium
        port = _find_free_port()
        args = [
            self._chromium_cmd,
            "--headless",
            "--disable-gpu",
            "--no-sandbox",
            "--disable-dev-shm-usage",
            f"--remote-debugging-port={port}",
            "--window-size=1280,720",
            "--disable-extensions",
            "--disable-background-networking",
            "about:blank",
        ]
        self._process = subprocess.Popen(
            args,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

        # Wait for DevTools to be ready
        cdp_base = f"http://127.0.0.1:{port}"
        ws_url = await self._wait_for_devtools(cdp_base, timeout=15)
        self._ws_url = ws_url

        # Connect WebSocket
        if self._http_session is None or self._http_session.closed:
            self._http_session = aiohttp.ClientSession()

        self._ws = await self._http_session.ws_connect(
            ws_url, max_msg_size=50 * 1024 * 1024
        )
        self._recv_task = asyncio.create_task(self._recv_loop(), name="cdp-recv")

        # Enable required domains
        await self.send("Page.enable")
        await self.send("Runtime.enable")
        await self.send("DOM.enable")
        logger.info("Browser session connected: %s", ws_url)

    async def _wait_for_devtools(self, cdp_base: str, timeout: float) -> str:
        """Poll the CDP /json endpoint until it returns a page target."""
        deadline = time.monotonic() + timeout
        last_err = None
        while time.monotonic() < deadline:
            try:
                async with aiohttp.ClientSession() as s:
                    async with s.get(f"{cdp_base}/json", timeout=aiohttp.ClientTimeout(total=2)) as resp:
                        targets = await resp.json()
                        for t in targets:
                            if t.get("type") == "page" and "webSocketDebuggerUrl" in t:
                                return t["webSocketDebuggerUrl"]
            except Exception as e:
                last_err = e
            await asyncio.sleep(0.5)
        raise RuntimeError(f"Chromium DevTools not ready after {timeout}s: {last_err}")

    async def send(self, method: str, params: dict | None = None, timeout: float = 30) -> dict:
        """Send a CDP command and await the response."""
        if self._ws is None or self._ws.closed:
            raise RuntimeError("WebSocket not connected")

        self._msg_id += 1
        msg_id = self._msg_id
        msg: dict[str, Any] = {"id": msg_id, "method": method}
        if params:
            msg["params"] = params

        future: asyncio.Future = asyncio.get_running_loop().create_future()
        self._pending[msg_id] = future

        await self._ws.send_json(msg)

        try:
            result = await asyncio.wait_for(future, timeout=timeout)
        except asyncio.TimeoutError:
            self._pending.pop(msg_id, None)
            raise RuntimeError(f"CDP timeout for {method} after {timeout}s")

        if "error" in result:
            raise RuntimeError(f"CDP error: {result['error']}")
        return result.get("result", {})

    async def wait_for_event(self, event_name: str, timeout: float = 30) -> None:
        """Wait for a specific CDP event."""
        evt = self._events.setdefault(event_name, asyncio.Event())
        evt.clear()
        try:
            await asyncio.wait_for(evt.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            pass  # Non-fatal — page may have already loaded

    async def _recv_loop(self) -> None:
        """Background task dispatching incoming CDP messages."""
        try:
            async for msg in self._ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    data = json.loads(msg.data)
                    msg_id = data.get("id")
                    if msg_id is not None and msg_id in self._pending:
                        self._pending.pop(msg_id).set_result(data)
                    elif "method" in data:
                        evt = self._events.get(data["method"])
                        if evt is not None:
                            evt.set()
                elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                    break
        except Exception:
            logger.debug("CDP recv loop ended", exc_info=True)

    async def close(self) -> None:
        """Shut down browser and clean up."""
        if self._recv_task is not None:
            self._recv_task.cancel()
            self._recv_task = None

        if self._ws is not None and not self._ws.closed:
            await self._ws.close()
            self._ws = None

        if self._http_session is not None and not self._http_session.closed:
            await self._http_session.close()
            self._http_session = None

        if self._process is not None:
            try:
                self._process.terminate()
                self._process.wait(timeout=5)
            except Exception:
                self._process.kill()
            self._process = None

        self._pending.clear()
        self._events.clear()
        logger.info("Browser session closed")


def _find_free_port() -> int:
    """Find an available TCP port."""
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


async def _ensure_session() -> BrowserSession:
    """Get or create the browser session."""
    session = BrowserSession.get()
    await session.ensure_connected()
    return session


# -- Tool Handlers ---------------------------------------------------------


async def _handle_navigate(url: str) -> str:
    try:
        session = await _ensure_session()
        await session.send("Page.navigate", {"url": url})
        await session.wait_for_event("Page.loadEventFired", timeout=30)
        # Small extra wait for JS rendering
        await asyncio.sleep(1)

        title = await session.send(
            "Runtime.evaluate", {"expression": "document.title"}
        )
        text = await session.send(
            "Runtime.evaluate",
            {"expression": "document.body?.innerText?.substring(0, 3000) || ''"},
        )
        return json.dumps({
            "url": url,
            "title": title.get("result", {}).get("value", ""),
            "text_snippet": text.get("result", {}).get("value", ""),
            "status": "loaded",
        })
    except Exception as e:
        return json.dumps({"error": f"navigate failed: {e}"})


async def _handle_click(selector: str) -> str:
    try:
        session = await _ensure_session()
        result = await session.send(
            "Runtime.evaluate",
            {
                "expression": f"""
                    (() => {{
                        const el = document.querySelector({json.dumps(selector)});
                        if (!el) return {{found: false}};
                        el.scrollIntoView({{block: 'center'}});
                        el.click();
                        return {{found: true, tag: el.tagName, text: el.innerText?.substring(0, 100)}};
                    }})()
                """,
                "returnByValue": True,
            },
        )
        val = result.get("result", {}).get("value", {})
        if not val.get("found"):
            return json.dumps({"error": f"Element not found: {selector}"})
        return json.dumps({"selector": selector, "clicked": True, **val})
    except Exception as e:
        return json.dumps({"error": f"click failed: {e}"})


async def _handle_type(selector: str, text: str, clear: bool = True) -> str:
    try:
        session = await _ensure_session()
        clear_js = "el.value = '';" if clear else ""
        result = await session.send(
            "Runtime.evaluate",
            {
                "expression": f"""
                    (() => {{
                        const el = document.querySelector({json.dumps(selector)});
                        if (!el) return {{found: false}};
                        el.focus();
                        {clear_js}
                        el.value = {json.dumps(text)};
                        el.dispatchEvent(new Event('input', {{bubbles: true}}));
                        el.dispatchEvent(new Event('change', {{bubbles: true}}));
                        return {{found: true, tag: el.tagName}};
                    }})()
                """,
                "returnByValue": True,
            },
        )
        val = result.get("result", {}).get("value", {})
        if not val.get("found"):
            return json.dumps({"error": f"Element not found: {selector}"})
        return json.dumps({"selector": selector, "typed": True, "text": text})
    except Exception as e:
        return json.dumps({"error": f"type failed: {e}"})


async def _handle_screenshot() -> str:
    try:
        session = await _ensure_session()
        result = await session.send(
            "Page.captureScreenshot", {"format": "png"}
        )
        data = base64.b64decode(result["data"])

        SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
        ts = int(time.time() * 1000)
        path = SCREENSHOT_DIR / f"screenshot_{ts}.png"
        path.write_bytes(data)

        return json.dumps({
            "path": str(path),
            "size_bytes": len(data),
        })
    except Exception as e:
        return json.dumps({"error": f"screenshot failed: {e}"})


async def _handle_get_text(selector: str | None = None) -> str:
    try:
        session = await _ensure_session()
        if selector:
            expr = f"document.querySelector({json.dumps(selector)})?.innerText || ''"
        else:
            expr = "document.body?.innerText || ''"

        result = await session.send(
            "Runtime.evaluate", {"expression": expr}
        )
        text = result.get("result", {}).get("value", "")
        max_chars = 20_000
        truncated = len(text) > max_chars
        if truncated:
            text = text[:max_chars]
        return json.dumps({
            "text": text,
            "length": len(text),
            "truncated": truncated,
        })
    except Exception as e:
        return json.dumps({"error": f"get_text failed: {e}"})


async def _handle_eval_js(code: str) -> str:
    try:
        session = await _ensure_session()
        result = await session.send(
            "Runtime.evaluate",
            {"expression": code, "returnByValue": True},
        )
        res = result.get("result", {})
        if res.get("subtype") == "error" or result.get("exceptionDetails"):
            desc = res.get("description", str(result.get("exceptionDetails", "")))
            return json.dumps({"error": f"JS error: {desc}"})
        val = res.get("value", res.get("description", ""))
        return json.dumps({"result": val, "type": res.get("type", "unknown")})
    except Exception as e:
        return json.dumps({"error": f"eval_js failed: {e}"})


async def _handle_get_elements(selector: str) -> str:
    try:
        session = await _ensure_session()
        result = await session.send(
            "Runtime.evaluate",
            {
                "expression": f"""
                    (() => {{
                        const els = document.querySelectorAll({json.dumps(selector)});
                        return Array.from(els).slice(0, 50).map((el, i) => ({{
                            index: i,
                            tag: el.tagName.toLowerCase(),
                            text: (el.innerText || '').substring(0, 200),
                            id: el.id || undefined,
                            className: el.className || undefined,
                            href: el.href || undefined,
                            value: el.value || undefined,
                            type: el.type || undefined,
                        }}));
                    }})()
                """,
                "returnByValue": True,
            },
        )
        elements = result.get("result", {}).get("value", [])
        return json.dumps({
            "selector": selector,
            "count": len(elements),
            "elements": elements,
        })
    except Exception as e:
        return json.dumps({"error": f"get_elements failed: {e}"})


async def _handle_wait(selector: str, timeout: int = 10) -> str:
    try:
        session = await _ensure_session()
        start = time.monotonic()
        deadline = start + timeout
        while time.monotonic() < deadline:
            result = await session.send(
                "Runtime.evaluate",
                {
                    "expression": f"!!document.querySelector({json.dumps(selector)})",
                    "returnByValue": True,
                },
            )
            if result.get("result", {}).get("value"):
                elapsed = int((time.monotonic() - start) * 1000)
                return json.dumps({
                    "selector": selector,
                    "found": True,
                    "elapsed_ms": elapsed,
                })
            await asyncio.sleep(0.5)
        return json.dumps({"error": f"Timeout waiting for: {selector}"})
    except Exception as e:
        return json.dumps({"error": f"wait failed: {e}"})


# -- Shutdown --------------------------------------------------------------

async def shutdown() -> None:
    """Close the browser session (called during gateway shutdown)."""
    if BrowserSession._instance is not None:
        await BrowserSession._instance.close()
        BrowserSession._instance = None


# -- Registration ----------------------------------------------------------

# Tool schemas

_NAVIGATE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "url": {"type": "string", "description": "URL to navigate to"},
    },
    "required": ["url"],
}

_CLICK_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "selector": {
            "type": "string",
            "description": "CSS selector of the element to click",
        },
    },
    "required": ["selector"],
}

_TYPE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "selector": {
            "type": "string",
            "description": "CSS selector of the input field",
        },
        "text": {
            "type": "string",
            "description": "Text to type into the field",
        },
        "clear": {
            "type": "boolean",
            "description": "Clear existing value before typing (default: true)",
            "default": True,
        },
    },
    "required": ["selector", "text"],
}

_SCREENSHOT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {},
}

_GET_TEXT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "selector": {
            "type": "string",
            "description": "Optional CSS selector. If omitted, returns all visible text on the page.",
        },
    },
}

_EVAL_JS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "code": {
            "type": "string",
            "description": "JavaScript code to evaluate in the page context",
        },
    },
    "required": ["code"],
}

_GET_ELEMENTS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "selector": {
            "type": "string",
            "description": "CSS selector to match elements (returns up to 50)",
        },
    },
    "required": ["selector"],
}

_WAIT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "selector": {
            "type": "string",
            "description": "CSS selector to wait for",
        },
        "timeout": {
            "type": "integer",
            "description": "Max seconds to wait (default: 10)",
            "default": 10,
        },
    },
    "required": ["selector"],
}

_TOOLS = [
    (
        "browser_navigate",
        "Navigate to a URL in the headless browser. Returns the page title and a text snippet. "
        "Use this for pages that need JavaScript rendering or interaction — for simple fetches, prefer web_scrape.",
        _NAVIGATE_SCHEMA,
        _handle_navigate,
    ),
    (
        "browser_click",
        "Click an element on the current page by CSS selector.",
        _CLICK_SCHEMA,
        _handle_click,
    ),
    (
        "browser_type",
        "Type text into an input field on the current page by CSS selector.",
        _TYPE_SCHEMA,
        _handle_type,
    ),
    (
        "browser_screenshot",
        "Take a screenshot of the current page. Returns the file path. "
        "Use send_file to send the screenshot to the user.",
        _SCREENSHOT_SCHEMA,
        _handle_screenshot,
    ),
    (
        "browser_get_text",
        "Get visible text from the current page, optionally filtered by CSS selector.",
        _GET_TEXT_SCHEMA,
        _handle_get_text,
    ),
    (
        "browser_eval_js",
        "Evaluate JavaScript code in the current page context and return the result.",
        _EVAL_JS_SCHEMA,
        _handle_eval_js,
    ),
    (
        "browser_get_elements",
        "List elements matching a CSS selector with their tag, text, id, class, href, and value. "
        "Useful for discovering page structure before clicking or typing.",
        _GET_ELEMENTS_SCHEMA,
        _handle_get_elements,
    ),
    (
        "browser_wait",
        "Wait for an element matching a CSS selector to appear on the page. "
        "Use after navigation or clicks that trigger dynamic content loading.",
        _WAIT_SCHEMA,
        _handle_wait,
    ),
]


def register(registry: ToolRegistry, config: dict[str, Any]) -> None:
    """Register all browser automation tools."""
    browser_cfg = config.get("tools", {}).get("browser", {})
    if not browser_cfg.get("enabled", True):
        logger.info("Browser tools disabled in config")
        return

    for name, description, schema, handler in _TOOLS:
        registry.register(
            name=name,
            description=description,
            parameters_schema=schema,
            handler=handler,
            run_in_executor=False,
        )
