"""Web scraping tool.

The handler is a **sync** function that runs in a
``ProcessPoolExecutor`` because BeautifulSoup HTML parsing is CPU-bound.
Uses ``urllib.request`` to avoid extra dependencies in the subprocess.
"""

from __future__ import annotations

import ipaddress
import json
import logging
import urllib.error
import urllib.parse
import urllib.request
from typing import TYPE_CHECKING, Any

from bs4 import BeautifulSoup

if TYPE_CHECKING:
    from spare_paw.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

# -- Schema ----------------------------------------------------------------

PARAMETERS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "url": {
            "type": "string",
            "description": "URL to fetch and scrape",
        },
        "selector": {
            "type": "string",
            "description": "Optional CSS selector to extract specific elements",
        },
        "timeout": {
            "type": "integer",
            "description": "Request timeout in seconds",
            "default": 15,
        },
    },
    "required": ["url"],
}

DESCRIPTION = (
    "Fetch and extract text from a specific URL. "
    "Use this when you already have a URL and need its content. "
    "Use web_search first if you need to find URLs."
)

# Tags whose content is noise — strip them before extracting text.
NOISE_TAGS = {"script", "style", "nav", "header", "footer", "noscript", "svg"}

USER_AGENT = (
    "Mozilla/5.0 (Linux; Android 14) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Mobile Safari/537.36"
)

# -- Handler ---------------------------------------------------------------


def execute_web_scrape(
    url: str,
    selector: str | None = None,
    timeout: int = 15,
    max_chars: int = 20_000,
) -> str:
    """Fetch *url*, parse HTML, and extract text.

    This is a **synchronous** function — dispatched to a
    ``ProcessPoolExecutor`` by the tool registry.
    """
    logger.info("web_scrape: fetching %s (selector=%s)", url, selector)

    # Security: block non-HTTP schemes and private/internal IPs
    try:
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme not in ("http", "https"):
            return json.dumps({"url": url, "error": f"Blocked scheme: {parsed.scheme}. Only http/https allowed."})
        hostname = parsed.hostname or ""
        try:
            addr = ipaddress.ip_address(hostname)
            if addr.is_private or addr.is_loopback or addr.is_link_local:
                return json.dumps({"url": url, "error": "Blocked: private/internal IP address"})
        except ValueError:
            # hostname is not an IP literal — check for localhost
            if hostname in ("localhost", "localhost.localdomain"):
                return json.dumps({"url": url, "error": "Blocked: localhost"})
    except Exception:
        pass  # Let urllib handle truly malformed URLs

    try:
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            # Read up to 5 MB of raw HTML to avoid memory issues.
            raw = resp.read(5 * 1024 * 1024)
            charset = resp.headers.get_content_charset() or "utf-8"
            html = raw.decode(charset, errors="replace")

        soup = BeautifulSoup(html, "html.parser")

        if selector:
            elements = soup.select(selector)
            if not elements:
                return json.dumps(
                    {
                        "url": url,
                        "content": "",
                        "length": 0,
                        "note": f"No elements matched selector: {selector}",
                    }
                )
            text = "\n\n".join(el.get_text(separator="\n", strip=True) for el in elements)
        else:
            # Remove noise tags.
            for tag in soup.find_all(NOISE_TAGS):
                tag.decompose()
            body = soup.find("body")
            target = body if body else soup
            text = target.get_text(separator="\n", strip=True)

        # Collapse excessive blank lines.
        lines = text.splitlines()
        collapsed: list[str] = []
        blank_count = 0
        for line in lines:
            stripped = line.strip()
            if not stripped:
                blank_count += 1
                if blank_count <= 1:
                    collapsed.append("")
            else:
                blank_count = 0
                collapsed.append(stripped)
        text = "\n".join(collapsed).strip()

        # Truncate.
        truncated = len(text) > max_chars
        if truncated:
            text = text[:max_chars] + f"\n... [truncated at {max_chars} chars]"

        return json.dumps(
            {
                "url": url,
                "content": text,
                "length": len(text),
                "truncated": truncated,
            }
        )

    except urllib.error.HTTPError as exc:
        return json.dumps(
            {"url": url, "error": f"HTTP {exc.code}: {exc.reason}"}
        )
    except urllib.error.URLError as exc:
        return json.dumps(
            {"url": url, "error": f"URL error: {exc.reason}"}
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("web_scrape: error fetching %s", url)
        return json.dumps(
            {"url": url, "error": f"{type(exc).__name__}: {exc}"}
        )


# -- Registration ----------------------------------------------------------


def register(registry: ToolRegistry, config: dict[str, Any]) -> None:
    """Register the ``web_scrape`` tool with *registry*."""
    # Use the top-level execute_web_scrape directly — closures can't be
    # pickled for ProcessPoolExecutor.
    registry.register(
        name="web_scrape",
        description=DESCRIPTION,
        parameters_schema=PARAMETERS_SCHEMA,
        handler=execute_web_scrape,
        run_in_executor=True,
    )
