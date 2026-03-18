"""Tavily Search API tool.

Uses aiohttp for async HTTP requests to the Tavily Search API.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

import aiohttp

if TYPE_CHECKING:
    from spare_paw.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

# -- Schema ----------------------------------------------------------------

PARAMETERS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": "Search query",
        },
        "count": {
            "type": "integer",
            "description": "Number of results to return",
            "default": 5,
        },
    },
    "required": ["query"],
}

DESCRIPTION = (
    "Search the web. Returns titles, URLs, and descriptions. "
    "Use this to find information or discover URLs. "
    "To read the content of a specific URL, use web_scrape instead."
)

TAVILY_SEARCH_URL = "https://api.tavily.com/search"

# -- Handler ---------------------------------------------------------------


async def execute_tavily_search(
    query: str,
    count: int = 5,
    api_key: str | None = None,
) -> str:
    """Search the web via Tavily Search API.

    Returns a JSON string with a list of ``{title, url, description}``
    objects, or an error message.
    """
    if not api_key:
        return json.dumps(
            {
                "error": (
                    "Tavily Search API key not configured. "
                    "Add your key to config.yaml under tavily.api_key"
                )
            }
        )

    logger.info("tavily_search: query=%r count=%d", query, count)

    payload = {
        "api_key": api_key,
        "query": query,
        "max_results": count,
        "search_depth": "basic",
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                TAVILY_SEARCH_URL,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=15),
            ) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    return json.dumps(
                        {
                            "error": f"Tavily Search API returned {resp.status}",
                            "details": body[:1000],
                        }
                    )

                data = await resp.json()

        results = [
            {
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "description": r.get("content", ""),
            }
            for r in data.get("results", [])
        ]

        return json.dumps({"results": results, "query": query})

    except aiohttp.ClientError as exc:
        msg = f"Tavily Search request failed: {type(exc).__name__}: {exc}"
        logger.exception(msg)
        return json.dumps({"error": msg})
    except Exception as exc:  # noqa: BLE001
        msg = f"Tavily Search unexpected error: {type(exc).__name__}: {exc}"
        logger.exception(msg)
        return json.dumps({"error": msg})


# -- Registration ----------------------------------------------------------


def register(registry: ToolRegistry, config: dict[str, Any]) -> None:
    """Register the ``web_search`` tool with *registry*."""
    api_key: str = config.get("tavily", {}).get("api_key", "")
    tool_cfg = config.get("tools", {}).get("web_search", {})
    default_count = tool_cfg.get("max_results", 5)

    async def _handler(query: str, count: int | None = None) -> str:
        return await execute_tavily_search(
            query=query,
            count=count if count is not None else default_count,
            api_key=api_key or None,
        )

    registry.register(
        name="web_search",
        description=DESCRIPTION,
        parameters_schema=PARAMETERS_SCHEMA,
        handler=_handler,
        run_in_executor=False,
    )
