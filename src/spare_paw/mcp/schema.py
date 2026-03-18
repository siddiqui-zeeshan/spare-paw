"""MCP <-> OpenAI schema conversion utilities.

Both MCP and OpenAI use JSON Schema for tool parameters, so conversion is
mostly structural reshuffling.
"""

from __future__ import annotations

from typing import Any


def mcp_to_openai_schema(mcp_tool: Any) -> dict[str, Any]:
    """Convert an MCP tool definition to OpenAI function-calling format.

    Parameters
    ----------
    mcp_tool:
        An MCP ``Tool`` object with ``name``, ``description``, and
        ``inputSchema`` attributes.

    Returns
    -------
    dict
        ``{"type": "function", "function": {"name": …, "description": …, "parameters": …}}``
    """
    return {
        "type": "function",
        "function": {
            "name": mcp_tool.name,
            "description": mcp_tool.description or "",
            "parameters": mcp_tool.inputSchema or {"type": "object", "properties": {}},
        },
    }


def extract_mcp_result(result: Any) -> str:
    """Extract text from an MCP ``CallToolResult``.

    Concatenates all text content blocks. Returns the error message if
    ``result.isError`` is set.
    """
    parts: list[str] = []
    for block in result.content:
        if hasattr(block, "text"):
            parts.append(block.text)
        elif hasattr(block, "data"):
            parts.append(f"[binary: {getattr(block, 'mimeType', 'unknown')}]")
        else:
            parts.append(str(block))

    text = "\n".join(parts)

    if getattr(result, "isError", False):
        return f"MCP error: {text}"

    return text
