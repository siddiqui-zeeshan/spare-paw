"""File operations tool.

Provides read, write, append, list, delete, and exists actions with
path-safety enforcement.
"""

from __future__ import annotations

import json
import logging
import os
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from spare_paw.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

# -- Schema ----------------------------------------------------------------

PARAMETERS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "action": {
            "type": "string",
            "enum": ["read", "write", "append", "list", "delete", "exists"],
            "description": "File operation to perform",
        },
        "path": {
            "type": "string",
            "description": "File or directory path",
        },
        "content": {
            "type": "string",
            "description": "Content for write/append actions",
        },
    },
    "required": ["action", "path"],
}

DESCRIPTION = (
    "Perform file operations (read, write, append, list, delete, exists) "
    "on the device filesystem. Paths are restricted to configured allowed directories."
)

MAX_READ_CHARS = 50_000

# -- Helpers ---------------------------------------------------------------


def _check_path(path: str, allowed_paths: list[str] | None) -> str | None:
    """Return an error string if *path* escapes *allowed_paths*, else ``None``."""
    if not allowed_paths:
        return None  # no restrictions configured

    real = os.path.realpath(path)
    for allowed in allowed_paths:
        allowed_real = os.path.realpath(allowed)
        # The resolved path must start with the allowed prefix (+ os.sep or be exact).
        if real == allowed_real or real.startswith(allowed_real + os.sep):
            return None

    return (
        f"Path '{path}' (resolved: '{real}') is outside allowed directories: "
        f"{allowed_paths}"
    )


# -- Handler ---------------------------------------------------------------


async def execute_files(
    action: str,
    path: str,
    content: str | None = None,
    allowed_paths: list[str] | None = None,
) -> str:
    """Execute a file operation and return a JSON result string."""
    logger.info("files: %s %s", action, path)

    # Path safety check
    err = _check_path(path, allowed_paths)
    if err is not None:
        logger.warning("files: path rejected — %s", err)
        return json.dumps({"error": err})

    try:
        if action == "read":
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                data = f.read(MAX_READ_CHARS + 1)
            truncated = len(data) > MAX_READ_CHARS
            if truncated:
                data = data[:MAX_READ_CHARS]
            return json.dumps(
                {"content": data, "truncated": truncated, "path": path}
            )

        elif action == "write":
            if content is None:
                return json.dumps({"error": "content is required for write action"})
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
            return json.dumps(
                {"success": True, "path": path, "bytes_written": len(content)}
            )

        elif action == "append":
            if content is None:
                return json.dumps({"error": "content is required for append action"})
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            with open(path, "a", encoding="utf-8") as f:
                f.write(content)
            return json.dumps(
                {"success": True, "path": path, "bytes_appended": len(content)}
            )

        elif action == "list":
            entries = sorted(os.listdir(path))
            return json.dumps({"path": path, "entries": entries})

        elif action == "delete":
            os.remove(path)
            return json.dumps({"success": True, "path": path})

        elif action == "exists":
            return "true" if os.path.exists(path) else "false"

        else:
            return json.dumps({"error": f"Unknown action: {action}"})

    except FileNotFoundError:
        return json.dumps({"error": f"File not found: {path}"})
    except PermissionError:
        return json.dumps({"error": f"Permission denied: {path}"})
    except IsADirectoryError:
        return json.dumps({"error": f"Is a directory: {path}"})
    except NotADirectoryError:
        return json.dumps({"error": f"Not a directory: {path}"})
    except OSError as exc:
        return json.dumps({"error": f"OS error: {exc}"})


# -- Registration ----------------------------------------------------------


def register(registry: ToolRegistry, config: dict[str, Any]) -> None:
    """Register the ``files`` tool with *registry*."""
    tool_cfg = config.get("tools", {}).get("files", {})
    allowed_paths: list[str] = tool_cfg.get("allowed_paths", [])

    async def _handler(
        action: str, path: str, content: str | None = None
    ) -> str:
        return await execute_files(
            action=action,
            path=path,
            content=content,
            allowed_paths=allowed_paths,
        )

    registry.register(
        name="files",
        description=DESCRIPTION,
        parameters_schema=PARAMETERS_SCHEMA,
        handler=_handler,
        run_in_executor=False,
    )
