"""Shell command execution tool.

The handler is a **sync** function designed to run inside a
``ProcessPoolExecutor`` so that long-running commands never block the
async event loop.
"""

from __future__ import annotations

import json
import logging
import subprocess
from typing import TYPE_CHECKING, Any

from spare_paw.util.redact import redact_secrets

if TYPE_CHECKING:
    from spare_paw.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

# -- Schema ----------------------------------------------------------------

PARAMETERS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "command": {
            "type": "string",
            "description": "Shell command to execute",
        },
        "timeout": {
            "type": "integer",
            "description": "Timeout in seconds",
            "default": 30,
        },
    },
    "required": ["command"],
}

DESCRIPTION = (
    "Execute a shell command on the Android phone (Termux). "
    "Use termux-api commands for device interactions: termux-battery-status, "
    "termux-location, termux-camera-photo, termux-notification, termux-tts-speak, "
    "termux-sensor, etc. Use 'su -c' for root commands."
)

# -- Handler ---------------------------------------------------------------


def execute_shell(
    command: str,
    timeout: int = 30,
    max_output_chars: int = 10_000,
) -> str:
    """Run *command* via the system shell and return a JSON result string.

    This is a **synchronous** function — it will be dispatched to a
    ``ProcessPoolExecutor`` by the tool registry.
    """
    logger.info("shell: executing (timeout=%ds): %s", timeout, redact_secrets(command)[:200])

    try:
        proc = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        stdout = proc.stdout
        stderr = proc.stderr

        # Truncate oversized output.
        if len(stdout) > max_output_chars:
            stdout = stdout[:max_output_chars] + f"\n... [truncated at {max_output_chars} chars]"
        if len(stderr) > max_output_chars:
            stderr = stderr[:max_output_chars] + f"\n... [truncated at {max_output_chars} chars]"

        return json.dumps(
            {
                "stdout": stdout,
                "stderr": stderr,
                "exit_code": proc.returncode,
            }
        )

    except subprocess.TimeoutExpired:
        logger.warning("shell: command timed out after %ds: %s", timeout, redact_secrets(command)[:200])
        return json.dumps(
            {
                "error": f"Command timed out after {timeout}s",
                "exit_code": -1,
            }
        )

    except Exception as exc:  # noqa: BLE001
        logger.exception("shell: unexpected error")
        return json.dumps(
            {
                "error": f"{type(exc).__name__}: {exc}",
                "exit_code": -1,
            }
        )


# -- Registration ----------------------------------------------------------


def register(registry: ToolRegistry, config: dict[str, Any]) -> None:
    """Register the ``shell`` tool with *registry*."""
    # Use the top-level execute_shell directly — closures can't be
    # pickled for ProcessPoolExecutor.
    registry.register(
        name="shell",
        description=DESCRIPTION,
        parameters_schema=PARAMETERS_SCHEMA,
        handler=execute_shell,
        run_in_executor=True,
    )
