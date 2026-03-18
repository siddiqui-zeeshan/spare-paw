"""Custom tool creation and management.

Allows the LLM to create shell-script-based tools that persist across restarts.
Tools are stored as shell scripts in ``~/.spare-paw/custom_tools/`` with
accompanying JSON metadata files.

The ``tool_create`` meta-tool lets the LLM propose a new tool, which requires
owner approval via Telegram before activation.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import stat
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from spare_paw.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

CUSTOM_TOOLS_DIR = Path.home() / ".spare-paw" / "custom_tools"
PENDING_DIR = CUSTOM_TOOLS_DIR / ".pending"

# ---------------------------------------------------------------------------
# Executor-compatible sync handler (top-level to avoid pickle errors)
# ---------------------------------------------------------------------------


def _execute_custom_tool(
    script_path: str,
    timeout: int = 30,
    max_output_chars: int = 10_000,
    **kwargs: Any,
) -> str:
    """Run a custom tool shell script and return JSON with stdout/stderr/exit_code.

    This is a **synchronous** function designed to run in a
    ``ProcessPoolExecutor``.  Parameters are passed to the script as
    environment variables.
    """
    env = os.environ.copy()
    for key, value in kwargs.items():
        env[f"TOOL_{key.upper()}"] = str(value)

    logger.info(
        "custom_tool: executing %s (timeout=%ds, params=%s)",
        script_path,
        timeout,
        list(kwargs.keys()),
    )

    try:
        proc = subprocess.run(
            ["bash", script_path],
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )

        stdout = proc.stdout
        stderr = proc.stderr

        if len(stdout) > max_output_chars:
            stdout = stdout[:max_output_chars] + f"\n... [truncated at {max_output_chars} chars]"
        if len(stderr) > max_output_chars:
            stderr = stderr[:max_output_chars] + f"\n... [truncated at {max_output_chars} chars]"

        return json.dumps({
            "stdout": stdout,
            "stderr": stderr,
            "exit_code": proc.returncode,
        })

    except subprocess.TimeoutExpired:
        logger.warning("custom_tool: timed out after %ds: %s", timeout, script_path)
        return json.dumps({
            "error": f"Script timed out after {timeout}s",
            "exit_code": -1,
        })

    except Exception as exc:  # noqa: BLE001
        logger.exception("custom_tool: unexpected error running %s", script_path)
        return json.dumps({
            "error": f"{type(exc).__name__}: {exc}",
            "exit_code": -1,
        })


# ---------------------------------------------------------------------------
# Loading custom tools at startup
# ---------------------------------------------------------------------------


def _register_custom_tool(
    registry: ToolRegistry,
    name: str,
    metadata: dict[str, Any],
    script_path: Path,
) -> None:
    """Register a single custom tool in the registry."""
    description = metadata.get("description", f"Custom tool: {name}")
    parameters_schema = metadata.get("parameters", {"type": "object", "properties": {}})
    timeout = metadata.get("timeout", 30)
    max_output = metadata.get("max_output_chars", 10_000)

    # Build a parameters schema that includes the optional timeout override
    full_schema = {
        "type": "object",
        "properties": {
            **parameters_schema.get("properties", {}),
            "timeout": {
                "type": "integer",
                "description": "Timeout in seconds (optional override)",
                "default": timeout,
            },
        },
        "required": parameters_schema.get("required", []),
    }

    # We need a top-level sync function with bound defaults for the executor.
    # Since _execute_custom_tool is already top-level, we use functools.partial
    # at call time via the registry. But the registry passes kwargs directly,
    # so we register a wrapper that fixes script_path and max_output.
    #
    # However, closures can't be pickled. Instead, we register the top-level
    # _execute_custom_tool with run_in_executor=True. The registry uses
    # functools.partial(**arguments) which works with ProcessPoolExecutor
    # as long as the base function is top-level.
    #
    # We need to inject script_path into the call. We do this by including it
    # as a fixed parameter in the schema handler. Since the model shouldn't
    # set script_path, we don't expose it in the schema — instead we use a
    # thin async wrapper that injects it.

    script_path_str = str(script_path)

    # Use an async wrapper that calls the executor manually
    async def _handler(**kwargs: Any) -> str:
        import asyncio
        import functools

        loop = asyncio.get_running_loop()
        # Extract timeout if provided, else use default
        t = kwargs.pop("timeout", timeout)
        fn = functools.partial(
            _execute_custom_tool,
            script_path=script_path_str,
            timeout=t,
            max_output_chars=max_output,
            **kwargs,
        )
        return await loop.run_in_executor(None, fn)

    registry.register(
        name=name,
        description=f"[custom] {description}",
        parameters_schema=full_schema,
        handler=_handler,
        run_in_executor=False,  # wrapper is async, manages executor itself
    )
    logger.info("Loaded custom tool: %s from %s", name, script_path)


def load_custom_tools(registry: ToolRegistry, executor: Any) -> None:
    """Scan ``~/.spare-paw/custom_tools/`` and register all active tools."""
    CUSTOM_TOOLS_DIR.mkdir(parents=True, exist_ok=True)
    PENDING_DIR.mkdir(parents=True, exist_ok=True)

    count = 0
    for meta_path in CUSTOM_TOOLS_DIR.glob("*.json"):
        name = meta_path.stem
        script_path = CUSTOM_TOOLS_DIR / f"{name}.sh"

        if not script_path.exists():
            logger.warning("Custom tool %s: metadata exists but script missing", name)
            continue

        try:
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.error("Custom tool %s: failed to read metadata: %s", name, exc)
            continue

        _register_custom_tool(registry, name, metadata, script_path)
        count += 1

    if count:
        logger.info("Loaded %d custom tool(s)", count)


# ---------------------------------------------------------------------------
# Approval logic (shared between meta-tool and /approve command)
# ---------------------------------------------------------------------------


async def approve_tool(
    name: str,
    registry: ToolRegistry,
    app_state: Any,
) -> str:
    """Move a pending tool to active and hot-register it.

    Returns a JSON string with the result.
    """
    pending_meta = PENDING_DIR / f"{name}.json"
    pending_script = PENDING_DIR / f"{name}.sh"

    if not pending_meta.exists() or not pending_script.exists():
        return json.dumps({"error": f"No pending tool found with name: {name}"})

    # Move files to active directory
    active_meta = CUSTOM_TOOLS_DIR / f"{name}.json"
    active_script = CUSTOM_TOOLS_DIR / f"{name}.sh"

    try:
        shutil.move(str(pending_meta), str(active_meta))
        shutil.move(str(pending_script), str(active_script))

        # Ensure script is executable
        active_script.chmod(active_script.stat().st_mode | stat.S_IEXEC)

        # Hot-register in the tool registry
        metadata = json.loads(active_meta.read_text(encoding="utf-8"))
        _register_custom_tool(registry, name, metadata, active_script)

        logger.info("Approved and activated custom tool: %s", name)
        return json.dumps({"success": True, "name": name, "status": "active"})

    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to approve custom tool: %s", name)
        return json.dumps({"error": f"Failed to approve tool: {exc}"})


# ---------------------------------------------------------------------------
# Meta-tool handlers
# ---------------------------------------------------------------------------


async def _handle_tool_create(
    app_state: Any,
    name: str,
    description: str,
    script: str,
    parameters: dict[str, Any] | None = None,
) -> str:
    """Propose a new custom tool. Saves to .pending and notifies the owner."""
    # Validate name (alphanumeric + underscores only)
    if not name.replace("_", "").isalnum():
        return json.dumps({
            "error": "Tool name must contain only alphanumeric characters and underscores"
        })

    # Check for conflicts with existing tools
    if name in app_state.tool_registry:
        return json.dumps({"error": f"A tool named '{name}' already exists"})

    # Check if already pending
    if (PENDING_DIR / f"{name}.json").exists():
        return json.dumps({"error": f"A tool named '{name}' is already pending approval"})

    PENDING_DIR.mkdir(parents=True, exist_ok=True)

    # Save metadata
    metadata = {
        "name": name,
        "description": description,
        "parameters": parameters or {"type": "object", "properties": {}},
    }
    meta_path = PENDING_DIR / f"{name}.json"
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    # Save script
    script_path = PENDING_DIR / f"{name}.sh"
    script_path.write_text(script, encoding="utf-8")
    script_path.chmod(script_path.stat().st_mode | stat.S_IEXEC)

    # Notify owner via Telegram
    try:
        owner_id = app_state.config.get("telegram.owner_id")
        if owner_id and app_state.application:
            bot = app_state.application.bot

            # Format parameters for display
            params_display = "(none)"
            if parameters and parameters.get("properties"):
                param_lines = []
                for pname, pschema in parameters["properties"].items():
                    ptype = pschema.get("type", "any")
                    pdesc = pschema.get("description", "")
                    param_lines.append(f"  - {pname} ({ptype}): {pdesc}")
                params_display = "\n".join(param_lines)

            # Truncate script display if very long
            script_display = script
            if len(script_display) > 2000:
                script_display = script_display[:2000] + "\n... [truncated]"

            msg = (
                f"New tool proposed: {name}\n\n"
                f"Description: {description}\n\n"
                f"Parameters:\n{params_display}\n\n"
                f"Script:\n```\n{script_display}\n```"
            )

            # Telegram has a 4096 char limit
            if len(msg) > 4096:
                msg = msg[:4090] + "\n..."

            # Send with inline approve/reject buttons
            from telegram import InlineKeyboardButton, InlineKeyboardMarkup
            keyboard = InlineKeyboardMarkup([
                [
                    InlineKeyboardButton("Approve", callback_data=f"approve:{name}"),
                    InlineKeyboardButton("Reject", callback_data=f"reject:{name}"),
                ]
            ])
            await bot.send_message(chat_id=owner_id, text=msg, reply_markup=keyboard)
    except Exception:  # noqa: BLE001
        logger.exception("Failed to send tool approval request to owner")

    return json.dumps({
        "status": "pending_approval",
        "name": name,
        "message": f"Tool '{name}' proposed. Waiting for owner approval via Telegram (/approve {name}).",
    })


async def _handle_tool_approve(
    app_state: Any,
    name: str,
) -> str:
    """Approve a pending custom tool."""
    return await approve_tool(name, app_state.tool_registry, app_state)


async def _handle_tool_list_custom(app_state: Any) -> str:
    """List all custom tools — both active and pending."""
    active: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []

    # Active tools
    for meta_path in CUSTOM_TOOLS_DIR.glob("*.json"):
        try:
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
            active.append({
                "name": meta_path.stem,
                "description": metadata.get("description", ""),
                "has_script": (CUSTOM_TOOLS_DIR / f"{meta_path.stem}.sh").exists(),
            })
        except (json.JSONDecodeError, OSError):
            active.append({"name": meta_path.stem, "error": "failed to read metadata"})

    # Pending tools
    if PENDING_DIR.exists():
        for meta_path in PENDING_DIR.glob("*.json"):
            try:
                metadata = json.loads(meta_path.read_text(encoding="utf-8"))
                pending.append({
                    "name": meta_path.stem,
                    "description": metadata.get("description", ""),
                    "has_script": (PENDING_DIR / f"{meta_path.stem}.sh").exists(),
                })
            except (json.JSONDecodeError, OSError):
                pending.append({"name": meta_path.stem, "error": "failed to read metadata"})

    return json.dumps({
        "active": active,
        "active_count": len(active),
        "pending": pending,
        "pending_count": len(pending),
    })


# ---------------------------------------------------------------------------
# Meta-tool schemas
# ---------------------------------------------------------------------------

TOOL_CREATE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "name": {
            "type": "string",
            "description": "Unique name for the tool (alphanumeric and underscores only)",
        },
        "description": {
            "type": "string",
            "description": "Human-readable description of what the tool does",
        },
        "script": {
            "type": "string",
            "description": "Bash script content for the tool",
        },
        "parameters": {
            "type": "object",
            "description": (
                "JSON Schema for the tool's parameters. Each parameter will be "
                "passed to the script as an environment variable named TOOL_<PARAM_NAME_UPPER>."
            ),
        },
    },
    "required": ["name", "description", "script"],
}

TOOL_APPROVE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "name": {
            "type": "string",
            "description": "Name of the pending tool to approve",
        },
    },
    "required": ["name"],
}

TOOL_LIST_CUSTOM_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {},
}


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def register_meta_tools(
    registry: ToolRegistry,
    config: dict[str, Any],
    app_state: Any,
) -> None:
    """Register ``tool_create``, ``tool_approve``, and ``tool_list_custom``."""

    # -- tool_create -------------------------------------------------------
    async def _create_handler(
        name: str,
        description: str,
        script: str,
        parameters: dict[str, Any] | None = None,
    ) -> str:
        return await _handle_tool_create(
            app_state, name=name, description=description,
            script=script, parameters=parameters,
        )

    registry.register(
        name="tool_create",
        description=(
            "Propose a new custom shell-script tool. The tool will be saved as "
            "pending and requires owner approval via Telegram before activation. "
            "Parameters are passed to the script as TOOL_<NAME> environment variables."
        ),
        parameters_schema=TOOL_CREATE_SCHEMA,
        handler=_create_handler,
        run_in_executor=False,
    )

    # tool_approve is NOT registered as an LLM-callable tool to prevent
    # the LLM from self-approving tools it creates. Approval is only
    # possible via the /approve Telegram command (owner-only).

    # -- tool_list_custom --------------------------------------------------
    async def _list_handler() -> str:
        return await _handle_tool_list_custom(app_state)

    registry.register(
        name="tool_list_custom",
        description="List all custom tools (both active and pending approval).",
        parameters_schema=TOOL_LIST_CUSTOM_SCHEMA,
        handler=_list_handler,
        run_in_executor=False,
    )
