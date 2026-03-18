"""Coding tool — delegates coding tasks to the smart model.

Runs a synchronous sub-agent with shell + files access, returns the
result inline so the main agent can continue the conversation.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from spare_paw.router.tool_loop import run_tool_loop

logger = logging.getLogger(__name__)

_CODE_SYSTEM_PROMPT = """\
You are a senior software engineer. You write clean, correct, minimal code.

You have access to shell and file tools on an Android phone (Termux).
Use them to read existing code, write fixes, run tests, and verify your work.

Guidelines:
- Read the relevant code before making changes
- Make targeted fixes — don't refactor unrelated code
- Test your changes when possible (run the code, check for errors)
- If you create or modify files, state what you changed and why
- Be concise in your explanations
"""


async def _handle_code(app_state: Any, task: str) -> str:
    """Run a coding sub-agent with the smart model."""
    # Resolve smart model
    smart_model = (
        app_state.config.get("models.smart")
        or app_state.config.get("models.default", "google/gemini-2.0-flash")
    )

    # Filter tools to shell + files + any MCP tools
    all_schemas = app_state.tool_registry.get_schemas()
    allowed = {"execute_shell", "execute_files"}
    tool_schemas = [
        s for s in all_schemas
        if s.get("function", {}).get("name") in allowed
        or "." in s.get("function", {}).get("name", "")  # MCP namespaced tools
    ]

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": _CODE_SYSTEM_PROMPT},
        {"role": "user", "content": task},
    ]

    max_iterations = app_state.config.get("agent.max_tool_iterations", 20)

    try:
        result = await run_tool_loop(
            client=app_state.router_client,
            messages=messages,
            model=smart_model,
            tools=tool_schemas,
            tool_registry=app_state.tool_registry,
            max_iterations=max_iterations,
            executor=app_state.executor,
        )
        return json.dumps({"result": result, "model": smart_model})
    except Exception as exc:
        error_msg = f"{type(exc).__name__}: {exc}"
        logger.exception("Coding tool failed: %s", error_msg)
        return json.dumps({"error": error_msg})


CODE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "task": {
            "type": "string",
            "description": (
                "The coding task to perform. Include all relevant context: "
                "file paths, issue description, expected behavior, etc."
            ),
        },
    },
    "required": ["task"],
}


def register(registry: Any, config: dict[str, Any], app_state: Any) -> None:
    """Register the code tool."""

    async def _handler(task: str) -> str:
        return await _handle_code(app_state, task)

    registry.register(
        name="code",
        description=(
            "Delegate a coding task to a specialized agent running on the smart model. "
            "Use this for writing code, fixing bugs, debugging errors, and code reviews. "
            "Provide full context (file paths, issue details, expected behavior). "
            "The coding agent has access to shell, files, and MCP tools."
        ),
        parameters_schema=CODE_SCHEMA,
        handler=_handler,
        run_in_executor=False,
    )
