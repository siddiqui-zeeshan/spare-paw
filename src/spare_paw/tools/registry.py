"""Tool registry — manages tool registration, JSON schema generation, and execution."""

from __future__ import annotations

import asyncio
import logging
from concurrent.futures import Executor
from dataclasses import dataclass
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass
class _ToolEntry:
    name: str
    description: str
    parameters_schema: dict[str, Any]
    handler: Callable
    run_in_executor: bool


class ToolRegistry:
    """Central registry for all agent tools.

    Handles registration, OpenAI-format schema generation, and safe
    execution (errors are returned as strings, never raised).
    """

    def __init__(self) -> None:
        self._tools: dict[str, _ToolEntry] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(
        self,
        name: str,
        description: str,
        parameters_schema: dict[str, Any],
        handler: Callable,
        run_in_executor: bool = False,
    ) -> None:
        """Register a tool.

        Parameters
        ----------
        name:
            Unique tool name (used in function-calling).
        description:
            Human-readable description shown to the model.
        parameters_schema:
            JSON Schema dict describing the tool's parameters.
        handler:
            Async callable (or sync if *run_in_executor* is True).
        run_in_executor:
            If True the handler is a **sync** function and will be
            dispatched to a ``ProcessPoolExecutor`` via
            ``loop.run_in_executor``.
        """
        if name in self._tools:
            logger.warning("Tool %r already registered — overwriting", name)
        self._tools[name] = _ToolEntry(
            name=name,
            description=description,
            parameters_schema=parameters_schema,
            handler=handler,
            run_in_executor=run_in_executor,
        )
        logger.info("Registered tool: %s (executor=%s)", name, run_in_executor)

    # ------------------------------------------------------------------
    # Schema generation
    # ------------------------------------------------------------------

    def get_schemas(self) -> list[dict[str, Any]]:
        """Return OpenAI function-calling format tool schemas.

        Each entry:
        ``{"type": "function", "function": {"name": …, "description": …, "parameters": …}}``
        """
        schemas: list[dict[str, Any]] = []
        for entry in self._tools.values():
            schemas.append(
                {
                    "type": "function",
                    "function": {
                        "name": entry.name,
                        "description": entry.description,
                        "parameters": entry.parameters_schema,
                    },
                }
            )
        return schemas

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    async def execute(
        self,
        name: str,
        arguments: dict[str, Any],
        executor: Executor | None = None,
    ) -> str:
        """Execute a tool by *name* with the given *arguments*.

        If the tool was registered with ``run_in_executor=True`` and an
        *executor* is provided, the handler runs in that executor (typically
        a ``ProcessPoolExecutor``).

        Returns
        -------
        str
            The result as a string.  On any error an error message is
            returned — this method **never raises** so that tool errors
            are model-recoverable.
        """
        entry = self._tools.get(name)
        if entry is None:
            msg = f"Unknown tool: {name}"
            logger.error(msg)
            return msg

        try:
            if entry.run_in_executor:
                loop = asyncio.get_running_loop()
                import functools

                fn = functools.partial(entry.handler, **arguments)
                result = await loop.run_in_executor(executor, fn)
            else:
                result = await entry.handler(**arguments)

            return str(result)

        except Exception as exc:  # noqa: BLE001
            msg = f"Tool {name} failed: {type(exc).__name__}: {exc}"
            logger.exception(msg)
            return msg

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def get_enabled_tools(self, config: dict[str, Any]) -> list[str]:
        """Return names of tools that are enabled in *config*.

        Config layout expected::

            tools:
              shell:
                enabled: true
              files:
                enabled: false
              ...

        Tools without a ``tools.<name>.enabled`` entry default to
        **enabled**.  Cron tools use the key ``cron``.
        """
        tools_cfg = config.get("tools", {})
        enabled: list[str] = []
        for name in self._tools:
            # cron_create / cron_delete / cron_list share the "cron" key
            cfg_key = "cron" if name.startswith("cron_") else name
            tool_cfg = tools_cfg.get(cfg_key, {})
            if tool_cfg.get("enabled", True):
                enabled.append(name)
        return enabled

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)
