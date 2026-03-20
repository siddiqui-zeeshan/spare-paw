"""Main async entry point: Telegram bot + cron scheduler + heartbeat.

Orchestrates startup, wires all components together, and runs until stopped.
"""

from __future__ import annotations

import asyncio
import logging
import signal
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

from spare_paw.config import Config, config
from spare_paw.db import close_db, init_db
from spare_paw.util.redact import redact_secrets

logger = logging.getLogger(__name__)

LOG_DIR = Path.home() / ".spare-paw" / "logs"
HEARTBEAT_PATH = Path.home() / ".spare-paw" / "heartbeat"


@dataclass
class AppState:
    """Shared application state passed to handlers and subsystems."""

    config: Config
    executor: ProcessPoolExecutor
    semaphore: asyncio.Semaphore
    tool_registry: Any = None
    router_client: Any = None
    backend: Any = None
    webhook: Any = None
    _application: Any = None
    scheduler: Any = None
    mcp_client: Any = None
    start_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def application(self) -> Any:
        """Deprecated: use backend instead. Returns the underlying Application."""
        warnings.warn(
            "AppState.application is deprecated, use AppState.backend",
            DeprecationWarning,
            stacklevel=2,
        )
        if self.backend is not None:
            return self.backend._application
        return self._application

    @application.setter
    def application(self, value: Any) -> None:
        self._application = value


# Module-level singleton so other modules can access shared state
app_state: AppState | None = None


class _RedactingFormatter(logging.Formatter):
    """Logging formatter that redacts secrets from every log message."""

    def format(self, record: logging.LogRecord) -> str:
        return redact_secrets(super().format(record))


def _setup_logging() -> None:
    """Configure rotating file + stderr logging."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    level_name = config.get("logging.level", "INFO")
    level = getattr(logging, level_name.upper(), logging.INFO)
    max_bytes = config.get("logging.max_bytes", 10485760)
    backup_count = config.get("logging.backup_count", 3)

    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Clear existing handlers to avoid duplicates on reload
    root_logger.handlers.clear()

    formatter = _RedactingFormatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Rotating file handler
    file_handler = RotatingFileHandler(
        LOG_DIR / "spare-paw.log",
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    # Stderr handler
    stderr_handler = logging.StreamHandler()
    stderr_handler.setLevel(level)
    stderr_handler.setFormatter(formatter)
    root_logger.addHandler(stderr_handler)


async def _heartbeat() -> None:
    """Touch the heartbeat file every 30 seconds."""
    while True:
        try:
            HEARTBEAT_PATH.parent.mkdir(parents=True, exist_ok=True)
            HEARTBEAT_PATH.write_text(
                datetime.now(timezone.utc).isoformat(), encoding="utf-8"
            )
        except OSError:
            logger.exception("Failed to write heartbeat file")
        await asyncio.sleep(30)


async def init_app_state() -> AppState:
    """Initialize shared application state (config, DB, tools, router).

    Used by the gateway and by the CLI standalone mode.
    """
    global app_state

    config.load()
    _setup_logging()

    await init_db()

    executor = ProcessPoolExecutor(max_workers=4)
    semaphore = asyncio.Semaphore(1)

    from spare_paw.router.openrouter import OpenRouterClient

    api_key = config.get("openrouter.api_key")
    if not api_key:
        raise RuntimeError("No openrouter.api_key in config. Run 'python -m spare_paw setup' first.")

    router_client = OpenRouterClient(api_key=api_key, semaphore=semaphore)

    from spare_paw.tools.registry import ToolRegistry

    tool_registry = ToolRegistry()

    state = AppState(
        config=config,
        executor=executor,
        semaphore=semaphore,
        tool_registry=tool_registry,
        router_client=router_client,
    )
    app_state = state

    # Register tools
    config_data = config.data

    from spare_paw.tools import code, cron_tools, files, lcm_tools, memory, shell, subagent, tavily_search, web_scrape
    from spare_paw.tools.custom_tools import load_custom_tools, register_meta_tools

    shell.register(tool_registry, config_data)
    files.register(tool_registry, config_data)
    tavily_search.register(tool_registry, config_data)
    web_scrape.register(tool_registry, config_data)
    cron_tools.register(tool_registry, config_data, state)
    memory.register(tool_registry, config_data)
    code.register(tool_registry, config_data, state)
    load_custom_tools(tool_registry, executor)
    register_meta_tools(tool_registry, config_data, state)
    subagent.register(tool_registry, config_data, state)
    lcm_tools.register(tool_registry, config_data)

    # Inline read_logs tool
    async def _read_logs(count: int = 50) -> str:
        import json as _json
        log_path = LOG_DIR / "spare-paw.log"
        if not log_path.exists():
            return _json.dumps({"error": "Log file not found"})
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        tail = [line for line in lines[-min(count, 200):] if "getUpdates" not in line]
        return _json.dumps({"lines": tail, "count": len(tail)})

    tool_registry.register(
        name="read_logs",
        description="Read recent application log lines. Useful for debugging errors.",
        parameters_schema={
            "type": "object",
            "properties": {
                "count": {"type": "integer", "description": "Number of lines to read", "default": 50},
            },
        },
        handler=_read_logs,
        run_in_executor=False,
    )

    # Inline send_file tool
    _send_file_blocked = {".spare-paw/config.yaml", ".ssh/", ".gnupg/"}

    async def _send_file(path: str, caption: str = "") -> str:
        import json as _json
        from pathlib import Path as _Path
        fpath = _Path(path).resolve()
        if not fpath.exists():
            return _json.dumps({"error": f"File not found: {path}"})
        fpath_str = str(fpath)
        if any(blocked in fpath_str for blocked in _send_file_blocked):
            return _json.dumps({"error": "Access denied: sensitive path"})
        if state.backend is None:
            return _json.dumps({"error": "Backend not available"})
        await state.backend.send_file(str(fpath), caption=caption)
        suffix = fpath.suffix.lower()
        return _json.dumps({"success": True, "path": path, "type": suffix})

    tool_registry.register(
        name="send_file",
        description="Send a file (photo, video, audio, document) to the user.",
        parameters_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Absolute path to the file to send"},
                "caption": {"type": "string", "description": "Optional caption for the file"},
            },
            "required": ["path"],
        },
        handler=_send_file,
        run_in_executor=False,
    )

    # Inline send_message tool
    async def _send_message(text: str) -> str:
        import json as _json
        if state.backend is None:
            return _json.dumps({"error": "Backend not available"})
        await state.backend.send_text(text)
        return _json.dumps({"success": True})

    tool_registry.register(
        name="send_message",
        description=(
            "Send a message to the user. Use this inside cron jobs and "
            "background agents to deliver results. In normal conversation, prefer "
            "replying directly instead of calling this tool."
        ),
        parameters_schema={
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Message text to send"},
            },
            "required": ["text"],
        },
        handler=_send_message,
        run_in_executor=False,
    )

    # Connect to MCP servers
    mcp_servers = config.get("mcp.servers", [])
    if mcp_servers:
        try:
            from spare_paw.mcp.client import MCPClientManager

            mcp_client = MCPClientManager()
            await mcp_client.connect_all(mcp_servers, tool_registry)
            state.mcp_client = mcp_client
            logger.info("MCP client connected to %d servers", len(mcp_client.get_status()["servers"]))
        except Exception:
            logger.exception("Failed to initialize MCP client — continuing without MCP")

    logger.info("Tool registry initialized with %d tools", len(tool_registry))
    return state


async def _async_main() -> None:
    """Core async entry point that wires everything together and runs until stopped."""
    global app_state

    # 1-9. Initialize app state (config, DB, tools, router)
    try:
        app_state = await init_app_state()
    except RuntimeError as e:
        logger.error(str(e))
        return

    logger.info("Starting spare-paw gateway")

    # 10. Build backend (Telegram or webhook)
    backend_type = config.get("backend", "telegram")
    start_queue_processor = None

    if backend_type == "webhook":
        from spare_paw.webhook.backend import WebhookBackend

        backend = WebhookBackend(
            port=config.get("webhook.port", 8080),
            secret=config.get("webhook.secret", ""),
            app_state=app_state,
        )
        app_state.backend = backend
        logger.info("Using webhook backend on port %s", config.get("webhook.port", 8080))
    else:
        bot_token = config.get("telegram.bot_token")
        if not bot_token:
            logger.error("No telegram.bot_token in config. Run 'python -m spare_paw setup' first.")
            return

        owner_id = config.get("telegram.owner_id")

        from spare_paw.bot.backend import TelegramBackend

        backend = TelegramBackend.create(bot_token, owner_id)
        app_state.backend = backend
        backend.set_app_state(app_state)

        # Register Telegram handlers
        try:
            from spare_paw.bot.handler import setup_handlers, start_queue_processor

            setup_handlers(backend._application)
            logger.info("Bot handlers registered")
        except ImportError:
            start_queue_processor = None
            logger.warning("bot.handler not yet implemented; skipping handler registration")

    # 10b. Start secondary webhook backend if enabled alongside primary
    webhook_backend = None
    if backend_type != "webhook" and config.get("webhook.enabled", False):
        from spare_paw.webhook.backend import WebhookBackend

        webhook_backend = WebhookBackend(
            port=config.get("webhook.port", 8080),
            secret=config.get("webhook.secret", ""),
            app_state=app_state,
        )
        app_state.webhook = webhook_backend

    # 12. Init cron scheduler
    try:
        from spare_paw.cron.scheduler import init_scheduler

        scheduler = await init_scheduler(app_state)
        app_state.scheduler = scheduler
        logger.info("Cron scheduler initialized")
    except ImportError:
        logger.warning("cron.scheduler not yet implemented; skipping cron init")

    # 13. Start heartbeat
    heartbeat_task = asyncio.create_task(_heartbeat(), name="heartbeat")
    logger.info("Heartbeat task started")

    # 14. Setup graceful shutdown
    shutdown_event = asyncio.Event()

    def _signal_handler(sig: int) -> None:
        logger.info("Received signal %s, shutting down...", signal.Signals(sig).name)
        shutdown_event.set()

    loop = asyncio.get_running_loop()
    if sys.platform != "win32":
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, _signal_handler, sig)
    else:
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, lambda s, _f: _signal_handler(s))

    # 15. Start the bot via backend
    await backend.start()

    # 15b. Start secondary webhook backend
    if webhook_backend is not None:
        await webhook_backend.start()
        logger.info("Webhook API listening on port %d", config.get("webhook.port", 8080))

    # 16. Start message queue processor (must be after initialize/start)
    if start_queue_processor is not None:
        start_queue_processor(backend._application)

    logger.info("Bot started (%s backend)", backend_type)

    # Wait until shutdown signal
    await shutdown_event.wait()

    # ---- Graceful shutdown ----
    logger.info("Shutting down...")

    heartbeat_task.cancel()
    try:
        await heartbeat_task
    except asyncio.CancelledError:
        pass

    # Stop cron scheduler
    if app_state.scheduler is not None:
        try:
            await app_state.scheduler.stop()
        except Exception:
            logger.exception("Error shutting down scheduler")

    # Close MCP client
    if app_state.mcp_client is not None:
        try:
            await app_state.mcp_client.close()
        except Exception:
            logger.exception("Error closing MCP client")

    # Close OpenRouter client
    try:
        await app_state.router_client.close()
    except Exception:
        logger.exception("Error closing OpenRouter client")

    # Stop secondary webhook backend
    if webhook_backend is not None:
        await webhook_backend.stop()

    # Stop primary backend
    await backend.stop()

    # Shutdown process pool
    app_state.executor.shutdown(wait=False)

    # Close database
    await close_db()

    logger.info("spare-paw gateway stopped")


def run() -> None:
    """Synchronous entry point called from __main__."""
    asyncio.run(_async_main())
