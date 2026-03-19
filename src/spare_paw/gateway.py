"""Main async entry point: Telegram bot + cron scheduler + heartbeat.

Orchestrates startup, wires all components together, and runs until stopped.
"""

from __future__ import annotations

import asyncio
import logging
import signal
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

from telegram.ext import Application

from spare_paw.config import Config, config
from spare_paw.db import close_db, init_db

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
    application: Application | None = None
    scheduler: Any = None
    mcp_client: Any = None
    start_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# Module-level singleton so other modules can access shared state
app_state: AppState | None = None


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

    # Rotating file handler
    file_handler = RotatingFileHandler(
        LOG_DIR / "spare-paw.log",
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    file_handler.setLevel(level)
    file_formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)

    # Stderr handler
    stderr_handler = logging.StreamHandler()
    stderr_handler.setLevel(level)
    stderr_handler.setFormatter(file_formatter)
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


async def _async_main() -> None:
    """Core async entry point that wires everything together and runs until stopped."""
    global app_state

    # 1. Load config
    config.load()

    # 2. Setup logging (needs config loaded first)
    _setup_logging()
    logger.info("Starting spare-paw gateway")

    # 3. Init database
    await init_db()
    logger.info("Database initialized")

    # 4. Create process pool executor
    executor = ProcessPoolExecutor(max_workers=4)

    # 5. Create semaphore for model call serialization
    semaphore = asyncio.Semaphore(1)

    # 6. Create OpenRouter client
    from spare_paw.router.openrouter import OpenRouterClient

    api_key = config.get("openrouter.api_key")
    if not api_key:
        logger.error("No openrouter.api_key in config. Run 'python -m spare_paw setup' first.")
        return

    router_client = OpenRouterClient(api_key=api_key, semaphore=semaphore)

    # 7. Create tool registry and register all tools
    from spare_paw.tools.registry import ToolRegistry

    tool_registry = ToolRegistry()

    # 8. Build app state
    app_state = AppState(
        config=config,
        executor=executor,
        semaphore=semaphore,
        tool_registry=tool_registry,
        router_client=router_client,
    )

    # 9. Register tools (needs app_state for cron_tools)
    config_data = config.data

    from spare_paw.tools import code, cron_tools, files, lcm_tools, memory, shell, subagent, tavily_search, web_scrape
    from spare_paw.tools.custom_tools import load_custom_tools, register_meta_tools

    shell.register(tool_registry, config_data)
    files.register(tool_registry, config_data)
    tavily_search.register(tool_registry, config_data)
    web_scrape.register(tool_registry, config_data)
    cron_tools.register(tool_registry, config_data, app_state)
    memory.register(tool_registry, config_data)
    code.register(tool_registry, config_data, app_state)
    load_custom_tools(tool_registry, executor)
    register_meta_tools(tool_registry, config_data, app_state)
    subagent.register(tool_registry, config_data, app_state)
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

    # Inline send_file tool — sends a file via Telegram to the owner
    _send_file_blocked = {".spare-paw/config.yaml", ".ssh/", ".gnupg/"}

    async def _send_file(path: str, caption: str = "") -> str:
        import json as _json
        from pathlib import Path as _Path
        fpath = _Path(path).resolve()
        if not fpath.exists():
            return _json.dumps({"error": f"File not found: {path}"})
        # Block sensitive paths
        fpath_str = str(fpath)
        if any(blocked in fpath_str for blocked in _send_file_blocked):
            return _json.dumps({"error": "Access denied: sensitive path"})
        owner_id = config.get("telegram.owner_id")
        if not owner_id or not app_state.application:
            return _json.dumps({"error": "Telegram not available"})
        bot = app_state.application.bot
        suffix = fpath.suffix.lower()
        with open(fpath, "rb") as f:
            if suffix in (".jpg", ".jpeg", ".png", ".gif", ".webp"):
                await bot.send_photo(chat_id=owner_id, photo=f, caption=caption or None)
            elif suffix in (".mp4", ".mov", ".avi"):
                await bot.send_video(chat_id=owner_id, video=f, caption=caption or None)
            elif suffix in (".mp3", ".ogg", ".m4a", ".wav"):
                await bot.send_audio(chat_id=owner_id, audio=f, caption=caption or None)
            else:
                await bot.send_document(chat_id=owner_id, document=f, caption=caption or None)
        return _json.dumps({"success": True, "path": path, "type": suffix})

    tool_registry.register(
        name="send_file",
        description="Send a file (photo, video, audio, document) to the user via Telegram.",
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

    # Inline send_message tool — proactively send a text message to the owner
    async def _send_message(text: str) -> str:
        import json as _json
        owner_id = config.get("telegram.owner_id")
        if not owner_id or not app_state.application:
            return _json.dumps({"error": "Telegram not available"})
        bot = app_state.application.bot
        # Chunk if needed
        while text:
            chunk = text[:4096]
            text = text[4096:]
            await bot.send_message(chat_id=owner_id, text=chunk)
        return _json.dumps({"success": True})

    tool_registry.register(
        name="send_message",
        description=(
            "Send a message to the user via Telegram. Use this inside cron jobs and "
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

    # 9b. Connect to MCP servers (if configured)
    mcp_servers = config.get("mcp.servers", [])
    if mcp_servers:
        try:
            from spare_paw.mcp.client import MCPClientManager

            mcp_client = MCPClientManager()
            await mcp_client.connect_all(mcp_servers, tool_registry)
            app_state.mcp_client = mcp_client
            logger.info("MCP client connected to %d servers", len(mcp_client.get_status()["servers"]))
        except Exception:
            logger.exception("Failed to initialize MCP client — continuing without MCP")

    logger.info("Tool registry initialized with %d tools", len(tool_registry))

    # 10. Build Telegram application
    bot_token = config.get("telegram.bot_token")
    if not bot_token:
        logger.error("No telegram.bot_token in config. Run 'python -m spare_paw setup' first.")
        return

    application = (
        Application.builder()
        .token(bot_token)
        .build()
    )
    app_state.application = application

    # 11. Store app_state in bot_data so handlers can access it
    application.bot_data["app_state"] = app_state

    # 12. Register handlers
    try:
        from spare_paw.bot.handler import setup_handlers, start_queue_processor

        setup_handlers(application)
        logger.info("Bot handlers registered")
    except ImportError:
        start_queue_processor = None
        logger.warning("bot.handler not yet implemented; skipping handler registration")

    # 13. Init cron scheduler
    try:
        from spare_paw.cron.scheduler import init_scheduler

        scheduler = await init_scheduler(app_state)
        app_state.scheduler = scheduler
        logger.info("Cron scheduler initialized")
    except ImportError:
        logger.warning("cron.scheduler not yet implemented; skipping cron init")

    # 14. Start heartbeat
    heartbeat_task = asyncio.create_task(_heartbeat(), name="heartbeat")
    logger.info("Heartbeat task started")

    # 15. Setup graceful shutdown
    shutdown_event = asyncio.Event()

    def _signal_handler(sig: int) -> None:
        logger.info("Received signal %s, shutting down...", signal.Signals(sig).name)
        shutdown_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _signal_handler, sig)

    # 16. Start the bot
    await application.initialize()
    await application.start()

    # 17. Start message queue processor (must be after initialize/start)
    if start_queue_processor is not None:
        start_queue_processor(application)

    if application.updater is not None:
        await application.updater.start_polling(drop_pending_updates=True)
    logger.info("Telegram bot started polling")

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
        await router_client.close()
    except Exception:
        logger.exception("Error closing OpenRouter client")

    # Stop telegram bot
    if application.updater is not None:
        await application.updater.stop()
    await application.stop()
    await application.shutdown()

    # Shutdown process pool
    executor.shutdown(wait=False)

    # Close database
    await close_db()

    logger.info("spare-paw gateway stopped")


def run() -> None:
    """Synchronous entry point called from __main__."""
    asyncio.run(_async_main())
