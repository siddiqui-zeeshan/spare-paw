"""Slash command handlers for the Telegram bot.

Commands:
    /cron list|remove|pause|resume|info — cron job management
    /config show|model|reset            — runtime configuration
    /status                             — uptime, memory, DB info
    /search <query>                     — FTS5 search over history
    /forget                             — start a new conversation
    /model <name>                       — shortcut for /config model
    /approve <name>                     — approve a pending custom tool
    /mcp                                — list connected MCP servers
"""

from __future__ import annotations

import json
import logging
import os
import resource
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from telegram import Update
from telegram.ext import CommandHandler, ContextTypes

from spare_paw.db import DB_PATH, get_db

if TYPE_CHECKING:
    from telegram.ext import Application

logger = logging.getLogger(__name__)


def _get_app_state(context: ContextTypes.DEFAULT_TYPE) -> Any:
    """Retrieve AppState from bot_data."""
    return context.bot_data["app_state"]


def _is_owner(update: Update, app_state: Any) -> bool:
    """Check whether the update comes from the configured owner."""
    owner_id = app_state.config.get("telegram.owner_id")
    return update.effective_user is not None and update.effective_user.id == owner_id


# ---------------------------------------------------------------------------
# /cron — subcommand dispatcher
# ---------------------------------------------------------------------------

async def _cron_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /cron <subcommand> [args]."""
    app_state = _get_app_state(context)
    if not _is_owner(update, app_state):
        return

    args = context.args or []
    if not args:
        await update.message.reply_text(
            "Usage: /cron <list|remove|pause|resume|info> [id]"
        )
        return

    subcommand = args[0].lower()
    sub_args = args[1:]

    dispatch = {
        "list": _cron_list,
        "remove": _cron_remove,
        "pause": _cron_pause,
        "resume": _cron_resume,
        "info": _cron_info,
    }

    handler = dispatch.get(subcommand)
    if handler is None:
        await update.message.reply_text(
            f"Unknown subcommand: {subcommand}\n"
            "Usage: /cron <list|remove|pause|resume|info> [id]"
        )
        return

    await handler(update, app_state, sub_args)


async def _cron_list(update: Update, app_state: Any, _args: list[str]) -> None:
    """List all cron jobs."""
    db = await get_db()
    async with db.execute(
        "SELECT id, name, schedule, enabled, last_run_at FROM cron_jobs ORDER BY created_at"
    ) as cursor:
        rows = await cursor.fetchall()

    if not rows:
        await update.message.reply_text("No cron jobs configured.")
        return

    lines: list[str] = []
    for row in rows:
        status = "enabled" if row["enabled"] else "PAUSED"
        last_run = row["last_run_at"] or "never"
        lines.append(
            f"  {row['id'][:8]}  {row['name']}\n"
            f"    schedule: {row['schedule']}  |  {status}  |  last: {last_run}"
        )

    text = "Cron jobs:\n\n" + "\n\n".join(lines)
    await update.message.reply_text(text)


async def _cron_remove(update: Update, app_state: Any, args: list[str]) -> None:
    """Delete a cron job by ID (prefix match)."""
    if not args:
        await update.message.reply_text("Usage: /cron remove <id>")
        return

    cron_id = await _resolve_cron_id(args[0])
    if cron_id is None:
        await update.message.reply_text(f"No cron job found matching: {args[0]}")
        return

    db = await get_db()
    await db.execute("DELETE FROM cron_jobs WHERE id = ?", (cron_id,))
    await db.commit()

    # Remove from APScheduler if present
    scheduler = app_state.scheduler
    if scheduler is not None:
        try:
            await scheduler.remove_job(cron_id)
        except Exception:
            pass  # Job may not be scheduled

    await update.message.reply_text(f"Cron job {cron_id[:8]} removed.")


async def _cron_pause(update: Update, app_state: Any, args: list[str]) -> None:
    """Pause a cron job."""
    if not args:
        await update.message.reply_text("Usage: /cron pause <id>")
        return

    cron_id = await _resolve_cron_id(args[0])
    if cron_id is None:
        await update.message.reply_text(f"No cron job found matching: {args[0]}")
        return

    db = await get_db()
    await db.execute("UPDATE cron_jobs SET enabled = 0 WHERE id = ?", (cron_id,))
    await db.commit()

    scheduler = app_state.scheduler
    if scheduler is not None:
        try:
            await scheduler.pause_job(cron_id)
        except Exception:
            pass

    await update.message.reply_text(f"Cron job {cron_id[:8]} paused.")


async def _cron_resume(update: Update, app_state: Any, args: list[str]) -> None:
    """Resume a paused cron job."""
    if not args:
        await update.message.reply_text("Usage: /cron resume <id>")
        return

    cron_id = await _resolve_cron_id(args[0])
    if cron_id is None:
        await update.message.reply_text(f"No cron job found matching: {args[0]}")
        return

    db = await get_db()
    await db.execute("UPDATE cron_jobs SET enabled = 1 WHERE id = ?", (cron_id,))
    await db.commit()

    scheduler = app_state.scheduler
    if scheduler is not None:
        try:
            await scheduler.resume_job(cron_id)
        except Exception:
            pass

    await update.message.reply_text(f"Cron job {cron_id[:8]} resumed.")


async def _cron_info(update: Update, app_state: Any, args: list[str]) -> None:
    """Show full details for a cron job."""
    if not args:
        await update.message.reply_text("Usage: /cron info <id>")
        return

    cron_id = await _resolve_cron_id(args[0])
    if cron_id is None:
        await update.message.reply_text(f"No cron job found matching: {args[0]}")
        return

    db = await get_db()
    async with db.execute(
        "SELECT * FROM cron_jobs WHERE id = ?", (cron_id,)
    ) as cursor:
        row = await cursor.fetchone()

    if row is None:
        await update.message.reply_text(f"Cron job {args[0]} not found.")
        return

    status = "enabled" if row["enabled"] else "PAUSED"
    model = row["model"] or "(default)"
    last_run = row["last_run_at"] or "never"
    last_result = row["last_result"] or "(none)"
    last_error = row["last_error"] or "(none)"

    # Truncate long results for readability
    if len(last_result) > 500:
        last_result = last_result[:500] + "..."
    if len(last_error) > 500:
        last_error = last_error[:500] + "..."

    text = (
        f"Cron: {row['name']}\n"
        f"ID: {row['id']}\n"
        f"Schedule: {row['schedule']}\n"
        f"Status: {status}\n"
        f"Model: {model}\n"
        f"Created: {row['created_at']}\n"
        f"Last run: {last_run}\n\n"
        f"Prompt:\n{row['prompt']}\n\n"
        f"Last result:\n{last_result}\n\n"
        f"Last error:\n{last_error}"
    )
    await update.message.reply_text(text)


async def _resolve_cron_id(partial_id: str) -> str | None:
    """Resolve a partial cron ID to a full ID via prefix match."""
    db = await get_db()
    async with db.execute(
        "SELECT id FROM cron_jobs WHERE id LIKE ?", (partial_id + "%",)
    ) as cursor:
        rows = await cursor.fetchall()

    if len(rows) == 1:
        return rows[0]["id"]
    # Try exact match if multiple prefix matches
    for row in rows:
        if row["id"] == partial_id:
            return partial_id
    return None


# ---------------------------------------------------------------------------
# /config — runtime configuration management
# ---------------------------------------------------------------------------

async def _config_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /config <show|model|reset> [args]."""
    app_state = _get_app_state(context)
    if not _is_owner(update, app_state):
        return

    args = context.args or []
    if not args:
        await update.message.reply_text("Usage: /config <show|model|reset> [value]")
        return

    subcommand = args[0].lower()

    if subcommand == "show":
        await _config_show(update, app_state)
    elif subcommand == "model":
        if len(args) < 2:
            await update.message.reply_text("Usage: /config model <model_name>")
            return
        await _config_model(update, app_state, args[1])
    elif subcommand == "reset":
        await _config_reset(update, app_state)
    else:
        await update.message.reply_text(
            f"Unknown subcommand: {subcommand}\n"
            "Usage: /config <show|model|reset>"
        )


async def _config_show(update: Update, app_state: Any) -> None:
    """Show current model configuration and runtime overrides."""
    cfg = app_state.config
    default_model = cfg.get("models.default", "(not set)")
    smart_model = cfg.get("models.smart", "(not set)")
    cron_model = cfg.get("models.cron_default", "(not set)")

    overrides = cfg._overrides  # noqa: SLF001
    override_text = "(none)"
    if overrides:
        override_lines = []
        _flatten_overrides(overrides, "", override_lines)
        override_text = "\n".join(f"  {k} = {v}" for k, v in override_lines)

    text = (
        f"Models:\n"
        f"  default: {default_model}\n"
        f"  smart: {smart_model}\n"
        f"  cron_default: {cron_model}\n\n"
        f"Runtime overrides:\n{override_text}"
    )
    await update.message.reply_text(text)


def _flatten_overrides(
    d: dict, prefix: str, out: list[tuple[str, Any]]
) -> None:
    """Flatten a nested dict into dot-separated key/value pairs."""
    for k, v in d.items():
        key = f"{prefix}{k}" if not prefix else f"{prefix}.{k}"
        if isinstance(v, dict):
            _flatten_overrides(v, key, out)
        else:
            out.append((key, v))


async def _config_model(update: Update, app_state: Any, model_name: str) -> None:
    """Set a runtime model override."""
    app_state.config.set_override("models.default", model_name)
    await update.message.reply_text(f"Default model set to: {model_name}")


async def _config_reset(update: Update, app_state: Any) -> None:
    """Clear all runtime overrides."""
    app_state.config.reset_overrides()
    await update.message.reply_text("Runtime overrides cleared. Using config.yaml defaults.")


# ---------------------------------------------------------------------------
# /status — system status
# ---------------------------------------------------------------------------

async def _status_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show system status: uptime, memory, DB size, cron count, last error."""
    app_state = _get_app_state(context)
    if not _is_owner(update, app_state):
        return

    now = datetime.now(timezone.utc)
    uptime = now - app_state.start_time
    uptime_str = _format_timedelta(uptime)

    # Memory usage via resource module (portable, no psutil needed)
    usage = resource.getrusage(resource.RUSAGE_SELF)
    # ru_maxrss is in kilobytes on Linux, bytes on macOS
    rss_kb = usage.ru_maxrss
    if os.uname().sysname == "Darwin":
        rss_kb = rss_kb // 1024  # macOS reports bytes
    rss_mb = rss_kb / 1024

    # DB file size
    db_size_str = "(not found)"
    if DB_PATH.exists():
        db_bytes = DB_PATH.stat().st_size
        if db_bytes < 1024 * 1024:
            db_size_str = f"{db_bytes / 1024:.1f} KB"
        else:
            db_size_str = f"{db_bytes / (1024 * 1024):.1f} MB"

    # Active cron count
    db = await get_db()
    async with db.execute(
        "SELECT COUNT(*) as cnt FROM cron_jobs WHERE enabled = 1"
    ) as cursor:
        row = await cursor.fetchone()
    active_crons = row["cnt"] if row else 0

    # Current model
    current_model = app_state.config.get("models.default", "(not set)")

    text = (
        f"Uptime: {uptime_str}\n"
        f"Memory (RSS): {rss_mb:.1f} MB\n"
        f"Database: {db_size_str}\n"
        f"Active crons: {active_crons}\n"
        f"Current model: {current_model}"
    )
    await update.message.reply_text(text)


def _format_timedelta(td) -> str:
    """Format a timedelta as a human-readable string."""
    total_seconds = int(td.total_seconds())
    days, remainder = divmod(total_seconds, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, seconds = divmod(remainder, 60)

    parts: list[str] = []
    if days:
        parts.append(f"{days}d")
    if hours:
        parts.append(f"{hours}h")
    if minutes:
        parts.append(f"{minutes}m")
    parts.append(f"{seconds}s")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# /search — FTS5 full-text search
# ---------------------------------------------------------------------------

async def _search_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Search conversation history via FTS5."""
    app_state = _get_app_state(context)
    if not _is_owner(update, app_state):
        return

    query = " ".join(context.args) if context.args else ""
    if not query:
        await update.message.reply_text("Usage: /search <query>")
        return

    from spare_paw import context as ctx_module

    try:
        results = await ctx_module.search(query)
    except Exception as exc:
        await update.message.reply_text(f"Search error: {exc}")
        return

    if not results:
        await update.message.reply_text(f"No results for: {query}")
        return

    lines: list[str] = []
    for r in results:
        content_preview = r["content"][:120].replace("\n", " ")
        lines.append(
            f"[{r['role']}] {r['created_at'][:16]}\n  {content_preview}"
        )

    text = f"Search results for '{query}':\n\n" + "\n\n".join(lines)

    # Respect Telegram message length limit
    if len(text) > 4096:
        text = text[:4090] + "\n..."

    await update.message.reply_text(text)


# ---------------------------------------------------------------------------
# /forget — start new conversation
# ---------------------------------------------------------------------------

async def _forget_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Start a new conversation, clearing the context window."""
    app_state = _get_app_state(context)
    if not _is_owner(update, app_state):
        return

    from spare_paw import context as ctx_module

    await ctx_module.new_conversation()
    await update.message.reply_text(
        "New conversation started. (Previous history is preserved in DB.)"
    )


# ---------------------------------------------------------------------------
# /model — shortcut for /config model
# ---------------------------------------------------------------------------

async def _model_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Shortcut: /model <name> sets the default model."""
    app_state = _get_app_state(context)
    if not _is_owner(update, app_state):
        return

    if not context.args:
        current = app_state.config.get("models.default", "(not set)")
        await update.message.reply_text(f"Current model: {current}\nUsage: /model <name>")
        return

    model_name = context.args[0]
    app_state.config.set_override("models.default", model_name)
    await update.message.reply_text(f"Default model set to: {model_name}")


# ---------------------------------------------------------------------------
# /tools — list available tools
# ---------------------------------------------------------------------------

async def _tools_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """List all registered tools."""
    app_state = _get_app_state(context)
    if not _is_owner(update, app_state):
        return

    registry = app_state.tool_registry
    if registry is None:
        await update.message.reply_text("Tool registry not available.")
        return

    schemas = registry.get_schemas()
    if not schemas:
        await update.message.reply_text("No tools registered.")
        return

    lines: list[str] = []
    for schema in schemas:
        func = schema.get("function", {})
        name = func.get("name", "?")
        desc = func.get("description", "")
        # Truncate long descriptions
        if len(desc) > 80:
            desc = desc[:77] + "..."
        lines.append(f"  {name}\n    {desc}")

    text = f"Available tools ({len(schemas)}):\n\n" + "\n\n".join(lines)
    await update.message.reply_text(text)


# ---------------------------------------------------------------------------
# /approve — approve a pending custom tool
# ---------------------------------------------------------------------------

async def _approve_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Approve a pending custom tool: /approve <name>."""
    app_state = _get_app_state(context)
    if not _is_owner(update, app_state):
        return

    if not context.args:
        # List pending tools as a hint
        from spare_paw.tools.custom_tools import PENDING_DIR

        pending: list[str] = []
        if PENDING_DIR.exists():
            pending = [p.stem for p in PENDING_DIR.glob("*.json")]

        if pending:
            await update.message.reply_text(
                f"Usage: /approve <name>\n\nPending tools: {', '.join(pending)}"
            )
        else:
            await update.message.reply_text(
                "Usage: /approve <name>\n\nNo tools currently pending approval."
            )
        return

    name = context.args[0]

    from spare_paw.tools.custom_tools import approve_tool

    result_str = await approve_tool(name, app_state.tool_registry, app_state)
    result = json.loads(result_str)

    if result.get("error"):
        await update.message.reply_text(f"Error: {result['error']}")
    else:
        await update.message.reply_text(f"Tool '{name}' approved and activated.")


# ---------------------------------------------------------------------------
# /logs — show recent log lines
# ---------------------------------------------------------------------------

async def _logs_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show the last N log lines (default 50)."""
    app_state = _get_app_state(context)
    if not _is_owner(update, app_state):
        return

    count = 50
    if context.args:
        try:
            count = int(context.args[0])
            count = min(count, 200)  # cap at 200
        except ValueError:
            pass

    log_path = Path.home() / ".spare-paw" / "logs" / "spare-paw.log"
    if not log_path.exists():
        await update.message.reply_text("Log file not found.")
        return

    try:
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        tail = lines[-count:]
        # Filter out noisy getUpdates lines
        tail = [line for line in tail if "getUpdates" not in line]
        text = "\n".join(tail) or "(no logs)"
    except OSError as exc:
        await update.message.reply_text(f"Error reading logs: {exc}")
        return

    if len(text) > 4096:
        text = text[-4090:] + "\n..."

    await update.message.reply_text(text)


# ---------------------------------------------------------------------------
# /agents — list background agents
# ---------------------------------------------------------------------------

async def _agents_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """List all background agents and their status."""
    app_state = _get_app_state(context)
    if not _is_owner(update, app_state):
        return

    from spare_paw.tools.subagent import _agents

    if not _agents:
        await update.message.reply_text("No agents have been spawned.")
        return

    lines: list[str] = []
    for aid, info in sorted(_agents.items(), key=lambda x: x[1].get("created_at", ""), reverse=True):
        status = info.get("status", "?")
        name = info.get("name", "?")
        lines.append(f"  {aid[:8]}  {name}  [{status}]")

    text = f"Agents ({len(_agents)}):\n\n" + "\n".join(lines[:20])
    await update.message.reply_text(text)


# ---------------------------------------------------------------------------
# /mcp — list connected MCP servers and tools
# ---------------------------------------------------------------------------

async def _mcp_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show connected MCP servers and their tools."""
    app_state = _get_app_state(context)
    if not _is_owner(update, app_state):
        return

    mcp_client = getattr(app_state, "mcp_client", None)
    if mcp_client is None:
        await update.message.reply_text("No MCP servers connected.")
        return

    status = mcp_client.get_status()
    if not status["servers"]:
        await update.message.reply_text("No MCP servers connected.")
        return

    lines: list[str] = []
    for srv in status["servers"]:
        state = "connected" if srv["connected"] else "DISCONNECTED"
        lines.append(f"  {srv['name']} [{state}] — {srv['tools']} tools")
        for tool_name in srv["tool_names"]:
            lines.append(f"    - {tool_name}")

    text = f"MCP servers ({len(status['servers'])}):\n\n" + "\n".join(lines)
    await update.message.reply_text(text)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def register_commands(application: "Application") -> None:
    """Register all command handlers on the application."""
    application.add_handler(CommandHandler("cron", _cron_handler))
    application.add_handler(CommandHandler("config", _config_handler))
    application.add_handler(CommandHandler("status", _status_handler))
    application.add_handler(CommandHandler("search", _search_handler))
    application.add_handler(CommandHandler("forget", _forget_handler))
    application.add_handler(CommandHandler("model", _model_handler))
    application.add_handler(CommandHandler("tools", _tools_handler))
    application.add_handler(CommandHandler("approve", _approve_handler))
    application.add_handler(CommandHandler("agents", _agents_handler))
    application.add_handler(CommandHandler("logs", _logs_handler))
    application.add_handler(CommandHandler("mcp", _mcp_handler))
