"""Platform-agnostic command functions.

Each function accepts app_state and returns a string response.
No Telegram imports — the bot/commands.py layer handles Update parsing
and sends the returned string.
"""

from __future__ import annotations

import logging
import platform as platform_mod
import resource
from datetime import datetime, timezone
from typing import Any

from spare_paw import context as ctx_module
from spare_paw.config import MODEL_ROLES
from spare_paw.db import DB_PATH, get_db

logger = logging.getLogger(__name__)


async def cmd_status(app_state: Any) -> str:
    """Return system status: uptime, memory, DB size, cron count, model."""
    now = datetime.now(timezone.utc)
    uptime = now - app_state.start_time
    uptime_str = _format_timedelta(uptime)

    usage = resource.getrusage(resource.RUSAGE_SELF)
    rss_kb = usage.ru_maxrss
    if platform_mod.system() == "Darwin":
        rss_kb = rss_kb // 1024
    rss_mb = rss_kb / 1024

    db_size_str = "(not found)"
    if DB_PATH.exists():
        db_bytes = DB_PATH.stat().st_size
        if db_bytes < 1024 * 1024:
            db_size_str = f"{db_bytes / 1024:.1f} KB"
        else:
            db_size_str = f"{db_bytes / (1024 * 1024):.1f} MB"

    db = await get_db()
    async with db.execute(
        "SELECT COUNT(*) as cnt FROM cron_jobs WHERE enabled = 1"
    ) as cursor:
        row = await cursor.fetchone()
    active_crons = row["cnt"] if row else 0

    current_model = app_state.config.get("models.main_agent", "(not set)")

    return (
        f"Uptime: {uptime_str}\n"
        f"Memory (RSS): {rss_mb:.1f} MB\n"
        f"Database: {db_size_str}\n"
        f"Active crons: {active_crons}\n"
        f"Current model: {current_model}"
    )


async def cmd_forget(app_state: Any) -> str:
    """Start a new conversation."""
    await ctx_module.new_conversation()
    return "New conversation started. (Previous history is preserved in DB.)"


async def cmd_search(app_state: Any, query: str) -> str:
    """Full-text search over conversation history."""
    if not query:
        return "Usage: /search <query>"

    try:
        results = await ctx_module.search(query)
    except Exception as exc:
        return f"Search error: {exc}"

    if not results:
        return f"No results for: {query}"

    lines: list[str] = []
    for r in results:
        content_preview = r["content"][:120].replace("\n", " ")
        lines.append(f"[{r['role']}] {r['created_at'][:16]}\n  {content_preview}")

    text = f"Search results for '{query}':\n\n" + "\n\n".join(lines)
    if len(text) > 4096:
        text = text[:4090] + "\n..."
    return text


async def cmd_roles() -> str:
    """List available model roles."""
    lines = [f"  {role}" for role in MODEL_ROLES]
    return (
        "Available roles:\n" + "\n".join(lines)
        + "\n\nUsage: /model <role> <model_id>"
    )


async def cmd_model(app_state: Any, args: list[str] | None) -> str:
    """Show or set model role assignments.

    /model              — show all roles
    /model <model_id>   — set main_agent (backward compat)
    /model <role> <id>  — set a specific role
    """
    if not args:
        lines: list[str] = []
        for role in MODEL_ROLES:
            model = app_state.config.get(f"models.{role}", "(not set)")
            lines.append(f"  {role}: {model}")
        return "Model roles:\n" + "\n".join(lines)

    if len(args) == 1:
        model_id = args[0]
        app_state.config.set_override("models.main_agent", model_id)
        return f"main_agent model set to: {model_id}"

    role, model_id = args[0], args[1]
    if role not in MODEL_ROLES:
        return f"Unknown role: {role}\nValid roles: {', '.join(MODEL_ROLES)}"

    # Best-effort validation against OpenRouter model list
    router_client = getattr(app_state, "router_client", None)
    if router_client is not None:
        try:
            models = await router_client.list_models()
            valid_ids = {m["id"] for m in models}
            if model_id not in valid_ids:
                return f"Unknown model: {model_id}\nUse /models to list available models."
        except Exception:
            pass  # Validation is best-effort

    app_state.config.set_override(f"models.{role}", model_id)
    return f"{role} model set to: {model_id}"


async def cmd_models(app_state: Any, query: str | None = None) -> str:
    """List available models from OpenRouter, optionally filtered by keyword."""
    router_client = getattr(app_state, "router_client", None)
    if router_client is None:
        return "OpenRouter client not available."
    try:
        models = await router_client.list_models()
    except Exception as exc:
        return f"Failed to fetch models: {exc}"

    if query:
        query_lower = query.lower()
        models = [
            m for m in models
            if query_lower in m.get("id", "").lower()
            or query_lower in m.get("name", "").lower()
        ]

    if not models:
        return "No models found."

    lines = [f"  {m['id']}" for m in models[:30]]
    result = f"Available models ({len(models)} total):\n" + "\n".join(lines)
    if len(models) > 30:
        result += f"\n  ... and {len(models) - 30} more. Use /models <filter> to narrow."
    return result


async def cmd_config_show(app_state: Any) -> str:
    """Show current model configuration and runtime overrides."""
    cfg = app_state.config

    model_lines: list[str] = []
    for role in MODEL_ROLES:
        model = cfg.get(f"models.{role}", "(not set)")
        model_lines.append(f"  {role}: {model}")

    overrides = cfg._overrides  # noqa: SLF001
    override_text = "(none)"
    if overrides:
        override_lines: list[tuple[str, Any]] = []
        _flatten_overrides(overrides, "", override_lines)
        override_text = "\n".join(f"  {k} = {v}" for k, v in override_lines)

    return (
        "Models:\n" + "\n".join(model_lines) + "\n\n"
        f"Runtime overrides:\n{override_text}"
    )


async def cmd_config_reset(app_state: Any) -> str:
    """Clear all runtime overrides."""
    app_state.config.reset_overrides()
    return "Runtime overrides cleared. Using config.yaml defaults."


def _flatten_overrides(d: dict, prefix: str, out: list[tuple[str, Any]]) -> None:
    for k, v in d.items():
        key = f"{prefix}{k}" if not prefix else f"{prefix}.{k}"
        if isinstance(v, dict):
            _flatten_overrides(v, key, out)
        else:
            out.append((key, v))


def _format_timedelta(td: Any) -> str:
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
