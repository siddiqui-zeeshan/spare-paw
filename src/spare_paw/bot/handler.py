"""Main message handler with async queue and backpressure.

Incoming messages are placed on an asyncio.Queue and processed sequentially.
While a message is being processed, a typing indicator is sent to signal
the bot is busy. Voice messages are transcribed via Groq Whisper before
processing.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import CallbackQueryHandler, ContextTypes, MessageHandler, filters

from spare_paw.bot.commands import register_commands
from spare_paw.core.prompt import build_system_prompt as _build_system_prompt  # noqa: F401
from spare_paw.core.voice import VoiceTranscriptionError  # noqa: F401
from spare_paw.bot.backend import md_to_html as _md_to_html  # noqa: F401

if TYPE_CHECKING:
    from telegram.ext import Application

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public setup
# ---------------------------------------------------------------------------

def setup_handlers(application: "Application") -> None:
    """Register message handlers and all commands on *application*.

    The background queue processor is started via ``post_init``.
    """
    # Command handlers first (higher priority by default in ptb)
    register_commands(application)

    # Text messages (non-command)
    application.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, _queue_message)
    )
    # Voice messages
    application.add_handler(
        MessageHandler(filters.VOICE, _queue_message)
    )
    # Photo messages (with or without caption)
    application.add_handler(
        MessageHandler(filters.PHOTO, _queue_message)
    )
    # Inline keyboard button callbacks
    application.add_handler(CallbackQueryHandler(_handle_callback))

    # Queue processor is started explicitly by the gateway after initialize()


# ---------------------------------------------------------------------------
# Queue management
# ---------------------------------------------------------------------------

def start_queue_processor(application: "Application") -> None:
    """Start the background queue processor via core/engine.

    Delegates to engine.start_queue_processor which owns the queue.
    """
    from spare_paw.core.engine import start_queue_processor as _engine_start

    app_state = application.bot_data.get("app_state")
    backend = getattr(app_state, "backend", None) if app_state else None

    if app_state and backend:
        _engine_start(app_state, backend)
    else:
        logger.warning("Cannot start queue processor: app_state or backend not available")


async def _queue_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Build IncomingMessage from Telegram Update and enqueue for processing."""
    app_state = context.bot_data.get("app_state")
    if app_state is None:
        return

    # Owner-only auth: silently ignore messages from non-owner
    owner_id = app_state.config.get("telegram.owner_id")
    if update.effective_user is None or update.effective_user.id != owner_id:
        return

    from spare_paw.backend import IncomingMessage
    from spare_paw.core.engine import enqueue

    message = update.message
    if message is None:
        return

    msg = IncomingMessage(
        text=message.text,
        voice_bytes=(await _download_voice(message)) if message.voice else None,
        image_bytes=(await _download_photo(message)) if message.photo else None,
        caption=message.caption,
        cron_context=_extract_cron_context(update),
        user_id=update.effective_user.id if update.effective_user else None,
    )

    await enqueue(msg)


# ---------------------------------------------------------------------------
# Telegram-specific download helpers
# ---------------------------------------------------------------------------


async def _download_voice(message: Any) -> bytes | None:
    """Download voice message bytes from Telegram."""
    try:
        voice_file = await message.voice.get_file()
        return bytes(await voice_file.download_as_bytearray())
    except Exception:
        logger.exception("Failed to download voice message")
        return None


async def _download_photo(message: Any) -> bytes | None:
    """Download the largest photo resolution from Telegram."""
    try:
        photo = message.photo[-1]
        file = await photo.get_file()
        return bytes(await file.download_as_bytearray())
    except Exception:
        logger.exception("Failed to download photo")
        return None


def _extract_cron_context(update: Update) -> str | None:
    """If the user is replying to a cron result message, return its text.

    Cron result messages are identified by having metadata stored in the
    message text or by the bot being the sender of the replied-to message.
    We check if the replied-to message is from the bot itself and contains
    a cron marker.
    """
    message = update.message
    if message is None or message.reply_to_message is None:
        return None

    replied = message.reply_to_message

    # The replied-to message must be from the bot
    if replied.from_user is None or not replied.from_user.is_bot:
        return None

    # Check if the replied message has cron metadata in its text
    # Cron results are prefixed with a marker or stored with specific format
    text = replied.text or ""
    if not text:
        return None

    # Return the full text of the replied-to bot message as context.
    # The handler will inject it as one-off context for this turn.
    return text


# ---------------------------------------------------------------------------
# Inline keyboard callback handler
# ---------------------------------------------------------------------------

async def _handle_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle inline keyboard button presses."""
    query = update.callback_query
    if query is None:
        return

    app_state = context.bot_data.get("app_state")
    if app_state is None:
        await query.answer("Not ready")
        return

    # Owner-only
    owner_id = app_state.config.get("telegram.owner_id")
    if query.from_user is None or query.from_user.id != owner_id:
        await query.answer("Unauthorized")
        return

    data = query.data or ""
    await query.answer()

    if data.startswith("approve:"):
        tool_name = data[len("approve:"):]
        from spare_paw.tools.custom_tools import approve_tool
        import json
        result_str = await approve_tool(tool_name, app_state.tool_registry, app_state)
        result = json.loads(result_str)
        if result.get("error"):
            await query.edit_message_text(f"Error: {result['error']}")
        else:
            await query.edit_message_text(f"Tool '{tool_name}' approved and activated.")

    elif data.startswith("reject:"):
        tool_name = data[len("reject:"):]
        # Delete pending files
        from spare_paw.tools.custom_tools import PENDING_DIR
        for ext in (".json", ".sh"):
            path = PENDING_DIR / f"{tool_name}{ext}"
            if path.exists():
                path.unlink()
        await query.edit_message_text(f"Tool '{tool_name}' rejected and discarded.")


# ---------------------------------------------------------------------------
# Typing indicator
# ---------------------------------------------------------------------------

async def _send_typing_loop(
    context: ContextTypes.DEFAULT_TYPE, chat_id: int
) -> None:
    """Send 'typing' chat action every 5 seconds until cancelled."""
    try:
        while True:
            await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
            await asyncio.sleep(5)
    except asyncio.CancelledError:
        raise
    except Exception:
        # If sending typing fails, just stop silently
        pass
