"""Main message handler with async queue and backpressure.

Incoming messages are placed on an asyncio.Queue and processed sequentially.
While a message is being processed, a typing indicator is sent to signal
the bot is busy. Voice messages are transcribed via Groq Whisper before
processing.
"""

from __future__ import annotations

import asyncio
import base64
import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

import re

from telegram import Update
from telegram.constants import ChatAction, ParseMode
from telegram.ext import CallbackQueryHandler, ContextTypes, MessageHandler, filters

from pathlib import Path

from spare_paw.bot.commands import register_commands
from spare_paw.bot.voice import VoiceTranscriptionError, transcribe

# Prompt files loaded from ~/.spare-paw/ in this order
_PROMPT_DIR = Path.home() / ".spare-paw"
_PROMPT_FILES = ["IDENTITY.md", "USER.md", "SYSTEM.md"]

if TYPE_CHECKING:
    from telegram.ext import Application

logger = logging.getLogger(__name__)

# Telegram message length limit
_MAX_MESSAGE_LENGTH = 4096

# Module-level queue — initialized per-application in post_init
_message_queue: asyncio.Queue | None = None
_queue_task: asyncio.Task | None = None


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
    """Start the background task that drains the message queue."""
    global _message_queue, _queue_task
    _message_queue = asyncio.Queue()
    _queue_task = asyncio.create_task(_process_queue(application))

    # Share the queue with the subagent module for callbacks
    from spare_paw.tools import subagent as subagent_mod
    subagent_mod._message_queue = _message_queue

    logger.info("Message queue processor started")


async def _queue_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Put an incoming message on the queue for sequential processing."""
    app_state = context.bot_data.get("app_state")
    if app_state is None:
        return

    # Owner-only auth: silently ignore messages from non-owner
    owner_id = app_state.config.get("telegram.owner_id")
    if update.effective_user is None or update.effective_user.id != owner_id:
        return

    if _message_queue is not None:
        await _message_queue.put((update, context))


async def _process_queue(application: "Application") -> None:
    """Drain the message queue, processing one message at a time.

    Handles both regular Telegram messages and synthetic agent callback
    messages (pushed by subagents when a group completes).
    """
    global _message_queue
    assert _message_queue is not None

    while True:
        try:
            item = await _message_queue.get()
            try:
                # Agent callback: ("agent_callback", synthetic_text)
                if isinstance(item, tuple) and len(item) == 2 and item[0] == "agent_callback":
                    await _handle_agent_callback(item[1], application)
                else:
                    update, context = item
                    await _handle_message(update, context)
            except Exception:
                logger.exception("Unhandled error processing queue item")
                # Only try to reply if it was a regular message
                if not (isinstance(item, tuple) and item[0] == "agent_callback"):
                    try:
                        update, _ctx = item
                        await update.message.reply_text(
                            "An internal error occurred. Please try again."
                        )
                    except Exception:
                        logger.exception("Failed to send error reply")
            finally:
                _message_queue.task_done()
        except asyncio.CancelledError:
            logger.info("Message queue processor cancelled")
            break
        except Exception:
            logger.exception("Fatal error in queue processor loop")
            await asyncio.sleep(1)  # Prevent tight error loop


# ---------------------------------------------------------------------------
# Core message processing
# ---------------------------------------------------------------------------

async def _handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Process a single user message end-to-end."""
    app_state = context.bot_data["app_state"]
    chat_id = update.effective_chat.id

    # Start typing indicator in background
    typing_task = asyncio.create_task(_send_typing_loop(context, chat_id))

    try:
        # 1. Extract text + optional image
        text, image_url = await _extract_content(update, app_state)
        if not text:
            return

        # 2. Import context module
        from spare_paw import context as ctx_module

        # 3. Check if this is a reply to a cron result
        cron_context = _extract_cron_context(update)

        # 4. Get or create conversation
        conversation_id = await ctx_module.get_or_create_conversation()

        # 5. Ingest user message into context (text only for storage)
        await ctx_module.ingest(conversation_id, "user", text)

        # 6. Assemble context with system prompt + prompt files
        system_prompt = await _build_system_prompt(app_state.config)

        messages = await ctx_module.assemble(conversation_id, system_prompt)

        # 6a. If this message has an image, replace the last user message
        # with multimodal content (text + image_url)
        if image_url and messages:
            # Find the last user message and make it multimodal
            for i in range(len(messages) - 1, -1, -1):
                if messages[i]["role"] == "user":
                    messages[i]["content"] = [
                        {"type": "text", "text": text},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ]
                    break

        # 6b. If replying to a cron result, inject one-off context
        if cron_context:
            messages.append({
                "role": "user",
                "content": (
                    f"[Context: The user is replying to a cron job result. "
                    f"Original cron output:\n{cron_context}]"
                ),
            })

        # 7. Run tool loop
        model = app_state.config.get("models.default", "google/gemini-2.0-flash")
        tool_schemas = app_state.tool_registry.get_schemas()
        max_iterations = app_state.config.get("agent.max_tool_iterations", 20)

        from spare_paw.router.tool_loop import run_tool_loop

        response_text = await run_tool_loop(
            client=app_state.router_client,
            messages=messages,
            model=model,
            tools=tool_schemas,
            tool_registry=app_state.tool_registry,
            max_iterations=max_iterations,
            executor=app_state.executor,
        )

        # 8. Ingest assistant response
        await ctx_module.ingest(conversation_id, "assistant", response_text)

        # 9. Send response back via Telegram (chunked if needed)
        await _send_response(update, response_text)

    finally:
        typing_task.cancel()
        try:
            await typing_task
        except asyncio.CancelledError:
            pass


async def _handle_agent_callback(synthetic_text: str, application: "Application") -> None:
    """Process a synthetic agent callback by feeding results to the main LLM.

    The main LLM synthesizes a coherent response from the agent results
    and sends it to the user. The response is ingested into conversation
    memory so the LLM can reference it in future turns.
    """
    app_state = application.bot_data.get("app_state")
    if app_state is None:
        logger.warning("Agent callback received but app_state not available")
        return

    owner_id = app_state.config.get("telegram.owner_id")
    if not owner_id:
        return

    try:
        from spare_paw import context as ctx_module
        from spare_paw.router.tool_loop import run_tool_loop

        # Get or create conversation
        conversation_id = await ctx_module.get_or_create_conversation()

        # Ingest the agent results with instructions for the main LLM
        augmented_text = (
            f"{synthetic_text}\n\n"
            "[INSTRUCTIONS] The above are results from background agents you spawned. "
            "Present the FULL findings to the user — include all details, data, links, "
            "and comparisons the agents found. Do NOT summarize into a single sentence. "
            "Format the response clearly for Telegram."
        )
        await ctx_module.ingest(conversation_id, "user", augmented_text)

        # Assemble context with the agent results included
        system_prompt = await _build_system_prompt(app_state.config)
        messages = await ctx_module.assemble(conversation_id, system_prompt)

        # Run the main LLM to synthesize a response
        model = app_state.config.get("models.default", "google/gemini-2.0-flash")
        tool_schemas = app_state.tool_registry.get_schemas()
        max_iterations = app_state.config.get("agent.max_tool_iterations", 20)

        response_text = await run_tool_loop(
            client=app_state.router_client,
            messages=messages,
            model=model,
            tools=tool_schemas,
            tool_registry=app_state.tool_registry,
            max_iterations=max_iterations,
            executor=app_state.executor,
        )

        # Ingest the synthesized response into memory
        await ctx_module.ingest(conversation_id, "assistant", response_text)

        # Send to user via Telegram
        bot = application.bot
        chunks = _split_text(response_text, _MAX_MESSAGE_LENGTH)
        for chunk in chunks:
            try:
                await bot.send_message(
                    chat_id=owner_id, text=chunk,
                    parse_mode=ParseMode.MARKDOWN_V2,
                )
            except Exception:
                try:
                    escaped = _escape_md2(chunk)
                    await bot.send_message(
                        chat_id=owner_id, text=escaped,
                        parse_mode=ParseMode.MARKDOWN_V2,
                    )
                except Exception:
                    await bot.send_message(chat_id=owner_id, text=chunk)

    except Exception:
        logger.exception("Failed to handle agent callback")
        try:
            bot = application.bot
            await bot.send_message(
                chat_id=owner_id,
                text="Agent results received but I failed to process them. "
                     "Use /search to find the raw results.",
            )
        except Exception:
            logger.exception("Failed to send agent callback error")


async def _build_system_prompt(config: Any) -> str:
    """Build the system prompt from config + markdown files + memories.

    Loads IDENTITY.md, USER.md, and SYSTEM.md (if they exist) and appends
    them to the base system prompt from config. Also injects all persistent
    memories. Files are re-read on every call so edits take effect without restart.
    """
    base = config.get("agent.system_prompt", "")
    current_time = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    base = base.replace("{current_time}", current_time)

    sections = [base]
    for filename in _PROMPT_FILES:
        path = _PROMPT_DIR / filename
        if path.is_file():
            try:
                content = path.read_text(encoding="utf-8").strip()
                if content:
                    sections.append(content)
            except OSError:
                logger.warning("Failed to read prompt file: %s", path)

    # Load skills from ~/.spare-paw/skills/
    skills_dir = _PROMPT_DIR / "skills"
    if skills_dir.is_dir():
        for skill_path in sorted(skills_dir.glob("*.md")):
            try:
                content = skill_path.read_text(encoding="utf-8").strip()
                if content:
                    sections.append(content)
            except OSError:
                logger.warning("Failed to read skill file: %s", skill_path)

    # Inject persistent memories
    try:
        from spare_paw.tools.memory import get_all_memories
        memories = await get_all_memories()
        if memories:
            mem_lines = [f"- {m['key']}: {m['value']}" for m in memories]
            sections.append("# Memories\n" + "\n".join(mem_lines))
    except Exception:
        logger.debug("Failed to load memories for system prompt", exc_info=True)

    return "\n\n".join(sections)


async def _extract_content(
    update: Update, app_state: Any
) -> tuple[str | None, str | None]:
    """Extract text and optional base64 image from a message.

    Returns (text, image_base64). image_base64 is a data URI if a photo
    is attached, or None for text/voice-only messages.
    """
    message = update.message

    # Voice message
    if message.voice is not None:
        try:
            voice_file = await message.voice.get_file()
            text = await transcribe(voice_file, app_state.config.data)
            await message.reply_text(f"[Voice] {text}", do_quote=True)
            return text, None
        except VoiceTranscriptionError as exc:
            await message.reply_text(str(exc))
            return None, None

    # Photo message
    if message.photo:
        # Get the largest resolution photo
        photo = message.photo[-1]
        file = await photo.get_file()
        photo_bytes = await file.download_as_bytearray()
        b64 = base64.b64encode(photo_bytes).decode("ascii")
        image_url = f"data:image/jpeg;base64,{b64}"
        text = message.caption or "What do you see in this image?"
        return text, image_url

    # Text message
    if message.text:
        return message.text, None

    return None, None


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
# Response sending (MarkdownV2 with plain text fallback)
# ---------------------------------------------------------------------------

# Characters that must be escaped in MarkdownV2
_MD2_ESCAPE_RE = re.compile(r"([_*\[\]()~`>#+\-=|{}.!\\])")


def _escape_md2(text: str) -> str:
    """Escape special characters for Telegram MarkdownV2."""
    return _MD2_ESCAPE_RE.sub(r"\\\1", text)


async def _send_response(update: Update, text: str) -> None:
    """Send a response with MarkdownV2, falling back to plain text on error."""
    if not text:
        text = "(empty response)"

    chunks = _split_text(text, _MAX_MESSAGE_LENGTH)
    for chunk in chunks:
        # Try MarkdownV2 first
        try:
            await update.message.reply_text(chunk, parse_mode=ParseMode.MARKDOWN_V2)
        except Exception:
            # If MarkdownV2 fails (bad formatting), try escaped version
            try:
                escaped = _escape_md2(chunk)
                await update.message.reply_text(escaped, parse_mode=ParseMode.MARKDOWN_V2)
            except Exception:
                # Last resort: plain text
                await update.message.reply_text(chunk)


def _split_text(text: str, max_length: int) -> list[str]:
    """Split text into chunks of at most *max_length* characters.

    Prefers splitting at newlines, falling back to hard cuts.
    """
    chunks: list[str] = []
    while text:
        if len(text) <= max_length:
            chunks.append(text)
            break

        # Try to find a newline to split at
        cut_at = text.rfind("\n", 0, max_length)
        if cut_at <= 0:
            # No good newline break; hard cut
            cut_at = max_length

        chunks.append(text[:cut_at])
        text = text[cut_at:].lstrip("\n")

    return chunks


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
