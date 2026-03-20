"""Platform-agnostic message processor.

Handles message processing, agent callbacks, and the message queue.
No Telegram imports — all platform-specific behavior is delegated
to the MessageBackend.
"""

from __future__ import annotations

import asyncio
import base64
import logging
from typing import TYPE_CHECKING, Any

from spare_paw import context as ctx_module
from spare_paw.context import compact_with_retry
from spare_paw.core.prompt import build_system_prompt
from spare_paw.core.voice import VoiceTranscriptionError, transcribe
from spare_paw.router.tool_loop import run_tool_loop

if TYPE_CHECKING:
    from spare_paw.backend import IncomingMessage, MessageBackend

logger = logging.getLogger(__name__)

# Module-level queue — initialized by start_queue_processor
_message_queue: asyncio.Queue | None = None
_queue_task: asyncio.Task | None = None


def split_text(text: str, max_length: int) -> list[str]:
    """Split text into chunks of at most *max_length* characters.

    Prefers splitting at newlines, falling back to hard cuts.
    """
    if not text:
        return [text]
    chunks: list[str] = []
    while text:
        if len(text) <= max_length:
            chunks.append(text)
            break

        cut_at = text.rfind("\n", 0, max_length)
        if cut_at <= 0:
            cut_at = max_length

        chunks.append(text[:cut_at])
        text = text[cut_at:].lstrip("\n")

    return chunks


async def process_message(
    app_state: Any,
    msg: IncomingMessage,
    backend: MessageBackend,
) -> None:
    """Process a single user message end-to-end.

    1. Extract text from message (voice transcription if needed)
    2. Handle image (base64 encode)
    3. Assemble context window
    4. Inject cron context if present
    5. Run tool loop
    6. Ingest response + send via backend
    7. Background LCM compaction
    """
    ctx = ctx_module

    # 1. Determine text content
    text = msg.text
    image_url = None

    if msg.voice_bytes:
        try:
            config_data = getattr(app_state.config, "data", app_state.config)
            text = await transcribe(msg.voice_bytes, config_data)
        except VoiceTranscriptionError:
            logger.exception("Voice transcription failed")
            return

    if msg.image_bytes:
        b64 = base64.b64encode(msg.image_bytes).decode("ascii")
        image_url = f"data:{msg.image_mime};base64,{b64}"
        text = msg.caption or "What do you see in this image?"

    if not text:
        return

    # 2. Get or create conversation
    conversation_id = await ctx.get_or_create_conversation()

    # 3. Ingest user message
    await ctx.ingest(conversation_id, "user", text)

    # 4. Assemble context
    system_prompt = await build_system_prompt(app_state.config)
    messages = await ctx.assemble(conversation_id, system_prompt)

    # 5. Image: make last user message multimodal
    if image_url and messages:
        for i in range(len(messages) - 1, -1, -1):
            if messages[i]["role"] == "user":
                messages[i]["content"] = [
                    {"type": "text", "text": text},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ]
                break

    # 6. Cron context injection
    if msg.cron_context:
        messages.append({
            "role": "user",
            "content": (
                f"[Context: The user is replying to a cron job result. "
                f"Original cron output:\n{msg.cron_context}]"
            ),
        })

    # 7. Run tool loop (duck-type callbacks from the backend)
    on_event = getattr(backend, "on_tool_event", None)
    on_token = getattr(backend, "on_token", None)

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
        on_event=on_event,
        on_token=on_token,
    )

    # 8. Ingest assistant response
    await ctx.ingest(conversation_id, "assistant", response_text)

    # 9. LCM compaction in background
    summary_model = app_state.config.get(
        "context.summary_model", "google/gemini-3.1-flash-lite-preview"
    )
    asyncio.create_task(
        compact_with_retry(conversation_id, app_state.router_client, summary_model),
        name="lcm-compact",
    )

    # 10. Send response via backend (markdown — backend handles formatting)
    await backend.send_text(response_text)


async def process_agent_callback(
    app_state: Any,
    synthetic_text: str,
    backend: MessageBackend,
) -> None:
    """Process a synthetic agent callback by feeding results to the main LLM.

    The main LLM synthesizes a coherent response from the agent results
    and sends it to the user via the backend.
    """
    ctx = ctx_module

    try:
        conversation_id = await ctx.get_or_create_conversation()

        augmented_text = (
            f"{synthetic_text}\n\n"
            "[INSTRUCTIONS] The above are results from background agents you spawned. "
            "Present the FULL findings to the user — include all details, data, links, "
            "and comparisons the agents found. Do NOT summarize into a single sentence. "
            "Format the response clearly."
        )
        await ctx.ingest(conversation_id, "user", augmented_text)

        system_prompt = await build_system_prompt(app_state.config)
        messages = await ctx.assemble(conversation_id, system_prompt)

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

        await ctx.ingest(conversation_id, "assistant", response_text)
        await backend.send_text(response_text)

    except Exception:
        logger.exception("Failed to handle agent callback")
        try:
            await backend.send_text(
                "Agent results received but I failed to process them. "
                "Use /search to find the raw results."
            )
        except Exception:
            logger.exception("Failed to send agent callback error")


# ---------------------------------------------------------------------------
# Queue management
# ---------------------------------------------------------------------------


async def enqueue(item: IncomingMessage | tuple) -> None:
    """Put a message or agent callback on the processing queue."""
    if _message_queue is not None:
        await _message_queue.put(item)


def start_queue_processor(app_state: Any, backend: MessageBackend) -> None:
    """Start the background task that drains the message queue."""
    global _message_queue, _queue_task
    _message_queue = asyncio.Queue()
    _queue_task = asyncio.create_task(_process_queue(app_state, backend))

    # Share the queue with the subagent module for callbacks
    from spare_paw.tools import subagent as subagent_mod
    subagent_mod._message_queue = _message_queue

    logger.info("Message queue processor started")


async def _process_queue(app_state: Any, backend: MessageBackend) -> None:
    """Drain the message queue, processing one message at a time."""
    from spare_paw.backend import IncomingMessage as _IncomingMessage

    assert _message_queue is not None

    while True:
        try:
            item = await _message_queue.get()
            try:
                if isinstance(item, tuple) and len(item) == 2 and item[0] == "agent_callback":
                    await process_agent_callback(app_state, item[1], backend)
                elif isinstance(item, _IncomingMessage):
                    # Start typing indicator
                    await backend.send_typing()
                    await process_message(app_state, item, backend)
                else:
                    logger.warning("Unknown item type in queue: %s", type(item))
            except Exception:
                logger.exception("Unhandled error processing queue item")
                try:
                    await backend.send_text(
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
            await asyncio.sleep(1)
