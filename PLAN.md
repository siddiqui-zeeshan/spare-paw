# Plan: Fully decouple the bot from Telegram via `MessageBackend` protocol

## Goal

**Complete decoupling.** After this refactor, the `bot/` package can be deleted entirely and the core still compiles and runs. Zero Telegram imports exist outside `bot/`.

The bot can run against **Telegram** or a **plain HTTP webhook** (for Docker/CI testing). The HTTP backend requires zero external dependencies — just `aiohttp` (already a dependency).

Eventually, adding new frontends (Discord, Slack, WhatsApp) should only require implementing one more `MessageBackend`.

## Design principles

1. **`core/` never imports `telegram`** — not even transitively. All Telegram-specific logic lives in `bot/`.
2. **Engine outputs markdown** — each backend converts to its own format (Telegram→HTML, webhook→raw markdown).
3. **Backends own their formatting** — `_md_to_html()`, `_convert_tables()`, chunking at 4096 chars all live in `TelegramBackend`, not core.
4. **`IncomingMessage` is self-contained** — backends populate all fields (including `cron_context`) before handing off to the engine. The engine never needs to inspect platform-specific reply structures.
5. **`voice.py` moves to `core/`** — it accepts raw `bytes`, not `TelegramFile`. The Groq Whisper API call has nothing to do with Telegram.

---

## Architecture

```
                    ┌──────────────────────────────┐
                    │         gateway.py            │
                    │  (config, DB, tools, router)  │
                    └──────────┬───────────────────┘
                               │
                    ┌──────────▼───────────────────┐
                    │      core/engine.py           │
                    │  message loop, tool loop,     │
                    │  context assembly, callbacks   │
                    │  (100% platform-agnostic)     │
                    └──────────┬───────────────────┘
                               │ uses MessageBackend
              ┌────────────────┼────────────────────┐
              │                │                     │
    ┌─────────▼──────┐  ┌─────▼──────────┐  ┌──────▼──────────┐
    │ TelegramBackend │  │ WebhookBackend │  │ future backends │
    │  (bot/ package) │  │ (webhook/ pkg) │  │                 │
    └────────────────┘  └────────────────┘  └─────────────────┘
```

---

## Step 1: Define the `MessageBackend` protocol and `IncomingMessage` dataclass

**New file:** `src/spare_paw/backend.py`

```python
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

@dataclass
class IncomingMessage:
    """Platform-agnostic representation of a user message.

    Backends populate ALL fields before handing off to the engine.
    The engine never needs to inspect platform-specific structures.
    """
    text: str | None = None
    image_bytes: bytes | None = None        # raw image (JPEG/PNG)
    image_mime: str = "image/jpeg"
    voice_bytes: bytes | None = None        # raw audio (OGG)
    caption: str | None = None              # caption on photo/file
    cron_context: str | None = None         # text of the cron result being replied to (backend detects this)
    command: str | None = None              # e.g. "status", "forget" — if this is a command, not a message
    command_args: list[str] = field(default_factory=list)  # args for the command
    user_id: int | str | None = None        # sender identity (for auth)

@runtime_checkable
class MessageBackend(Protocol):
    """Interface every messaging backend must implement.

    The engine outputs markdown. Each backend is responsible for
    converting to its own format (HTML for Telegram, raw for webhook).
    Chunking is also the backend's responsibility.
    """

    async def send_text(self, text: str) -> None:
        """Send a text message to the owner.

        The text is markdown. Backend handles format conversion and chunking.
        """
        ...

    async def send_file(self, path: str, caption: str = "") -> None:
        """Send a file (photo/video/audio/document) to the owner."""
        ...

    async def send_typing(self) -> None:
        """Signal that the bot is processing. No-op if unsupported."""
        ...

    async def start(self) -> None:
        """Start receiving messages (polling, webhook server, etc.)."""
        ...

    async def stop(self) -> None:
        """Graceful shutdown."""
        ...
```

This is intentionally minimal. No `InlineKeyboard`, no `edit_message`, no `parse_mode` — those are Telegram-specific and will stay in `TelegramBackend` as extensions. The `send_text()` contract is simple: you receive markdown, you figure out how to display it.

---

## Step 2: Create `core/engine.py` — the platform-agnostic message processor

**New file:** `src/spare_paw/core/__init__.py` (empty)
**New file:** `src/spare_paw/core/engine.py`

Extract from `handler.py` everything that doesn't touch Telegram:

- `_build_system_prompt()` → moves here (already platform-agnostic)
- `_split_text()` → moves here (generic utility, used by backends too — they can import it from core)
- The core of `_handle_message()` becomes `process_message(app_state, msg: IncomingMessage, backend: MessageBackend)`:
  1. Voice transcription (if `msg.voice_bytes`) — call `core/voice.py` with raw bytes
  2. Image handling (if `msg.image_bytes`) — base64 encode
  3. Context assembly (conversation, system prompt, memories)
  4. Cron context injection (if `msg.cron_context` — already extracted by the backend)
  5. Tool loop execution
  6. Ingest response
  7. LCM compaction (background)
  8. `await backend.send_text(response_text)` — engine passes markdown, backend handles conversion + chunking
- The core of `_handle_agent_callback()` becomes `process_agent_callback(app_state, synthetic_text, backend)` — same pattern
- Message queue stays here: `start_queue_processor()`, `_process_queue()` now work with `(IncomingMessage | ("agent_callback", str))` tuples

**What does NOT move to core:**
- `_md_to_html()` and `_convert_tables()` → these are Telegram-specific formatting. They move to `bot/backend.py` as private methods of `TelegramBackend`.
- `_send_response()` → replaced by `backend.send_text()`.
- `_extract_cron_context()` → stays in `bot/handler.py`. The Telegram adapter populates `msg.cron_context` before queueing. The webhook adapter populates it from the HTTP request body.

**Key invariant:** `core/` never imports from `bot/`, `webhook/`, or `telegram`. It only depends on `backend.py`, `context.py`, `db.py`, `router/`, `tools/`, and `config.py`.

---

## Step 3: Create `core/commands.py` — platform-agnostic command dispatch

**New file:** `src/spare_paw/core/commands.py`

Extract the business logic from `bot/commands.py`. Each command becomes a pure function:

```python
async def cmd_status(app_state: AppState) -> str:
    """Return status text."""
    ...

async def cmd_forget(app_state: AppState) -> str:
    ...

async def cmd_search(app_state: AppState, query: str) -> str:
    ...

async def cmd_cron(app_state: AppState, subcommand: str, args: list[str]) -> str:
    ...

# etc.

COMMANDS: dict[str, Callable] = {
    "status": cmd_status,
    "forget": cmd_forget,
    "search": cmd_search,
    "cron": cmd_cron,
    "config": cmd_config,
    "model": cmd_model,
    "tools": cmd_tools,
    "approve": cmd_approve,
    "agents": cmd_agents,
    "logs": cmd_logs,
    "mcp": cmd_mcp,
}
```

Each returns a `str` response. The backend-specific layer (Telegram handler or HTTP endpoint) calls the function and sends the result.

---

## Step 4: Refactor `bot/` into `TelegramBackend`

**Modified files:**
- `bot/handler.py` → thin wrapper that creates `IncomingMessage` from `Update` and delegates to `core/engine.py`
- `bot/commands.py` → thin wrapper that parses Telegram `Update` args and delegates to `core/commands.py`

**Deleted file:** `bot/voice.py` — voice transcription moves to `core/voice.py` (see Step 2b)

**New file:** `src/spare_paw/bot/backend.py`

This file owns ALL Telegram-specific formatting and delivery:

```python
from telegram.constants import ChatAction, ParseMode

from spare_paw.backend import MessageBackend
from spare_paw.core.engine import split_text  # generic utility


class TelegramBackend(MessageBackend):
    """Telegram implementation of the messaging backend.

    Owns markdown→HTML conversion, 4096-char chunking, and all
    bot.send_*() calls. Nothing outside bot/ imports telegram.
    """

    _MAX_LENGTH = 4096

    def __init__(self, application, owner_id: int):
        self._application = application
        self.bot = application.bot
        self.owner_id = owner_id

    async def send_text(self, text: str) -> None:
        """Convert markdown to Telegram HTML, chunk, and send."""
        if not text:
            text = "(empty response)"
        chunks = split_text(text, self._MAX_LENGTH)
        for chunk in chunks:
            try:
                html = self._md_to_html(chunk)
                await self.bot.send_message(
                    chat_id=self.owner_id, text=html,
                    parse_mode=ParseMode.HTML,
                )
            except Exception:
                await self.bot.send_message(chat_id=self.owner_id, text=chunk)

    async def send_file(self, path: str, caption: str = "") -> None:
        # Move existing _send_file logic from gateway.py here
        # (bot.send_photo/video/audio/document based on suffix)
        ...

    async def send_typing(self) -> None:
        await self.bot.send_chat_action(
            chat_id=self.owner_id, action=ChatAction.TYPING,
        )

    async def start(self) -> None:
        await self._application.initialize()
        await self._application.start()
        if self._application.updater is not None:
            await self._application.updater.start_polling(drop_pending_updates=True)

    async def stop(self) -> None:
        if self._application.updater is not None:
            await self._application.updater.stop()
        await self._application.stop()
        await self._application.shutdown()

    # --- Private: Telegram-specific formatting ---

    def _md_to_html(self, text: str) -> str:
        """Markdown → Telegram HTML. Moved from handler.py."""
        ...

    def _convert_tables(self, text: str) -> str:
        """Markdown tables → monospace <pre> blocks. Moved from handler.py."""
        ...
```

**`bot/handler.py` becomes:**
```python
async def _queue_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    # Auth check (owner_id)
    # Build IncomingMessage from Update — ALL extraction happens here:
    msg = IncomingMessage(
        text=update.message.text,
        voice_bytes=await _download_voice(update) if update.message.voice else None,
        image_bytes=await _download_photo(update) if update.message.photo else None,
        caption=update.message.caption,
        cron_context=_extract_cron_context(update),  # Telegram-specific detection, stays here
        user_id=update.effective_user.id,
    )
    await engine.enqueue(msg)
```

**`_extract_cron_context()` stays in `bot/handler.py`** — it inspects `reply_to_message.from_user.is_bot`, which is a Telegram concept. The webhook backend has its own way of providing cron context (explicit `cron_id` in the HTTP body).

The queue processor calls `engine.process_message(app_state, msg, backend)`.

---

## Step 5: Create `webhook/` package — the HTTP backend

**New files:**
- `src/spare_paw/webhook/__init__.py`
- `src/spare_paw/webhook/backend.py`
- `src/spare_paw/webhook/server.py`

### `webhook/server.py` — aiohttp web server

```
POST /message          {"text": "...", "image_b64": "...", "voice_b64": "..."}
POST /command/{name}   {"args": "..."}
GET  /events           SSE stream of bot responses
GET  /health           {"status": "ok", "uptime": 123}
```

### `webhook/backend.py`

```python
class WebhookBackend(MessageBackend):
    def __init__(self, owner_id: str | None = None):
        self._response_queue: asyncio.Queue[dict] = asyncio.Queue()

    async def send_text(self, text: str, *, parse_mode: str | None = None) -> None:
        # Push to SSE response queue (no HTML conversion, plain markdown)
        await self._response_queue.put({"type": "text", "content": text})

    async def send_file(self, path: str, caption: str = "") -> None:
        # Push file path/URL to response queue
        await self._response_queue.put({"type": "file", "path": path, "caption": caption})

    async def send_typing(self) -> None:
        await self._response_queue.put({"type": "typing"})

    async def start(self) -> None:
        # Start aiohttp web server on configured port
        ...

    async def stop(self) -> None:
        # Stop aiohttp web server
        ...
```

No Telegram dependency. No `python-telegram-bot` import. Can run in a Docker container with just `aiohttp`.

---

## Step 6: Refactor `gateway.py` — backend selection

**Modified file:** `src/spare_paw/gateway.py`

The gateway becomes backend-agnostic. It:

1. Loads config, sets up DB, executor, semaphore, router, tools (unchanged)
2. Reads `mode` from config or CLI arg (`--mode telegram|webhook`, default `telegram`)
3. Creates the appropriate backend:
   ```python
   if mode == "webhook":
       from spare_paw.webhook.backend import WebhookBackend
       backend = WebhookBackend(host=host, port=port)
   else:
       from spare_paw.bot.backend import TelegramBackend
       backend = TelegramBackend(bot_token, owner_id)
   ```
4. Stores `backend` on `AppState` (replaces `application` field)
5. Starts the engine's queue processor with the backend
6. `await backend.start()` / `await backend.stop()` for lifecycle

**AppState changes:**
```python
@dataclass
class AppState:
    config: Config
    executor: ProcessPoolExecutor
    semaphore: asyncio.Semaphore
    tool_registry: Any = None
    router_client: Any = None
    backend: MessageBackend | None = None    # replaces 'application'
    scheduler: Any = None
    mcp_client: Any = None
    start_time: datetime = ...
```

---

## Step 7: Refactor `send_message` / `send_file` tools

**Modified file:** `src/spare_paw/gateway.py` (tool registration section)

Instead of reaching into `app_state.application.bot`, the tools call `app_state.backend`:

```python
async def _send_message(text: str) -> str:
    if not app_state.backend:
        return json.dumps({"error": "No messaging backend available"})
    await app_state.backend.send_text(text)
    return json.dumps({"success": True})

async def _send_file(path: str, caption: str = "") -> str:
    if not app_state.backend:
        return json.dumps({"error": "No messaging backend available"})
    await app_state.backend.send_file(path, caption)
    return json.dumps({"success": True})
```

---

## Step 8: Refactor `cron/executor.py`

**Modified file:** `src/spare_paw/cron/executor.py`

Replace `app_state.application.bot` → `app_state.backend`:

```python
# Before:
bot = app_state.application.bot
await _send_chunked(bot, owner_id, result)

# After:
if app_state.backend:
    await app_state.backend.send_text(result)
```

Delete `_send_chunked()` — chunking is now the backend's responsibility.

---

## Step 2b: Move `voice.py` to `core/voice.py`

**Move:** `src/spare_paw/bot/voice.py` → `src/spare_paw/core/voice.py`

The only Telegram dependency was `TelegramFile`. Remove it — accept raw `bytes`:

```python
# Before (bot/voice.py):
from telegram import File as TelegramFile

async def transcribe(voice_file: TelegramFile, config: dict) -> str:
    voice_bytes = await voice_file.download_as_bytearray()
    ...

# After (core/voice.py):
async def transcribe(voice_bytes: bytes, config: dict) -> str:
    # voice_bytes already provided by caller
    # Rest of the Groq Whisper API call is unchanged
    ...
```

The Telegram handler downloads bytes via `voice.get_file().download_as_bytearray()` before calling `transcribe()`. The webhook handler receives base64 bytes from the HTTP request body. The engine in `core/engine.py` calls `core.voice.transcribe(msg.voice_bytes, config)` — no Telegram import anywhere in the chain.

---

## Step 10: Update `__main__.py` — add `webhook` command

```python
elif command == "gateway":
    from spare_paw.gateway import run
    run()
elif command == "webhook":
    from spare_paw.gateway import run
    run(mode="webhook")
```

Or use `--mode webhook` flag parsed in gateway.

---

## Step 11: Update `subagent.py` queue interaction

**Modified file:** `src/spare_paw/tools/subagent.py`

No change needed — subagents push `("agent_callback", text)` to the module-level `_message_queue`. The queue processor in `core/engine.py` handles it the same way regardless of backend.

---

## Step 12: Inline keyboard callbacks (Telegram-only)

The `_handle_callback` function for tool approval buttons stays in `bot/handler.py` — it's a Telegram-specific UI feature. The webhook backend can expose tool approval via `POST /command/approve` instead.

---

## File change summary

| File | Action |
|------|--------|
| `src/spare_paw/backend.py` | **NEW** — `MessageBackend` protocol + `IncomingMessage` dataclass |
| `src/spare_paw/core/__init__.py` | **NEW** — empty |
| `src/spare_paw/core/engine.py` | **NEW** — platform-agnostic message processor + queue (extracted from handler.py) |
| `src/spare_paw/core/voice.py` | **MOVE** from `bot/voice.py` — accepts `bytes`, no Telegram import |
| `src/spare_paw/core/commands.py` | **NEW** — platform-agnostic command logic (extracted from bot/commands.py) |
| `src/spare_paw/webhook/__init__.py` | **NEW** — empty |
| `src/spare_paw/webhook/backend.py` | **NEW** — `WebhookBackend` implementation |
| `src/spare_paw/webhook/server.py` | **NEW** — aiohttp HTTP server |
| `src/spare_paw/bot/backend.py` | **NEW** — `TelegramBackend` (owns `_md_to_html`, `_convert_tables`, chunking, `send_file` media dispatch) |
| `src/spare_paw/bot/handler.py` | **MODIFY** — thin Telegram→IncomingMessage adapter, delegates to core/engine |
| `src/spare_paw/bot/commands.py` | **MODIFY** — thin wrapper, delegates to core/commands.py |
| `src/spare_paw/bot/voice.py` | **DELETE** — moved to core/voice.py |
| `src/spare_paw/cron/executor.py` | **MODIFY** — use `backend.send_text()` instead of `bot.send_message()`, delete `_send_chunked()` |
| `src/spare_paw/gateway.py` | **MODIFY** — `AppState.backend` replaces `.application`, mode selection, tool rewiring |
| `src/spare_paw/__main__.py` | **MODIFY** — add `webhook` command |
| `src/spare_paw/tools/subagent.py` | **NO CHANGE** — already decoupled via queue |

### Telegram isolation verification

After this refactor, grep for `telegram` imports across the codebase:

- `bot/backend.py` — yes (implements TelegramBackend)
- `bot/handler.py` — yes (converts Update → IncomingMessage)
- `bot/commands.py` — yes (thin wrapper for command parsing)
- **Everything else** — zero Telegram imports

Deleting the `bot/` directory should leave a fully functional core that works with any `MessageBackend`.

---

## Implementation order

1. **backend.py** — protocol + dataclass (no dependencies, can test immediately)
2. **core/voice.py** — move from bot/voice.py, change signature to accept bytes
3. **core/engine.py** — extract from handler.py (biggest refactor, most risk)
4. **core/commands.py** — extract from bot/commands.py
5. **bot/backend.py** — TelegramBackend (move `_md_to_html`, `_convert_tables`, `_send_file` here)
6. **bot/handler.py** — slim down to Telegram→IncomingMessage adapter
7. **bot/commands.py** — slim down to Telegram wrapper
8. **gateway.py** — `AppState.backend`, mode selection, tool rewiring
9. **cron/executor.py** — swap to `backend.send_text()`
10. **webhook/** — new HTTP backend
11. **__main__.py** — add webhook command
12. **Tests** — for engine, webhook backend, and integration

Steps 1-9 don't change any behavior — Telegram mode works exactly as before. Step 10 adds the new capability. This ordering minimizes risk.

---

## Testing strategy

- **Unit tests for `core/engine.py`**: Mock `MessageBackend`, feed `IncomingMessage` objects, verify tool loop + context assembly + `backend.send_text()` called with markdown
- **Unit tests for `core/commands.py`**: Call each command function directly, verify string output
- **Unit tests for `core/voice.py`**: Mock aiohttp, pass raw bytes, verify Groq API call
- **Unit tests for `TelegramBackend`**: Verify `_md_to_html()` output, chunking at 4096, HTML fallback to plain text
- **Unit tests for `WebhookBackend`**: Start server, POST messages, read SSE stream, verify responses
- **Integration test**: Start gateway in webhook mode, send a message via HTTP, verify response comes back through SSE
- **Existing tests**: Should pass unchanged (they don't import Telegram)
- **Isolation check**: CI step that greps for `import telegram` or `from telegram` outside `bot/` — fails if found
