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
6. **No Telegram imports outside `bot/`** — this includes `tools/custom_tools.py`, `gateway.py`, and `cron/executor.py`. All Telegram-specific operations are routed through the backend.

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
                    └──┬───────────────────────┬───┘
                       │ uses MessageBackend   │ uses core/prompt.py
              ┌────────┼──────────────┐    ┌───▼──────────────┐
              │        │              │    │  core/prompt.py   │
    ┌─────────▼──────┐ │ ┌────────────▼┐  │  (leaf module,    │
    │ TelegramBackend│ │ │future b'ends│  │   no tool/engine  │
    │  (bot/ package)│ │ │             │  │   imports)        │
    └────────────────┘ │ └─────────────┘  └───────────────────┘
              ┌────────▼──────────┐
              │  WebhookBackend   │
              │  (webhook/ pkg)   │
              └───────────────────┘
```

**`core/prompt.py` is a leaf module** — it imports only from `config.py` and `tools/memory.py` (for memory injection). It does NOT import from `core/engine.py`, `tools/subagent.py`, or `router/`. This eliminates circular import risk: both `subagent.py` and `cron/executor.py` can import `_build_system_prompt` from `core/prompt.py` without creating cycles.

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
        """Send a file (photo/video/audio/document) to the owner.

        Backend handles media type detection (photo vs video vs audio vs
        document) based on file extension or MIME type.
        """
        ...

    async def send_typing(self) -> None:
        """Signal that the bot is processing. No-op if unsupported."""
        ...

    async def send_notification(self, text: str, actions: list[dict] | None = None) -> None:
        """Send a notification with optional action buttons.

        Used for tool approval requests, alerts, etc. The `actions` list
        contains dicts like {"label": "Approve", "callback_data": "approve:foo"}.
        Backends that don't support buttons just send plain text.
        """
        ...

    async def start(self) -> None:
        """Start receiving messages (polling, webhook server, etc.)."""
        ...

    async def stop(self) -> None:
        """Graceful shutdown."""
        ...
```

This is intentionally minimal. No `InlineKeyboard`, no `edit_message`, no `parse_mode` — those are Telegram-specific and will stay in `TelegramBackend` as extensions. The `send_text()` contract is simple: you receive markdown, you figure out how to display it.

The `send_notification()` method handles the tool approval use case — `custom_tools.py` currently imports `InlineKeyboardButton/InlineKeyboardMarkup` directly from telegram. After this refactor, it calls `backend.send_notification()` instead, and `TelegramBackend` renders that as inline keyboard buttons while `WebhookBackend` exposes it as a JSON event.

---

## Step 2: Create `core/prompt.py` and `core/engine.py` — platform-agnostic core

**New files:**
- `src/spare_paw/core/__init__.py` (empty)
- `src/spare_paw/core/prompt.py` — **leaf module**, no imports from engine/tools/router
- `src/spare_paw/core/engine.py`

### `core/prompt.py` — system prompt builder (leaf module)

`_build_system_prompt()` moves here — NOT into `engine.py`. This is critical for avoiding circular imports:
- `tools/subagent.py` imports `_build_system_prompt` (currently from `bot/handler.py`)
- `cron/executor.py` imports `_build_system_prompt` (currently from `bot/handler.py`)
- If `_build_system_prompt` lived in `engine.py`, and `engine.py` imported from `tools/`, we'd have `engine → tools → engine` cycle risk

By putting it in `core/prompt.py` (which only imports from `config` and `tools/memory`), both `subagent.py` and `cron/executor.py` can import it safely. `engine.py` also imports from `core/prompt.py` — no cycle.

**Note:** `cron/executor.py` and `tools/subagent.py` must update their import to `core.prompt._build_system_prompt`.

### `core/engine.py` — message processor

Extract from `handler.py` everything that doesn't touch Telegram:

- `_split_text()` → moves here (generic utility, used by backends too — they can import it from core)
- The core of `_handle_message()` becomes `process_message(app_state, msg: IncomingMessage, backend: MessageBackend)`:
  1. Voice transcription (if `msg.voice_bytes`) — call `core/voice.py` with raw bytes
  2. Image handling (if `msg.image_bytes`) — base64 encode
  3. Context assembly (conversation, system prompt, memories)
  4. Cron context injection (if `msg.cron_context` — already extracted by the backend)
  5. Tool loop execution via `run_tool_loop()` (import from `router/tool_loop.py`)
  6. Ingest response
  7. LCM compaction (background)
  8. `await backend.send_text(response_text)` — engine passes markdown, backend handles conversion + chunking
- The core of `_handle_agent_callback()` becomes `process_agent_callback(app_state, synthetic_text, backend)` — same pattern. **Important:** the current handler.py has a separate send path here that does its own `_md_to_html()` + `bot.send_message()` bypassing `_send_response()`. After the refactor, this path also goes through `backend.send_text()`, unifying the two send paths.
- Message queue moves here: `start_queue_processor()`, `_process_queue()` now work with `(IncomingMessage | ("agent_callback", str))` tuples. The queue is module-level in `core/engine.py`.

**What does NOT move to core:**
- `_md_to_html()` and `_convert_tables()` → these are Telegram-specific formatting. They move to `bot/backend.py` as private methods of `TelegramBackend`.
- `_send_response()` → replaced by `backend.send_text()`.
- `_extract_cron_context()` → stays in `bot/handler.py`. The Telegram adapter populates `msg.cron_context` before queueing. The webhook adapter populates it from the HTTP request body.
- `_extract_content()` → the voice/photo download logic stays in `bot/handler.py` as `_download_voice()` and `_download_photo()`. These are Telegram-specific (calling `voice.get_file().download_as_bytearray()` and `photo[-1].get_file().download_as_bytearray()`). The extracted bytes populate `IncomingMessage.voice_bytes` / `IncomingMessage.image_bytes`. The engine never downloads anything — it receives raw bytes.

**Key invariant:** `core/` never imports from `bot/`, `webhook/`, or `telegram`. It only depends on `backend.py`, `context.py`, `db.py`, `router/`, `tools/`, and `config.py`.

**`core/prompt.py` invariant:** This is a **leaf module**. It imports only from `config.py` and `tools/memory.py`. It does NOT import from `core/engine.py`, `router/`, or any other `tools/` module. This prevents circular import chains when `subagent.py` and `cron/executor.py` import `_build_system_prompt`.

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
from telegram import InlineKeyboardButton, InlineKeyboardMarkup

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
        """Send file with media type detection based on extension.

        Moves existing logic from gateway.py's send_file tool:
        - .jpg/.png/.gif/.webp → send_photo
        - .mp4/.mov/.avi → send_video
        - .mp3/.ogg/.m4a/.wav → send_audio
        - everything else → send_document
        """
        suffix = Path(path).suffix.lower()
        with open(path, "rb") as f:
            if suffix in {".jpg", ".jpeg", ".png", ".gif", ".webp"}:
                await self.bot.send_photo(chat_id=self.owner_id, photo=f, caption=caption)
            elif suffix in {".mp4", ".mov", ".avi"}:
                await self.bot.send_video(chat_id=self.owner_id, video=f, caption=caption)
            elif suffix in {".mp3", ".ogg", ".m4a", ".wav"}:
                await self.bot.send_audio(chat_id=self.owner_id, audio=f, caption=caption)
            else:
                await self.bot.send_document(chat_id=self.owner_id, document=f, caption=caption)

    async def send_typing(self) -> None:
        await self.bot.send_chat_action(
            chat_id=self.owner_id, action=ChatAction.TYPING,
        )

    async def send_notification(self, text: str, actions: list[dict] | None = None) -> None:
        """Send notification with optional inline keyboard buttons."""
        reply_markup = None
        if actions:
            buttons = [
                InlineKeyboardButton(a["label"], callback_data=a["callback_data"])
                for a in actions
            ]
            reply_markup = InlineKeyboardMarkup([buttons])
        await self.bot.send_message(
            chat_id=self.owner_id, text=text, reply_markup=reply_markup,
        )

    async def start(self) -> None:
        await self._application.initialize()
        # Register handlers BEFORE starting polling
        from spare_paw.bot.handler import setup_handlers
        setup_handlers(self._application)
        # Set bot command menu (Telegram-specific)
        from telegram import BotCommand
        commands = [
            BotCommand("status", "Show bot status"),
            BotCommand("forget", "Start new conversation"),
            # ... etc
        ]
        await self.bot.set_my_commands(commands)
        await self._application.start()
        if self._application.updater is not None:
            await self._application.updater.start_polling(drop_pending_updates=True)

    async def stop(self) -> None:
        if self._application.updater is not None:
            await self._application.updater.stop()
        await self._application.stop()
        await self._application.shutdown()

    def set_app_state(self, app_state) -> None:
        """Store app_state on bot_data so Telegram handlers can access it."""
        self._application.bot_data["app_state"] = app_state

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

**`_extract_content()` decomposition:** The existing `_extract_content()` function combines voice download, voice transcription, photo download, and text extraction. After the refactor:
- Voice/photo **download** stays in `bot/handler.py` as `_download_voice()` / `_download_photo()` (Telegram API calls)
- Voice **transcription** moves to `core/voice.py` (called by the engine, not the handler)
- Text extraction is trivial (`update.message.text` or `update.message.caption`)

**`_extract_cron_context()` stays in `bot/handler.py`** — it inspects `reply_to_message.from_user.is_bot`, which is a Telegram concept. The webhook backend has its own way of providing cron context (explicit `cron_id` in the HTTP body).

**`_handle_callback()` stays in `bot/handler.py`** — it handles inline keyboard presses for tool approval. It accesses `app_state` via `context.bot_data["app_state"]` (still valid since `TelegramBackend.set_app_state()` stores it on `application.bot_data`).

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

    async def send_text(self, text: str) -> None:
        # Push to SSE response queue (no HTML conversion, plain markdown)
        await self._response_queue.put({"type": "text", "content": text})

    async def send_file(self, path: str, caption: str = "") -> None:
        # Push file path/URL to response queue
        await self._response_queue.put({"type": "file", "path": path, "caption": caption})

    async def send_typing(self) -> None:
        await self._response_queue.put({"type": "typing"})

    async def send_notification(self, text: str, actions: list[dict] | None = None) -> None:
        # Push as JSON event — client handles rendering
        await self._response_queue.put({
            "type": "notification", "content": text, "actions": actions or [],
        })

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

**What moves OUT of gateway.py:**
- `set_my_commands()` → moves to `TelegramBackend.start()` (Telegram-specific)
- `setup_handlers(application)` → called by `TelegramBackend.start()` (Telegram-specific)
- `send_file` media dispatch logic (photo/video/audio/document) → moves to `TelegramBackend.send_file()`
- `send_message` chunking logic → handled by `backend.send_text()` (backend owns chunking)
- `from telegram import BotCommand` import → moves to `bot/backend.py`
- `from telegram.ext import Application` import → moves to `bot/backend.py`

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

**`app_state` wiring:** Currently `app_state` is stored on `application.bot_data["app_state"]` and all Telegram handlers access it via `context.bot_data["app_state"]`. After the refactor:
- `gateway.py` creates backend, sets `app_state.backend = backend`
- For Telegram: calls `backend.set_app_state(app_state)` which stores it on `application.bot_data["app_state"]`
- Telegram handlers continue accessing `app_state` via `context.bot_data["app_state"]` — unchanged from their perspective
- For webhook: `app_state` is passed directly to the engine — no `bot_data` needed

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

**Note:** The current `send_message` tool does its own chunking at 4096 chars. After the refactor, `backend.send_text()` owns chunking — remove the redundant chunking from the tool. The current `send_file` tool has media type dispatch (photo/video/audio/document based on suffix) — this moves into `TelegramBackend.send_file()`. The tool becomes a thin wrapper.

---

## Step 8: Refactor `cron/executor.py`

**Modified file:** `src/spare_paw/cron/executor.py`

Two changes:

### 8a: Update `_build_system_prompt` import

```python
# Before:
from spare_paw.bot.handler import _build_system_prompt

# After:
from spare_paw.core.prompt import _build_system_prompt
```

### 8b: Replace send calls with `backend.send_text()`

```python
# Before:
bot = app_state.application.bot
await _send_chunked(bot, owner_id, result)

# After:
if app_state.backend:
    await app_state.backend.send_text(result)
```

Delete `_send_chunked()` — chunking is now the backend's responsibility.

**Behavioral note:** The current `_send_chunked` sends plain text (no HTML conversion), while `_send_response` in handler.py converts markdown to HTML. After the refactor, `TelegramBackend.send_text()` always converts markdown→HTML. This changes cron result rendering from plain text to formatted HTML. This is actually an **improvement** (cron results with code blocks, bold, etc. will render properly), but it's a behavioral change to be aware of during testing.

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

## Step 9: Refactor `tools/custom_tools.py` — remove Telegram import

**Modified file:** `src/spare_paw/tools/custom_tools.py`

Currently `_handle_tool_create()` imports `InlineKeyboardButton` and `InlineKeyboardMarkup` from telegram at runtime (line 317) to send the approval notification with buttons. This is a **Telegram import outside `bot/`** that violates the isolation invariant.

### Fix:

Replace the direct Telegram call with `backend.send_notification()`:

```python
# Before:
from telegram import InlineKeyboardButton, InlineKeyboardMarkup
keyboard = InlineKeyboardMarkup([[
    InlineKeyboardButton("Approve", callback_data=f"approve:{name}"),
    InlineKeyboardButton("Reject", callback_data=f"reject:{name}"),
]])
await bot.send_message(chat_id=owner_id, text=details, reply_markup=keyboard)

# After:
if app_state.backend:
    await app_state.backend.send_notification(
        text=details,
        actions=[
            {"label": "Approve", "callback_data": f"approve:{name}"},
            {"label": "Reject", "callback_data": f"reject:{name}"},
        ],
    )
```

This requires `_handle_tool_create()` to receive `app_state` (or just `backend`) instead of `bot` + `owner_id`.

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

## Step 11: Update `subagent.py` queue reference

**Modified file:** `src/spare_paw/tools/subagent.py`

The queue moves from `bot.handler._message_queue` to `core.engine._message_queue`. The subagent module's `_message_queue` reference is set at startup by `start_queue_processor()`, so the actual change is in `start_queue_processor()` (now in `core/engine.py`), which sets `subagent._message_queue = _message_queue`. The import path in subagent.py doesn't change — the reference is injected, not imported.

However, if `start_queue_processor` currently lives in `handler.py` and imports `subagent`, then after the move to `core/engine.py` the import `from spare_paw.tools.subagent import _message_queue` (or the assignment `subagent._message_queue = ...`) still works — but verify the circular import chain.

---

## Step 12: Inline keyboard callbacks (Telegram-only)

The `_handle_callback` function for tool approval buttons stays in `bot/handler.py` — it's a Telegram-specific UI feature. The webhook backend can expose tool approval via `POST /command/approve` instead.

**`app_state` access:** `_handle_callback` currently accesses `app_state` via `context.bot_data["app_state"]`. This still works because `TelegramBackend.set_app_state()` stores `app_state` on `self._application.bot_data["app_state"]` — the wiring is unchanged from the handler's perspective.

---

## File change summary

| File | Action |
|------|--------|
| `src/spare_paw/backend.py` | **NEW** — `MessageBackend` protocol (with `send_notification`) + `IncomingMessage` dataclass |
| `src/spare_paw/core/__init__.py` | **NEW** — empty |
| `src/spare_paw/core/prompt.py` | **NEW** — `_build_system_prompt` (leaf module: only imports config + tools/memory, no engine/router/tools imports) |
| `src/spare_paw/core/engine.py` | **NEW** — platform-agnostic message processor + queue + `_split_text` (extracted from handler.py) |
| `src/spare_paw/core/voice.py` | **MOVE** from `bot/voice.py` — accepts `bytes`, no Telegram import |
| `src/spare_paw/core/commands.py` | **NEW** — platform-agnostic command logic (extracted from bot/commands.py) |
| `src/spare_paw/webhook/__init__.py` | **NEW** — empty |
| `src/spare_paw/webhook/backend.py` | **NEW** — `WebhookBackend` implementation |
| `src/spare_paw/webhook/server.py` | **NEW** — aiohttp HTTP server |
| `src/spare_paw/bot/backend.py` | **NEW** — `TelegramBackend` (owns `_md_to_html`, `_convert_tables`, chunking, `send_file` media dispatch, `send_notification` with inline keyboard, `set_my_commands`, handler registration) |
| `src/spare_paw/bot/handler.py` | **MODIFY** — thin Telegram→IncomingMessage adapter, delegates to core/engine. Keeps `_extract_cron_context`, `_download_voice`, `_download_photo`, `_handle_callback` |
| `src/spare_paw/bot/commands.py` | **MODIFY** — thin wrapper, delegates to core/commands.py |
| `src/spare_paw/bot/voice.py` | **DELETE** — moved to core/voice.py |
| `src/spare_paw/cron/executor.py` | **MODIFY** — use `backend.send_text()` instead of `bot.send_message()`, update `_build_system_prompt` import to `core.prompt`, delete `_send_chunked()` |
| `src/spare_paw/gateway.py` | **MODIFY** — `AppState.backend` replaces `.application`, mode selection, tool rewiring, remove `set_my_commands`/handler setup/media dispatch/`BotCommand` import/`Application` import |
| `src/spare_paw/tools/custom_tools.py` | **MODIFY** — replace `InlineKeyboardButton/Markup` import with `backend.send_notification()` call |
| `src/spare_paw/__main__.py` | **MODIFY** — add `webhook` command |
| `src/spare_paw/tools/subagent.py` | **MINOR CHANGE** — queue reference injection path changes (from handler.py to core/engine.py), but subagent.py code itself may not change if injection is done the same way |

### Telegram isolation verification

After this refactor, grep for `telegram` imports across the codebase:

- `bot/backend.py` — yes (implements TelegramBackend)
- `bot/handler.py` — yes (converts Update → IncomingMessage)
- `bot/commands.py` — yes (thin wrapper for command parsing)
- **Everything else** — zero Telegram imports

Deleting the `bot/` directory should leave a fully functional core that works with any `MessageBackend`.

---

## Implementation order (TDD)

Every step follows **Red → Green → Refactor**: write failing tests first, then implement until they pass.

### Phase 1: Foundation (no behavior change)

#### Step 1: `backend.py` — protocol + dataclass

**Tests first** (`tests/test_backend.py`):
- `IncomingMessage` dataclass: default values, field population, `voice_bytes`/`image_bytes` accept raw bytes
- `MessageBackend` is a `runtime_checkable` Protocol: a conforming class passes `isinstance()` check
- A minimal stub backend (all methods no-op) satisfies the protocol

**Then implement:** `src/spare_paw/backend.py`

#### Step 2: `core/voice.py` — accept bytes, no Telegram import

**Tests first** (`tests/test_voice.py`):
- `transcribe(voice_bytes, config)` sends bytes to Groq Whisper (mock aiohttp)
- Returns transcription text on 200
- Raises `VoiceTranscriptionError` on missing API key
- Raises `VoiceTranscriptionError` on non-200 response
- Raises `VoiceTranscriptionError` on empty transcription
- No `telegram` import in `core/voice.py` (static check)

**Then implement:** Move `bot/voice.py` → `core/voice.py`, change signature from `TelegramFile` to `bytes`

#### Step 3: `core/prompt.py` + `core/engine.py` — platform-agnostic core

This is the biggest step. Break it into sub-steps, each with its own test cycle.

**3a: `core/prompt.py` — `_build_system_prompt` (leaf module)**

Tests first (`tests/test_prompt.py`):
- `_build_system_prompt(config)` returns base prompt + prompt file contents + memories
- `_build_system_prompt` injects `{current_time}` replacement
- No `telegram` import in `core/prompt.py`
- No import from `core/engine.py`, `router/`, or `tools/subagent.py` (leaf module invariant)

Then implement: extract `_build_system_prompt` from `handler.py` into `core/prompt.py`. **Do NOT update `cron/executor.py` or `subagent.py` imports yet** — keep `handler.py` re-exporting via `from spare_paw.core.prompt import _build_system_prompt` so existing callers don't break.

**3b: `_split_text` (pure function in engine.py)**

Tests first (`tests/test_engine.py`):
- `_split_text("short", 100)` → `["short"]`
- `_split_text(long_text, 50)` → multiple chunks, each ≤ 50 chars
- `_split_text` prefers splitting at newlines
- `_split_text` hard-cuts when no newline found
- No `telegram` import in `core/engine.py`

Then implement: extract `_split_text` from `handler.py` into `core/engine.py`. Keep `handler.py` re-exporting it so `test_telegram_format.py` doesn't break.

**IMPORTANT: `_md_to_html` and `_convert_tables` stay in `handler.py` until Step 5.** Do NOT move formatting functions in this step — `test_telegram_format.py` imports them from `handler.py` and must keep passing.

**3c: `process_message(app_state, msg, backend)` — core message loop**

Tests first (`tests/test_engine.py`):
- `process_message` with text-only `IncomingMessage`: calls `context.ingest`, assembles context, runs tool loop, calls `backend.send_text()` with the response
- `process_message` with `voice_bytes`: calls `core.voice.transcribe`, then proceeds as text
- `process_message` with `image_bytes`: base64 encodes, builds multimodal content array
- `process_message` with `cron_context`: injects cron context message into assembled messages
- `process_message` calls `compact_with_retry` in background
- `process_message` ingests both user and assistant messages
- All tests use mock `MessageBackend` — verify `send_text()` called with markdown (not HTML)

Then implement: extract message processing logic from `_handle_message` into `process_message`.

**3d: `process_agent_callback(app_state, synthetic_text, backend)`**

Tests first (`tests/test_engine.py`):
- Ingests augmented text, assembles context, runs tool loop
- Calls `backend.send_text()` (not `bot.send_message`)
- On failure, calls `backend.send_text()` with error message

Then implement: extract from `_handle_agent_callback`.

**3e+6 (ATOMIC): Queue processor + handler adapter**

⚠️ **These two must land in a single commit.** The queue format changes from `(Update, context)` to `IncomingMessage`, and the handler must be updated to produce `IncomingMessage` in the same commit. Otherwise the handler pushes `(Update, context)` but the engine expects `IncomingMessage` — runtime crash.

Tests first (`tests/test_engine.py` + `tests/test_handler.py`):
- `enqueue(msg)` puts `IncomingMessage` on queue
- `enqueue(("agent_callback", text))` works for agent callbacks
- Queue processor calls `process_message` for `IncomingMessage` items
- Queue processor calls `process_agent_callback` for agent callback items
- `start_queue_processor` injects queue reference into `subagent._message_queue`
- `_queue_message` builds `IncomingMessage` from mock `Update` and calls `engine.enqueue`
- `_download_voice` returns raw bytes from `voice.get_file().download_as_bytearray()`
- `_download_photo` returns raw bytes from `photo[-1].get_file().download_as_bytearray()`
- `_extract_cron_context` returns text when replying to bot message, `None` otherwise
- `_handle_callback` for approve/reject still works via `context.bot_data["app_state"]`

Then implement: move queue logic from `handler.py` to `engine.py` AND slim down handler.py to produce `IncomingMessage` objects — all in one commit.

#### Step 4: `core/commands.py` — platform-agnostic command functions

**Tests first** (`tests/test_commands.py`):
- `cmd_status(app_state)` returns string with uptime, memory, DB size
- `cmd_forget(app_state)` creates new conversation, returns confirmation string
- `cmd_search(app_state, "query")` returns formatted search results string
- `cmd_model(app_state, "new-model")` sets override, returns confirmation
- `cmd_config_show(app_state)` returns model config + overrides
- `cmd_cron_list(app_state)` returns formatted cron list
- All functions return `str` — no `Update`, no `reply_text()`
- No `telegram` import in `core/commands.py`

**Then implement:** Extract business logic from `bot/commands.py` into pure functions.

### Phase 2: Backends (behavior-preserving refactor)

#### Step 5: `bot/backend.py` — TelegramBackend

**Tests first** (`tests/test_telegram_backend.py`):
- `TelegramBackend` satisfies `isinstance(backend, MessageBackend)` check
- `send_text("**bold**")` calls `bot.send_message` with HTML `<b>bold</b>` and `parse_mode=HTML`
- `send_text` with long text chunks at 4096 chars
- `send_text` falls back to plain text when HTML send raises
- `send_file("photo.jpg")` calls `bot.send_photo`
- `send_file("video.mp4")` calls `bot.send_video`
- `send_file("audio.mp3")` calls `bot.send_audio`
- `send_file("doc.pdf")` calls `bot.send_document`
- `send_typing()` calls `bot.send_chat_action(TYPING)`
- `send_notification` with actions creates `InlineKeyboardMarkup` with correct buttons
- `send_notification` without actions sends plain text
- `_md_to_html` tests: migrate all cases from `test_telegram_format.py` (they should test `TelegramBackend._md_to_html` now)
- `_convert_tables` tests: migrate from `test_telegram_format.py`
- `start()` calls `application.initialize`, `setup_handlers`, `set_my_commands`, `start_polling`
- `stop()` calls `updater.stop`, `application.stop`, `application.shutdown`
- `set_app_state()` stores on `application.bot_data["app_state"]`

**Then implement:** Create `TelegramBackend` class, move `_md_to_html`, `_convert_tables` from handler.py. Delete `test_telegram_format.py` (tests migrated to `test_telegram_backend.py`). NOW `_md_to_html` leaves handler.py.

#### Step 6: `bot/commands.py` — slim down to wrapper

**Tests first** (update `tests/test_commands.py`):
- Each `/command` handler calls the corresponding `core.commands.cmd_*` function
- Each handler sends the returned string via `update.message.reply_text()`
- Auth check (`_is_owner`) still happens before delegation

**Then implement:** Replace inline business logic with calls to `core.commands.*`.

### Phase 3: Wiring (behavior-preserving)

⚠️ **Order matters here.** Steps 7-8 rewire consumers that reference `app_state.application`. Step 9 removes `app_state.application`. If Step 9 lands before 7-8, those consumers crash at runtime.

#### Step 7: `cron/executor.py` — use backend

**Tests first** (update `tests/test_cron.py`):
- Executor calls `backend.send_text(result)` instead of `bot.send_message`
- `_build_system_prompt` imported from `core.prompt`, not `bot.handler`
- No `_send_chunked` function exists

**Then implement:** Update imports, replace send calls. This step removes the last `app_state.application` reference from executor.py.

#### Step 8: `tools/custom_tools.py` — remove Telegram import

**Tests first** (update `tests/test_tools.py`):
- `_handle_tool_create` calls `backend.send_notification()` with actions list
- No `telegram` import in `custom_tools.py`

**Then implement:** Replace `InlineKeyboardButton/Markup` with `backend.send_notification()`. This step removes the last `app_state.application` reference from custom_tools.py.

#### Step 9: `gateway.py` — backend selection + tool rewiring

**Tests first** (`tests/test_gateway.py`):
- `AppState` has `backend` field
- `AppState.application` property exists as deprecated shim (returns `backend._application` for `TelegramBackend`, `None` otherwise)
- `send_message` tool calls `app_state.backend.send_text()`
- `send_file` tool calls `app_state.backend.send_file()`
- No `telegram` import in gateway.py
- Mode selection: `mode="telegram"` creates `TelegramBackend`, `mode="webhook"` creates `WebhookBackend`

**Then implement:** Refactor gateway.py. Add `backend` field to `AppState`. Add deprecated `application` property as safety net:

```python
@property
def application(self):
    """Deprecated — use backend. Kept temporarily to catch missed references."""
    import warnings
    warnings.warn("app_state.application is deprecated, use app_state.backend", DeprecationWarning, stacklevel=2)
    if hasattr(self.backend, '_application'):
        return self.backend._application
    return None
```

This ensures any missed `app_state.application` reference logs a warning instead of crashing, giving us time to find and fix it. Remove the shim once all tests pass and the codebase is clean.

### Phase 4: New capability

#### Step 10: `webhook/` — HTTP backend

**Tests first** (`tests/test_webhook.py`):
- `WebhookBackend` satisfies `isinstance(backend, MessageBackend)` check
- `send_text("hello")` pushes to response queue
- `send_file("path")` pushes file event to queue
- `send_typing()` pushes typing event
- `send_notification` pushes notification with actions
- `POST /message {"text": "hello"}` creates `IncomingMessage` and enqueues
- `POST /message {"voice_b64": "..."}` decodes base64 into `IncomingMessage.voice_bytes`
- `POST /command/status` returns status string
- `GET /events` returns SSE stream with queued responses
- `GET /health` returns 200 with uptime

**Then implement:** `webhook/backend.py` + `webhook/server.py`.

#### Step 11: `__main__.py` — add webhook command

**Tests first** (`tests/test_main.py`):
- `python -m spare_paw webhook` calls `run(mode="webhook")`
- Default mode is still `gateway` (telegram)

**Then implement:** Update `__main__.py`.

### Phase 5: Isolation verification

#### Step 12: Import isolation check

**Test** (`tests/test_isolation.py`):
- Scan all `.py` files outside `bot/` for `import telegram` or `from telegram` — assert zero matches
- Scan `core/` for any platform-specific imports — assert zero
- This test runs in CI as a gate

#### Step 13: Remove deprecated `app_state.application` shim

Once all tests pass and production is stable, remove the deprecated property from Step 9.

---

### Summary: test file mapping

| Implementation | Test file | Key assertions |
|---|---|---|
| `backend.py` | `test_backend.py` | Protocol compliance, dataclass defaults |
| `core/voice.py` | `test_voice.py` | Bytes input, mock Groq, error cases |
| `core/prompt.py` | `test_prompt.py` | Leaf module invariant, prompt assembly |
| `core/engine.py` | `test_engine.py` | process_message, process_agent_callback, queue, _split_text |
| `core/commands.py` | `test_commands.py` | Pure functions returning strings |
| `bot/backend.py` | `test_telegram_backend.py` | md→HTML, chunking, media dispatch, inline keyboard |
| `bot/handler.py` | `test_handler.py` | Update→IncomingMessage adapter |
| `bot/commands.py` | `test_commands.py` | Delegation to core.commands |
| `cron/executor.py` | `test_cron.py` | backend.send_text, import from core.prompt |
| `custom_tools.py` | `test_tools.py` | backend.send_notification |
| `gateway.py` | `test_gateway.py` | Backend selection, tool rewiring, deprecated shim |
| `webhook/` | `test_webhook.py` | HTTP endpoints, SSE, queue |
| `__main__.py` | `test_main.py` | CLI mode selection |
| isolation | `test_isolation.py` | No telegram imports outside bot/ |

Steps 1-9 don't change any behavior — Telegram mode works exactly as before. Step 10 adds the new capability. This ordering minimizes risk.

---

## Known behavioral changes

1. **Cron results will render as HTML** — currently sent as plain text via `_send_chunked`. After the refactor, `TelegramBackend.send_text()` converts markdown→HTML. This is an improvement but worth testing.
2. **Agent callback responses unified** — currently `_handle_agent_callback` has a separate HTML send path that bypasses `_send_response`. After the refactor, both go through `backend.send_text()`.
3. **Tool approval notifications** — currently use direct Telegram inline keyboard API. After the refactor, they go through `backend.send_notification()` which `TelegramBackend` renders as inline keyboard buttons.

---

## Risks and mitigations

### HIGH

1. **Circular import: `engine.py` ↔ `subagent.py` via `_build_system_prompt`**

   The chain: `engine.py` → `tools/` → `subagent.py` → `_build_system_prompt`. If `_build_system_prompt` lives in `engine.py`, subagent importing it creates `engine → tools → engine`. Currently safe only because subagent uses a lazy import — but one careless top-level import later and it breaks.

   **Mitigation:** `_build_system_prompt` lives in `core/prompt.py` (leaf module), NOT in `engine.py`. `core/prompt.py` imports only from `config.py` and `tools/memory.py`. No cycle possible regardless of import style.

### MEDIUM

2. **`app_state.application` removal before consumers are rewired**

   `cron/executor.py:105` and `custom_tools.py:287` reach for `app_state.application.bot`. If Step 9 (gateway refactor) removes `app_state.application` before Steps 7-8 rewire these consumers → runtime crash. Crons silently stop delivering results; tool approval notifications disappear.

   **Mitigation:** Strict ordering — Steps 7 (executor) and 8 (custom_tools) MUST land before Step 9 (gateway). Step 9 adds a deprecated `application` property shim that logs `DeprecationWarning` instead of crashing, as safety net for any missed references.

3. **Queue format change breaks handler mid-refactor**

   The queue currently accepts `(Update, context)` tuples. After refactor, it accepts `IncomingMessage`. If engine's queue processor is updated (Step 3e) but handler still pushes `(Update, context)` (Step 6 not done yet) → `process_message` receives a Telegram `Update` object instead of `IncomingMessage` → `AttributeError` at runtime.

   **Mitigation:** Steps 3e and 6 (handler adapter) are merged into a single atomic commit. Both sides of the queue contract change together.

4. **Test imports break during migration window**

   `test_telegram_format.py:3` imports `from spare_paw.bot.handler import _md_to_html`. If Step 3 removes `_md_to_html` from handler.py (because "extract everything"), tests break before Step 5 moves them.

   **Mitigation:** `_md_to_html` and `_convert_tables` stay in `handler.py` until Step 5. Step 3 only extracts `_build_system_prompt` (to `core/prompt.py`) and `_split_text` (to `core/engine.py`). Handler re-exports both via `from spare_paw.core.prompt import _build_system_prompt` so existing callers don't break. `_md_to_html` moves in Step 5 when tests are migrated to `test_telegram_backend.py` in the same commit.

5. **`app_state` availability in Telegram handlers post-refactor**

   After hiding `Application` inside `TelegramBackend`, handlers still need `app_state` via `context.bot_data["app_state"]`.

   **Mitigation:** `TelegramBackend.set_app_state()` stores it on `application.bot_data`, preserving the existing access pattern. Tested in Step 5.

### LOW

6. **Queue reference migration**

   `handler._message_queue` moves to `engine._message_queue`. If subagent.py imports queue directly instead of using injection, circular import risk.

   **Mitigation:** Keep existing injection pattern — `start_queue_processor` sets `subagent._message_queue = _message_queue` at runtime. No import of engine from subagent needed.

7. **`_extract_cron_context` integration gap**

   Tested in isolation (Step 3e) but no integration test verifying the full flow: handler extracts cron context → populates `IncomingMessage.cron_context` → engine injects context into assembled messages → model receives it.

   **Mitigation:** Add an integration test in Step 3e that feeds an `IncomingMessage(cron_context="some cron output")` through `process_message` and verifies the context appears in the assembled messages passed to `run_tool_loop`.

8. **No incremental rollback path**

   13 steps, each changing internal APIs. If Step 6 breaks production, reverting it without also reverting Steps 3-5 is difficult because they share import path changes.

   **Mitigation:** Each step's commit must leave the codebase in a passing-tests, deployable state. Re-exports (e.g., handler.py re-exporting `_build_system_prompt` from `core/prompt.py`) act as compatibility shims during transition. The deprecated `application` property in Step 9 serves the same purpose. If a step breaks, revert only that step's commit — re-exports ensure earlier steps remain compatible.

---

## Testing strategy

Tests are written **before** implementation at every step (see Implementation order above). The approach is:

1. **Red**: Write tests that assert the desired behavior of the new module/function. Tests fail because the code doesn't exist yet.
2. **Green**: Implement the minimum code to make tests pass.
3. **Refactor**: Clean up, ensure no regressions in existing tests.

**Key testing principles:**
- Every `core/` module is tested with mock backends — no Telegram dependency in tests
- `test_prompt.py` verifies `core/prompt.py` is a leaf module (no engine/router/tools imports) — this is the circular import prevention gate
- `test_telegram_backend.py` absorbs all existing `test_telegram_format.py` tests (which currently import from `handler.py`)
- `test_isolation.py` is a static analysis gate — scans for `telegram` imports outside `bot/`
- Existing tests must pass at every step. Any import path changes (e.g. `_split_text` moving from handler to engine) require updating test imports in the same commit.
- Integration test: start gateway in webhook mode, send a message via HTTP, verify response comes back through SSE
