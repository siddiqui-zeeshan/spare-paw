# Talk Mode — Voice Replies via OpenRouter TTS

## Problem

spare-paw transcribes incoming voice notes (Groq Whisper, `core/voice.py`) but can only reply as text. Users who message by voice get a text reply back, and there is no way to have the assistant speak its responses when the user prefers audio output (driving, hands busy, accessibility).

## Solution

Add a **talk mode** that renders the final assistant response as a Telegram voice note (ogg/opus) using OpenRouter's `openai/gpt-4o-mini-tts-2025-12-15` text-to-speech model. Two activation paths:

1. **Mirror (default):** voice in → voice out; text in → text out.
2. **`/talk on` (override):** voice out for every reply in this conversation, regardless of how the user messaged.

When voice mode is active for a turn, the system prompt gains a short hint instructing the model to reply in a style suited for speech (no markdown, no code blocks, natural conversational tone).

State is per-conversation: a new `/new` resets `talk_mode` and the per-conversation voice override.

## Design

### 1. Components

Two new modules, changes to five existing files.

| File | Status | Responsibility |
|---|---|---|
| `router/tts.py` | new | `async synthesize(text: str, voice: str, config: dict) -> bytes` — single OpenRouter HTTPS POST to the TTS endpoint (exact path to confirm during implementation from the model's OpenRouter page; likely `/api/v1/audio/speech` following OpenAI's normalized convention), returns raw audio bytes. No ffmpeg. Raises `TTSError` on HTTP/timeout/decode failure after internal retries. Exposes `KNOWN_VOICES` tuple. |
| `core/voice_out.py` | new | `async render_voice_note(text, voice, config) -> bytes` — calls `tts.synthesize`, pipes bytes through `ffmpeg -i - -c:a libopus -b:a 32k -f ogg -` (asyncio subprocess), returns opus-in-ogg bytes ready for Telegram. Raises `VoiceRenderError` (wraps `TTSError`, ffmpeg failures, and missing-ffmpeg). |
| `core/prompt.py` | changed | `build_system_prompt(config, voice_mode: bool = False)` — when `voice_mode=True`, appends the voice-mode hint block. |
| `core/engine.py` | changed | After `run_tool_loop` returns, decide delivery based on `conversation.metadata.talk_mode` + `incoming.kind`. Also passes `voice_mode` into prompt builder. |
| `core/commands.py` | changed | New handlers for `/talk on|off`, `/talk`, `/voice <name>`, `/voice`, `/voice list`. |
| `context.py` | changed | `context.py` currently has message-level metadata only. Add two small helpers: `get_conversation_meta(conversation_id) -> dict` (reads the `conversations.metadata` JSON column) and `set_conversation_meta(conversation_id, key, value)` (reads, merges one key, writes back in a single `UPDATE`). |
| `bot/backend.py` (MessageBackend Protocol) | changed | Add `async send_voice(ogg_bytes: bytes) -> None`. |
| `bot/backend.py` (TelegramBackend) | changed | Implement `send_voice` via `self.bot.send_voice(chat_id=..., voice=BufferedInputFile(ogg_bytes, "voice.ogg"))`. |
| `webhook/` | changed | `send_voice` stub raises `NotImplementedError` with a clear message. Voice delivery over webhook is out of scope for v1. |

### 2. Data flow (per turn)

```
bot/handler._queue_message
  └─> IncomingMessage{text, voice_bytes, kind="voice"|"text", …}
      │
engine._process_queue → engine.process_message(incoming)
  ├─ if voice: text = voice.transcribe(voice_bytes, config)   # Groq Whisper, unchanged
  ├─ convo = ctx.get_or_create_conversation(chat_id)
  ├─ ctx.ingest(user, text)
  ├─ if text is a command: core.commands.handle(…) → early return
  ├─ voice_mode = convo.meta.get("talk_mode", False)
  ├─ system_prompt = prompt.build_system_prompt(config, voice_mode=voice_mode)
  ├─ messages = ctx.assemble(convo, system_prompt, budget)
  ├─ reply_text = await tool_loop.run_tool_loop(messages, …)   # unchanged
  ├─ ctx.ingest(assistant, reply_text)
  │
  ├─ should_voice = voice_mode or (incoming.kind == "voice")
  ├─ if should_voice and len(reply_text) ≤ tts_max_chars:
  │     try:
  │         voice_name = convo.meta.get("voice") or config["voice"]["tts_voice"]
  │         ogg = await voice_out.render_voice_note(reply_text, voice_name, config)
  │         await backend.send_voice(ogg)
  │     except VoiceRenderError as e:
  │         log.warning("TTS failed, falling back to text: %s", e)
  │         if not convo.meta.get("voice_error_notified"):
  │             await backend.send_text("🔇 Voice generation failed. Falling back to text.")
  │             ctx.set_conversation_meta(convo, "voice_error_notified", True)
  │         await backend.send_text(reply_text)
  ├─ elif should_voice and len(reply_text) > tts_max_chars:
  │     await backend.send_text("⚠️ reply too long for voice\n\n" + reply_text)
  ├─ elif reply_text:
  │     await backend.send_text(reply_text)
  │ # empty reply_text → send nothing (existing behavior)
  │
  └─ trigger background LCM compaction   # unchanged
```

Single branching point for "voice or text", single injection point for the voice-mode prompt hint. No scatter across layers.

Edge cases:
- Voice in, `talk_mode=False` → mirror → voice out.
- Voice in, `talk_mode=False`, reply > cap → text with warning banner.
- Empty `reply_text` → no message sent (unchanged).
- `voice_error_notified` is a per-conversation sticky bit that suppresses repeat "voice failed" notifications until the conversation is reset. Cleared on first successful `send_voice` in the same conversation.

### 3. Conversation state

Two new keys under `conversations.metadata` (existing JSON column):

| Key | Type | Default | Description |
|---|---|---|---|
| `talk_mode` | `bool` | `False` | True → voice out for every reply in this conversation. |
| `voice` | `str \| null` | `null` | Per-conversation voice override. When `null`, `config.voice.tts_voice` is used. |
| `voice_error_notified` | `bool` | `False` | Internal: suppresses repeat "voice failed" user notifications. Cleared on first successful `send_voice`. |

`/new` creates a fresh row → all three reset to defaults.

### 4. Commands

All handled in `core/commands.py` alongside `/new`, `/reset`.

| Command | Effect | User reply (always text) |
|---|---|---|
| `/talk on` | `metadata.talk_mode = True` (refused if `config.voice.tts_enabled = False`) | `🎙️ Talk mode on. I'll reply by voice.` |
| `/talk off` | `metadata.talk_mode = False` | `Talk mode off. I'll mirror — voice in, voice out.` |
| `/talk` | Read-only | `Talk mode is on.` / `Talk mode is off (mirror mode).` |
| `/voice <name>` | Validate `name` against `KNOWN_VOICES`; set `metadata.voice = name` | `Voice set to nova.` or `Unknown voice "xyz". Try: alloy, ash, ballad, coral, echo, fable, nova, onyx, sage, shimmer.` |
| `/voice` | Read-only | `Current voice: nova (set via /voice).` or `Current voice: alloy (from config).` |
| `/voice list` | Read-only | Comma-separated list of `KNOWN_VOICES`. |

Command confirmations are always text. The user just toggled a setting; instant text feedback beats TTS latency and cost.

### 5. System-prompt hint

Appended by `prompt.build_system_prompt` when `voice_mode=True`:

> You are replying by voice. Keep responses conversational and natural. No markdown, no code blocks, no bullet lists, no URLs read aloud. Spell numbers in words when natural. Avoid section headings.

Rebuilt per turn (existing behavior of `build_system_prompt`), so toggling `/talk` takes effect on the very next response.

### 6. Configuration

Under `voice:` in `~/.spare-paw/config.yaml`, merged with `config._build_defaults`:

```yaml
voice:
  # EXISTING (STT) — unchanged
  stt_provider: groq
  groq_api_key: ...

  # NEW (TTS)
  tts_enabled: true
  tts_voice: nova                              # default feminine voice
  tts_model: openai/gpt-4o-mini-tts-2025-12-15
  tts_max_chars: 2000
  tts_timeout_seconds: 30
  ffmpeg_path: ffmpeg
```

`OPENROUTER_API_KEY` is already present for the chat model; TTS reuses it.

**ffmpeg missing**: at module load, `voice_out.py` records `shutil.which(config.voice.ffmpeg_path)`. If absent and `tts_enabled=True`, log a `WARNING` at startup. First `/talk on` attempt refuses with `Talk mode needs ffmpeg but it's not installed. Install ffmpeg or set voice.tts_enabled=false.` Bot startup is not blocked.

**Setup wizard** (`setup_wizard.py`): add a prompt — *"Enable voice replies? (y/n, needs ffmpeg)"* — that flips `tts_enabled`. Doesn't block setup.

### 7. Error handling

Single source of truth is the voice branch in `engine.process_message` (section 2).

| Failure | Behavior |
|---|---|
| `tts_enabled=False` | Skip TTS entirely, send text. `/talk on` refuses with clear message. |
| ffmpeg missing at turn time | `VoiceRenderError("ffmpeg not found")` → text fallback + one-time user notification per conversation. |
| OpenRouter TTS HTTP error (4xx/5xx, timeout, net) | `TTSError` raised after internal retries in `router/tts.py` (same 3x exponential retry pattern as `router/openrouter.py`). Wrapped as `VoiceRenderError`. Text fallback. |
| ffmpeg non-zero exit | `VoiceRenderError("ffmpeg failed: <stderr tail>")`. Text fallback. |
| Telegram `send_voice` rejects the blob | Caught in `TelegramBackend.send_voice`, raised as `VoiceRenderError("telegram rejected: …")`. Text fallback. |
| `reply_text > tts_max_chars` | Short-circuit before TTS call; send text with `⚠️ reply too long for voice\n\n` banner. |
| Empty `reply_text` | Existing behavior — send nothing. |
| Unknown voice in `/voice xyz` | Command rejected; conversation state unchanged. |
| `/talk on` while `tts_enabled=False` | Command rejected with clear message. |

**No circuit breaker.** If TTS is globally broken, every turn falls back to text. Self-recovering when TTS is restored. The `voice_error_notified` sticky bit prevents user-facing notification spam within a single conversation.

**Logging**: `WARNING` for every `VoiceRenderError` with the reason. `INFO` for successful TTS with `{voice, char_count, audio_bytes, duration_ms, model}`. API keys never logged (global CLAUDE.md rule).

### 8. Testing

Following the repo's pytest + pytest-asyncio (strict) setup. Each test below is written **before** its corresponding implementation, failing first, then made to pass.

| Test file | Verifies |
|---|---|
| `tests/test_tts.py` | `router.tts.synthesize` — success path, 429 retry, timeout → `TTSError`, unknown voice → `ValueError` before HTTP call, API key not present in any raised error string. |
| `tests/test_voice_out.py` | `core.voice_out.render_voice_note` — patches `tts.synthesize` + `asyncio.create_subprocess_exec`; asserts opus bytes returned on success, `VoiceRenderError` on ffmpeg non-zero, `VoiceRenderError` when `shutil.which` returns `None`. |
| `tests/test_prompt_voice_mode.py` | `build_system_prompt(voice_mode=True)` appends the hint exactly once; `voice_mode=False` does not include it. |
| `tests/test_commands_talk.py` | All `/talk` and `/voice` subcommands — state transitions on `conversation.metadata`, error replies on invalid input, refusal when `tts_enabled=False`. In-memory SQLite fixture from `conftest.py`. |
| `tests/test_engine_voice_delivery.py` | Full branch coverage of `engine.process_message` delivery decision: <br>• voice-in + `talk_mode=False` → `send_voice` <br>• text-in + `talk_mode=True` → `send_voice` <br>• text-in + `talk_mode=False` → `send_text` <br>• reply > cap + `talk_mode=True` → `send_text` with banner <br>• `render_voice_note` raises → `send_text` + warning notification sent once per conversation <br>• `tts_enabled=False` + `talk_mode=True` → `send_text` (no TTS attempted) |
| `tests/test_context_conversation_meta.py` | `get_conversation_meta` / `set_conversation_meta` round-trip; `set_conversation_meta` merges (doesn't overwrite sibling keys); returns `{}` when the column is NULL or invalid JSON. |
| `tests/test_backend_send_voice.py` | `TelegramBackend.send_voice` — mocks `self.bot.send_voice`; asserts correct `BufferedInputFile` construction, correct `chat_id`, correct filename. |

**No integration tests in v1.** Real OpenRouter + real ffmpeg = flaky + costs money. Can be added later behind `@pytest.mark.slow` and a live API-key env var.

**Lint/typecheck:** repo has neither configured; not adding them as part of this feature.

### 9. Migration & rollout

- New feature, no data migration. Existing conversations have no `talk_mode` key → treated as `False`.
- Default `tts_enabled=True` in `_build_defaults`, so the feature is on for anyone with ffmpeg installed. Users without ffmpeg get a startup `WARNING` log but nothing breaks.
- Setup wizard update is additive; existing `~/.spare-paw/config.yaml` files without the new keys continue to work (defaults fill in).
- No changes required to `IDENTITY.md`, `USER.md`, or `SYSTEM.md`.

### 10. Out of scope (v1)

Deliberately excluded; revisit if/when needed:

- Voice delivery over the webhook backend (`send_voice` stubbed to `NotImplementedError`).
- Streaming / chunked TTS (one voice note per reply).
- Narrating tool progress or error events as voice.
- Caching TTS output (every reply is unique; cache hit rate would be ~0).
- `IDENTITY.md` voice declaration (`Voice: nova`) — kept as pure markdown.
- Multiple voices per conversation ("have the bot mimic two characters").
- Circuit breaker for repeated TTS failures.
- Live integration tests.
