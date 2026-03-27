# Vision Preprocessing for Non-Vision Main Agent

## Problem

The main agent model may not have vision capabilities, but users send photos and videos via Telegram. Currently, images are base64-encoded and injected as multimodal content into the user message — this fails silently when the main agent can't process images, and video isn't supported at all.

## Solution

Detect media (image or video) in incoming messages and preprocess it through a vision-capable model before the main agent sees it. The main agent receives a text description of the media instead of raw bytes. This always runs — even if the main agent has vision — for consistency and better tool-calling reliability.

## Design

### 1. Detection & Routing

In `process_message()`, after voice transcription but before context assembly, check if the incoming message has `image_bytes` or `video_bytes`. If present, call the vision preprocessor. If not, proceed as normal with zero overhead.

### 2. IncomingMessage Changes

Add two new fields to the `IncomingMessage` dataclass:

```python
video_bytes: bytes | None = None
video_mime: str = "video/mp4"
```

### 3. Telegram Handler Changes

- **Video messages:** Register handlers for `message.video` and `message.video_note` (circle videos). Download bytes via `get_file()`. For `message.video`, capture MIME type from `message.video.mime_type`. For `message.video_note`, hardcode `"video/mp4"` (Telegram always produces circular MP4s; `VideoNote` has no `mime_type` attribute).
- **Photo MIME:** Telegram always re-encodes photos as JPEG, so `"image/jpeg"` is correct. No change needed.

### 4. Webhook Backend Changes

Accept an optional `video` field (base64-encoded) in the JSON payload, alongside the existing `image` field. No separate `video_mime` field needed — the `IncomingMessage` default of `"video/mp4"` covers it.

### 5. Vision Preprocessor Function

A standalone async function `describe_media()` in a new module `core/vision.py`:

**Inputs:**
- `router_client`: OpenRouterClient instance
- `media_bytes`: raw image or video bytes
- `media_mime`: MIME type string (e.g., `image/jpeg`, `video/mp4`)
- `user_text`: the user's caption or message text (may be None)
**Behavior:**
1. Base64-encode the media bytes into a data URL: `data:{media_mime};base64,{b64}`
2. Build a single-turn message list:
   - System message: "Describe this image/video in the context of the user's message. Focus on details relevant to what the user is asking or saying."
   - User message (multimodal): content blocks with user text + media
3. For the media content block, use the correct type based on media kind:
   - Images: `{"type": "image_url", "image_url": {"url": data_url}}`
   - Videos: `{"type": "video_url", "video_url": {"url": data_url}}`
4. Call `router_client.chat()` with `models.vision` model, no tools
5. Return the description string

**Supported video formats:** video/mp4, video/mpeg, video/mov, video/webm

**Error handling:** If `describe_media()` raises, the exception propagates to the queue processor's existing error handler, which surfaces the error to the user. No retry or fallback logic.

### 6. Main Agent Message Construction

After `describe_media()` returns:

1. Append a user message at the end of the assembled message list (which always ends with the current user turn after `ctx.assemble()`):
   ```python
   {"role": "user", "content": "[Media analysis]: <description>"}
   ```
   Using `role: "user"` rather than `role: "system"` because mid-conversation system messages are non-standard in the OpenAI API format and may be rejected by some models.
2. The user's original text is preserved as-is in the user message.
3. The existing base64 image injection logic (multimodal content block in `process_message()`) is **removed** — the main agent always receives text descriptions, never raw media.

### 7. Configuration

Add `models.vision` key to the config system:

- Config path: `models.vision`
- Default value: `google/gemini-3.1-flash-lite-preview`
- Add `"vision"` to `MODEL_ROLES` tuple so `resolve_model()` recognizes it
- Add `"vision"` to `_build_defaults()` so the default is always present
- The fallback chain (`models.vision` → `models.main_agent` → `DEFAULT_MODEL`) must not silently fall back to a non-vision model. Since `vision` is added to defaults with an explicit vision-capable model, this is safe — but note that users who override `models.vision` to a non-vision model will get errors.

### 8. Always Preprocess

Media is always routed through the vision model, regardless of whether the main agent supports vision. This avoids a config flag (`main_agent_has_vision`) and ensures consistent behavior. If the main agent is later switched to a vision-capable model, no code changes are needed.

### 9. Text Fallback Ownership

The engine currently sets `text = msg.caption or "What do you see in this image?"` when image_bytes is present. This fallback is removed — `describe_media()` owns the fallback. If `user_text` is None or empty, `describe_media()` uses a generic prompt like "Describe this image/video in detail." The engine passes `msg.text or msg.caption` as `user_text` without substitution.

## What Changes

| Component | Change |
|-----------|--------|
| `IncomingMessage` (backend.py) | Add `video_bytes`, `video_mime` fields |
| Telegram handler (bot/handler.py) | Add video/video_note handlers |
| Webhook backend (webhook/backend.py) | Accept `video` base64 field |
| Vision preprocessor (core/vision.py) | New `describe_media()` async function |
| Engine (core/engine.py) | Call `describe_media()` before context assembly, remove inline image injection, remove text fallback for images |
| Config (config.py) | Add `"vision"` to `MODEL_ROLES`, add default in `_build_defaults()` |

## What Doesn't Change

- Context assembly, tool loop, subagent system, dialogue channels
- OpenRouter client (already handles any model)
- Storage/DB (media descriptions are just text messages)
- TUI backend (no media support currently)
- `image_bytes`, `image_mime` fields on `IncomingMessage` (still used to carry image data to the engine)
