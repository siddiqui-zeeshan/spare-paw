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

- **Video messages:** Register handlers for `message.video` and `message.video_note` (circle videos). Download bytes via `get_file()`, capture MIME type from `message.video.mime_type`.
- **Photo MIME detection:** Use file extension or Telegram file metadata instead of hardcoding `"image/jpeg"`.

### 4. Webhook Backend Changes

Accept an optional `video` field (base64-encoded) in the JSON payload, alongside the existing `image` field.

### 5. Vision Preprocessor Function

A standalone async function `describe_media()`:

**Inputs:**
- `router_client`: OpenRouterClient instance
- `media_bytes`: raw image or video bytes
- `media_mime`: MIME type string (e.g., `image/jpeg`, `video/mp4`)
- `user_text`: the user's caption or message text (may be None)

**Behavior:**
1. Base64-encode the media bytes into a data URL: `data:{media_mime};base64,{b64}`
2. Build a single-turn message list:
   - System message: "Describe this image/video in the context of the user's message. Focus on details relevant to what the user is asking or saying."
   - User message (multimodal): the media content + user text (or "Describe this image/video" if no text provided)
3. Call `router_client.chat()` with `models.vision` model, no tools
4. Return the description string

**Location:** New function in a suitable module (e.g., `core/vision.py` or inline in `core/engine.py`).

### 6. Main Agent Message Construction

After `describe_media()` returns:

1. Insert a system message after the last user message in the assembled message list:
   ```python
   {"role": "system", "content": "[Media analysis]: <description>"}
   ```
2. The user's original text is preserved as-is in the user message.
3. The existing base64 image injection logic (multimodal content block in `process_message()`) is **removed** — the main agent always receives text descriptions, never raw media.

### 7. Configuration

Add `models.vision` key to the config system:

- Config path: `models.vision`
- Default value: `google/gemini-3.1-flash-lite-preview`
- Follows the same resolution chain as other model keys

### 8. Always Preprocess

Media is always routed through the vision model, regardless of whether the main agent supports vision. This avoids a config flag (`main_agent_has_vision`) and ensures consistent behavior. If the main agent is later switched to a vision-capable model, no code changes are needed.

## What Changes

| Component | Change |
|-----------|--------|
| `IncomingMessage` (backend.py) | Add `video_bytes`, `video_mime` fields |
| Telegram handler (bot/handler.py) | Add video/video_note handlers, fix photo MIME detection |
| Webhook backend (webhook/backend.py) | Accept `video` base64 field |
| Vision preprocessor (new) | `describe_media()` async function |
| Engine (core/engine.py) | Call `describe_media()` before context assembly, remove inline image injection |
| Config (config.py) | Add `models.vision` default |

## What Doesn't Change

- Context assembly, tool loop, subagent system, dialogue channels
- OpenRouter client (already handles any model)
- Storage/DB (media descriptions are just text messages)
- TUI backend (no media support currently)
