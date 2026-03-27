# Vision Preprocessing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Preprocess images and videos through a vision-capable model before the main agent sees them, so the main agent receives text descriptions instead of raw media bytes.

**Architecture:** A new `describe_media()` function in `core/vision.py` calls the vision model (configurable via `models.vision`). The engine calls it after voice transcription but before context assembly, then appends the description as a user message. Telegram handler gains video/video_note support.

**Tech Stack:** Python, asyncio, python-telegram-bot, OpenRouter API, pytest

---

### Task 1: Add `vision` to config defaults and MODEL_ROLES

**Files:**
- Modify: `src/spare_paw/config.py:21` (MODEL_ROLES), `src/spare_paw/config.py:28-36` (_build_defaults models dict)
- Test: `tests/test_config.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_config.py` in `TestResolveModel`:

```python
def test_vision_role_in_model_roles(self):
    assert "vision" in MODEL_ROLES

def test_vision_has_default(self):
    cfg = Config()
    assert resolve_model(cfg, "vision") == "google/gemini-3.1-flash-lite-preview"

def test_all_roles_defined(self):
    expected = {"main_agent", "coder", "planner", "cron", "researcher", "analyst", "summary", "vision"}
    assert set(MODEL_ROLES) == expected
```

Note: the existing `test_all_roles_defined` must be updated (not duplicated) — replace its `expected` set.

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/zeeshans/myprojects/spare-paw && .venv/bin/python -m pytest tests/test_config.py::TestResolveModel -v`
Expected: FAIL — `"vision" not in MODEL_ROLES`, vision default not found

- [ ] **Step 3: Write minimal implementation**

In `src/spare_paw/config.py`:

Line 21 — add `"vision"` to MODEL_ROLES:
```python
MODEL_ROLES = ("main_agent", "coder", "planner", "cron", "researcher", "analyst", "summary", "vision")
```

Line 35 — add vision default inside `_build_defaults()` models dict, after the `"summary"` entry:
```python
"vision": "google/gemini-3.1-flash-lite-preview",
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/zeeshans/myprojects/spare-paw && .venv/bin/python -m pytest tests/test_config.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/spare_paw/config.py tests/test_config.py
git commit -m "feat: add vision model role to config defaults"
```

---

### Task 2: Add `video_bytes` and `video_mime` to IncomingMessage

**Files:**
- Modify: `src/spare_paw/backend.py:14-31`
- Test: `tests/test_backend.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_backend.py`:

```python
class TestIncomingMessageVideo:
    def test_video_fields_default(self):
        msg = IncomingMessage()
        assert msg.video_bytes is None
        assert msg.video_mime == "video/mp4"

    def test_video_fields_set(self):
        msg = IncomingMessage(video_bytes=b"\x00\x01", video_mime="video/webm")
        assert msg.video_bytes == b"\x00\x01"
        assert msg.video_mime == "video/webm"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/zeeshans/myprojects/spare-paw && .venv/bin/python -m pytest tests/test_backend.py::TestIncomingMessageVideo -v`
Expected: FAIL — `unexpected keyword argument 'video_bytes'`

- [ ] **Step 3: Write minimal implementation**

In `src/spare_paw/backend.py`, add after line 25 (`image_mime` field):

```python
video_bytes: bytes | None = None
video_mime: str = "video/mp4"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/zeeshans/myprojects/spare-paw && .venv/bin/python -m pytest tests/test_backend.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/spare_paw/backend.py tests/test_backend.py
git commit -m "feat: add video_bytes and video_mime to IncomingMessage"
```

---

### Task 3: Create `describe_media()` in `core/vision.py`

**Files:**
- Create: `src/spare_paw/core/vision.py`
- Create: `tests/test_vision.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_vision.py`:

```python
"""Tests for core/vision.py — media description preprocessing."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from spare_paw.core.vision import describe_media


class TestDescribeMedia:
    @pytest.mark.asyncio
    async def test_image_returns_description(self):
        """Image bytes + user text → calls vision model, returns description."""
        client = MagicMock()
        client.chat = AsyncMock(return_value={
            "choices": [{"message": {"content": "A cat on a keyboard"}}],
        })
        result = await describe_media(
            client=client,
            media_bytes=b"\xff\xd8\xff\xe0",
            media_mime="image/jpeg",
            user_text="what is this?",
            model="google/gemini-3.1-flash-lite-preview",
        )

        assert result == "A cat on a keyboard"
        # Verify chat was called with correct structure
        call_args = client.chat.call_args
        messages = call_args.kwargs["messages"]
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        # User message should be multimodal (list of content blocks)
        content = messages[1]["content"]
        assert isinstance(content, list)
        types = {block["type"] for block in content}
        assert "text" in types
        assert "image_url" in types

    @pytest.mark.asyncio
    async def test_video_uses_video_url_type(self):
        """Video bytes use video_url content block type."""
        client = MagicMock()
        client.chat = AsyncMock(return_value={
            "choices": [{"message": {"content": "A person waving"}}],
        })
        result = await describe_media(
            client=client,
            media_bytes=b"\x00\x00\x00\x1cftyp",
            media_mime="video/mp4",
            user_text="who is this?",
            model="google/gemini-3.1-flash-lite-preview",
        )

        assert result == "A person waving"
        call_args = client.chat.call_args
        messages = call_args.kwargs["messages"]
        content = messages[1]["content"]
        types = {block["type"] for block in content}
        assert "video_url" in types
        assert "image_url" not in types

    @pytest.mark.asyncio
    async def test_no_user_text_uses_generic_prompt(self):
        """When user_text is None, uses a generic description prompt."""
        client = MagicMock()
        client.chat = AsyncMock(return_value={
            "choices": [{"message": {"content": "A landscape photo"}}],
        })
        await describe_media(
            client=client,
            media_bytes=b"\xff\xd8",
            media_mime="image/jpeg",
            user_text=None,
            model="google/gemini-3.1-flash-lite-preview",
        )

        call_args = client.chat.call_args
        messages = call_args.kwargs["messages"]
        content = messages[1]["content"]
        text_blocks = [b for b in content if b["type"] == "text"]
        assert len(text_blocks) == 1
        assert text_blocks[0]["text"]  # not empty

    @pytest.mark.asyncio
    async def test_empty_user_text_uses_generic_prompt(self):
        """When user_text is empty string, uses a generic description prompt."""
        client = MagicMock()
        client.chat = AsyncMock(return_value={
            "choices": [{"message": {"content": "A photo"}}],
        })
        await describe_media(
            client=client,
            media_bytes=b"\xff\xd8",
            media_mime="image/jpeg",
            user_text="",
            model="google/gemini-3.1-flash-lite-preview",
        )

        call_args = client.chat.call_args
        messages = call_args.kwargs["messages"]
        content = messages[1]["content"]
        text_blocks = [b for b in content if b["type"] == "text"]
        assert text_blocks[0]["text"]  # not empty

    @pytest.mark.asyncio
    async def test_no_tools_passed(self):
        """describe_media should not pass tools to the vision model."""
        client = MagicMock()
        client.chat = AsyncMock(return_value={
            "choices": [{"message": {"content": "desc"}}],
        })
        await describe_media(
            client=client,
            media_bytes=b"\xff\xd8",
            media_mime="image/jpeg",
            user_text="test",
            model="test-model",
        )

        call_args = client.chat.call_args
        assert call_args.kwargs.get("tools") is None

    @pytest.mark.asyncio
    async def test_data_url_format(self):
        """Data URL should have correct MIME type and base64 content."""
        client = MagicMock()
        client.chat = AsyncMock(return_value={
            "choices": [{"message": {"content": "desc"}}],
        })
        await describe_media(
            client=client,
            media_bytes=b"\xff\xd8",
            media_mime="image/png",
            user_text="test",
            model="test-model",
        )

        call_args = client.chat.call_args
        messages = call_args.kwargs["messages"]
        content = messages[1]["content"]
        media_block = [b for b in content if b["type"] == "image_url"][0]
        url = media_block["image_url"]["url"]
        assert url.startswith("data:image/png;base64,")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/zeeshans/myprojects/spare-paw && .venv/bin/python -m pytest tests/test_vision.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'spare_paw.core.vision'`

- [ ] **Step 3: Write minimal implementation**

Create `src/spare_paw/core/vision.py`:

```python
"""Vision preprocessing — describe images/videos via a vision-capable model."""

from __future__ import annotations

import base64
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from spare_paw.router.openrouter import OpenRouterClient

_SYSTEM_PROMPT = (
    "Describe this image/video in the context of the user's message. "
    "Focus on details relevant to what the user is asking or saying."
)

_GENERIC_PROMPT = "Describe this image/video in detail."


async def describe_media(
    *,
    client: OpenRouterClient,
    media_bytes: bytes,
    media_mime: str,
    user_text: str | None,
    model: str,
) -> str:
    """Call a vision model to describe image or video content.

    Returns the model's text description of the media.
    No semaphore needed — client.chat() already acquires app_state.semaphore internally.
    """
    b64 = base64.b64encode(media_bytes).decode("ascii")
    data_url = f"data:{media_mime};base64,{b64}"

    is_video = media_mime.startswith("video/")
    if is_video:
        media_block: dict[str, Any] = {
            "type": "video_url",
            "video_url": {"url": data_url},
        }
    else:
        media_block = {
            "type": "image_url",
            "image_url": {"url": data_url},
        }

    prompt = user_text if user_text else _GENERIC_PROMPT

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                media_block,
            ],
        },
    ]

    response = await client.chat(messages=messages, model=model)
    return response["choices"][0]["message"]["content"]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/zeeshans/myprojects/spare-paw && .venv/bin/python -m pytest tests/test_vision.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/spare_paw/core/vision.py tests/test_vision.py
git commit -m "feat: add describe_media() vision preprocessor"
```

---

### Task 4: Integrate vision preprocessing into engine

**Files:**
- Modify: `src/spare_paw/core/engine.py:57-116`
- Test: `tests/test_engine.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_engine.py`. Update `_make_app_state` to include `models.vision`:

```python
def _make_app_state(response_text="Bot response."):
    app_state = MagicMock()
    app_state.config.get = lambda key, default=None: {
        "models.main_agent": "test-model",
        "models.summary": "summary-model",
        "models.vision": "vision-model",
        "agent.max_tool_iterations": 5,
        "agent.system_prompt": "You are a test bot.",
    }.get(key, default)
    app_state.tool_registry.get_schemas.return_value = []
    app_state.router_client = MagicMock()
    app_state.executor = None
    return app_state
```

Add new test class:

```python
class TestVisionPreprocessing:
    @pytest.mark.asyncio
    async def test_image_calls_describe_media(self):
        """Image message calls describe_media and appends result as user message."""
        app_state = _make_app_state()
        backend = _make_backend()
        msg = IncomingMessage(image_bytes=b"\xff\xd8", caption="what is this?")

        assembled = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "what is this?"},
        ]

        with patch("spare_paw.core.engine.ctx_module") as mock_ctx, \
             patch("spare_paw.core.engine.run_tool_loop", new_callable=AsyncMock, return_value="It's a cat"), \
             patch("spare_paw.core.engine.build_system_prompt", new_callable=AsyncMock, return_value="sys"), \
             patch("spare_paw.core.engine.compact_with_retry", new_callable=AsyncMock), \
             patch("spare_paw.core.engine.describe_media", new_callable=AsyncMock, return_value="A photo of a cat") as mock_describe:
            mock_ctx.get_or_create_conversation = AsyncMock(return_value="conv-1")
            mock_ctx.ingest = AsyncMock(return_value="msg-1")
            mock_ctx.assemble = AsyncMock(return_value=assembled)

            await process_message(app_state, msg, backend)

        mock_describe.assert_awaited_once()
        # Media analysis message should be appended
        analysis_msgs = [m for m in assembled if "[Media analysis]" in m.get("content", "")]
        assert len(analysis_msgs) == 1
        assert analysis_msgs[0]["role"] == "user"
        assert "A photo of a cat" in analysis_msgs[0]["content"]

    @pytest.mark.asyncio
    async def test_video_calls_describe_media(self):
        """Video message calls describe_media with video bytes."""
        app_state = _make_app_state()
        backend = _make_backend()
        msg = IncomingMessage(video_bytes=b"\x00\x00\x00\x1cftyp", caption="what's happening?")

        assembled = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "what's happening?"},
        ]

        with patch("spare_paw.core.engine.ctx_module") as mock_ctx, \
             patch("spare_paw.core.engine.run_tool_loop", new_callable=AsyncMock, return_value="A dance"), \
             patch("spare_paw.core.engine.build_system_prompt", new_callable=AsyncMock, return_value="sys"), \
             patch("spare_paw.core.engine.compact_with_retry", new_callable=AsyncMock), \
             patch("spare_paw.core.engine.describe_media", new_callable=AsyncMock, return_value="A person dancing") as mock_describe:
            mock_ctx.get_or_create_conversation = AsyncMock(return_value="conv-1")
            mock_ctx.ingest = AsyncMock(return_value="msg-1")
            mock_ctx.assemble = AsyncMock(return_value=assembled)

            await process_message(app_state, msg, backend)

        call_kwargs = mock_describe.call_args.kwargs
        assert call_kwargs["media_mime"] == "video/mp4"

    @pytest.mark.asyncio
    async def test_no_media_skips_describe(self):
        """Text-only message does not call describe_media."""
        app_state = _make_app_state()
        backend = _make_backend()
        msg = IncomingMessage(text="hello")

        with patch("spare_paw.core.engine.ctx_module") as mock_ctx, \
             patch("spare_paw.core.engine.run_tool_loop", new_callable=AsyncMock, return_value="Hi"), \
             patch("spare_paw.core.engine.build_system_prompt", new_callable=AsyncMock, return_value="sys"), \
             patch("spare_paw.core.engine.compact_with_retry", new_callable=AsyncMock), \
             patch("spare_paw.core.engine.describe_media", new_callable=AsyncMock) as mock_describe:
            mock_ctx.get_or_create_conversation = AsyncMock(return_value="conv-1")
            mock_ctx.ingest = AsyncMock(return_value="msg-1")
            mock_ctx.assemble = AsyncMock(return_value=[
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "hello"},
            ])

            await process_message(app_state, msg, backend)

        mock_describe.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_image_no_multimodal_content_on_main_agent(self):
        """Image message should NOT inject multimodal content blocks for the main agent."""
        app_state = _make_app_state()
        backend = _make_backend()
        msg = IncomingMessage(image_bytes=b"\xff\xd8", caption="what?")

        assembled = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "what?"},
        ]

        with patch("spare_paw.core.engine.ctx_module") as mock_ctx, \
             patch("spare_paw.core.engine.run_tool_loop", new_callable=AsyncMock, return_value="reply") as mock_loop, \
             patch("spare_paw.core.engine.build_system_prompt", new_callable=AsyncMock, return_value="sys"), \
             patch("spare_paw.core.engine.compact_with_retry", new_callable=AsyncMock), \
             patch("spare_paw.core.engine.describe_media", new_callable=AsyncMock, return_value="desc"):
            mock_ctx.get_or_create_conversation = AsyncMock(return_value="conv-1")
            mock_ctx.ingest = AsyncMock(return_value="msg-1")
            mock_ctx.assemble = AsyncMock(return_value=assembled)

            await process_message(app_state, msg, backend)

        # No message in assembled should have multimodal content (list)
        for m in assembled:
            assert isinstance(m["content"], str), f"Found multimodal content block: {m}"
```

Update the existing `test_image_message` test — it should now verify vision preprocessing behavior instead of multimodal injection. Replace the existing `TestProcessMessage.test_image_message` with:

```python
@pytest.mark.asyncio
async def test_image_message_uses_caption_as_text(self):
    """Image message: uses caption as text, calls describe_media."""
    app_state = _make_app_state()
    backend = _make_backend()
    msg = IncomingMessage(image_bytes=b"\xff\xd8", caption="what is this?")

    with patch("spare_paw.core.engine.ctx_module") as mock_ctx, \
         patch("spare_paw.core.engine.run_tool_loop", new_callable=AsyncMock, return_value="It's a photo"), \
         patch("spare_paw.core.engine.build_system_prompt", new_callable=AsyncMock, return_value="sys"), \
         patch("spare_paw.core.engine.compact_with_retry", new_callable=AsyncMock), \
         patch("spare_paw.core.engine.describe_media", new_callable=AsyncMock, return_value="A photo"):
        mock_ctx.get_or_create_conversation = AsyncMock(return_value="conv-1")
        mock_ctx.ingest = AsyncMock(return_value="msg-1")
        mock_ctx.assemble = AsyncMock(return_value=[
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "what is this?"},
        ])

        await process_message(app_state, msg, backend)

    # Caption should be used as the text for ingestion
    mock_ctx.ingest.assert_any_await("conv-1", "user", "what is this?")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/zeeshans/myprojects/spare-paw && .venv/bin/python -m pytest tests/test_engine.py::TestVisionPreprocessing -v`
Expected: FAIL — `describe_media` not imported in engine

- [ ] **Step 3: Write minimal implementation**

Modify `src/spare_paw/core/engine.py`:

**Add import** (after line 22, the voice import):
```python
from spare_paw.core.vision import describe_media
```

**Replace lines 77-93** (the text determination + image handling block) with:

```python
    # 1. Determine text content
    text = msg.text
    media_description = None

    if msg.voice_bytes:
        try:
            config_data = getattr(app_state.config, "data", app_state.config)
            text = await transcribe(msg.voice_bytes, config_data)
        except VoiceTranscriptionError:
            logger.exception("Voice transcription failed")
            return

    # 2. Vision preprocessing for images/videos
    media_bytes = msg.image_bytes or msg.video_bytes
    if media_bytes:
        media_mime = msg.video_mime if msg.video_bytes else msg.image_mime
        user_text = msg.text or msg.caption
        vision_model = resolve_model(app_state.config, "vision")
        media_description = await describe_media(
            client=app_state.router_client,
            media_bytes=media_bytes,
            media_mime=media_mime,
            user_text=user_text,
            model=vision_model,
        )
        if not text:
            text = msg.caption or "Sent media"

    if not text:
        return
```

Note: the `if not text: return` guard (currently at line 92) is moved to AFTER the media block. This ensures video/image-only messages (no text, no caption) still get `text = "Sent media"` and don't return early. The guard catches truly empty messages (no text, no media, no voice).

**Remove lines 108-116** (the old multimodal image injection block — step 5 "Image: make last user message multimodal"):
Delete the entire block:
```python
    # 5. Image: make last user message multimodal
    if image_url and messages:
        for i in range(len(messages) - 1, -1, -1):
            if messages[i]["role"] == "user":
                messages[i]["content"] = [
                    {"type": "text", "text": text},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ]
                break
```

**Add media description injection** after context assembly (after the `messages = await ctx.assemble(...)` line):
```python
    # 5. Inject media description
    if media_description:
        messages.append({
            "role": "user",
            "content": f"[Media analysis]: {media_description}",
        })
```

Also remove the now-unused `import base64` from line 11.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/zeeshans/myprojects/spare-paw && .venv/bin/python -m pytest tests/test_engine.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/spare_paw/core/engine.py tests/test_engine.py
git commit -m "feat: integrate vision preprocessing into engine"
```

---

### Task 5: Add video handlers to Telegram bot

**Files:**
- Modify: `src/spare_paw/bot/handler.py:34-55` (setup_handlers), `src/spare_paw/bot/handler.py:64-93` (_queue_message)
- Create: `src/spare_paw/bot/handler.py` — new `_download_video` helper
- Test: `tests/test_engine.py` (video already tested in Task 4)

- [ ] **Step 1: Write the failing test**

Add to a new file `tests/test_handler.py` (or add to existing `tests/test_engine.py` if no handler-specific test file exists). Since handler tests require mocking Telegram objects, add a focused unit test:

```python
"""Tests for bot/handler.py — Telegram message handlers."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestQueueMessageVideo:
    @pytest.mark.asyncio
    async def test_video_message_creates_incoming_with_video_bytes(self):
        """Video message populates video_bytes and video_mime on IncomingMessage."""
        from spare_paw.bot.handler import _queue_message

        update = MagicMock()
        update.effective_user.id = 12345
        update.message.text = None
        update.message.voice = None
        update.message.photo = None
        update.message.caption = "check this"
        update.message.reply_to_message = None

        # Video mock
        video_file = MagicMock()
        video_file.download_as_bytearray = AsyncMock(return_value=bytearray(b"\x00\x01\x02"))
        video_obj = MagicMock()
        video_obj.get_file = AsyncMock(return_value=video_file)
        video_obj.mime_type = "video/webm"
        update.message.video = video_obj
        update.message.video_note = None

        context = MagicMock()
        app_state = MagicMock()
        app_state.config.get.return_value = 12345  # owner_id
        context.bot_data = {"app_state": app_state}

        with patch("spare_paw.bot.handler.enqueue", new_callable=AsyncMock) as mock_enqueue:
            await _queue_message(update, context)

        msg = mock_enqueue.call_args.args[0]
        assert msg.video_bytes == b"\x00\x01\x02"
        assert msg.video_mime == "video/webm"

    @pytest.mark.asyncio
    async def test_video_note_uses_mp4_mime(self):
        """Video note (circle video) always uses video/mp4."""
        from spare_paw.bot.handler import _queue_message

        update = MagicMock()
        update.effective_user.id = 12345
        update.message.text = None
        update.message.voice = None
        update.message.photo = None
        update.message.video = None
        update.message.caption = None
        update.message.reply_to_message = None

        video_note_file = MagicMock()
        video_note_file.download_as_bytearray = AsyncMock(return_value=bytearray(b"\x00\x01"))
        video_note_obj = MagicMock()
        video_note_obj.get_file = AsyncMock(return_value=video_note_file)
        update.message.video_note = video_note_obj

        context = MagicMock()
        app_state = MagicMock()
        app_state.config.get.return_value = 12345
        context.bot_data = {"app_state": app_state}

        with patch("spare_paw.bot.handler.enqueue", new_callable=AsyncMock) as mock_enqueue:
            await _queue_message(update, context)

        msg = mock_enqueue.call_args.args[0]
        assert msg.video_bytes == b"\x00\x01"
        assert msg.video_mime == "video/mp4"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/zeeshans/myprojects/spare-paw && .venv/bin/python -m pytest tests/test_handler.py -v`
Expected: FAIL — `IncomingMessage` has no `video_bytes` being set in handler

- [ ] **Step 3: Write minimal implementation**

In `src/spare_paw/bot/handler.py`:

**Add video handler registrations** in `setup_handlers()`, after the PHOTO handler (line 53):
```python
    # Video messages
    application.add_handler(
        MessageHandler(filters.VIDEO, _queue_message)
    )
    # Video note (circle videos)
    application.add_handler(
        MessageHandler(filters.VIDEO_NOTE, _queue_message)
    )
```

**Update `_queue_message`** to handle video. Replace the `IncomingMessage` construction (lines 84-91) with:

```python
    # Download media
    voice_bytes = (await _download_voice(message)) if message.voice else None
    image_bytes = (await _download_photo(message)) if message.photo else None
    video_bytes, video_mime = await _download_video(message)

    msg = IncomingMessage(
        text=message.text,
        voice_bytes=voice_bytes,
        image_bytes=image_bytes,
        video_bytes=video_bytes,
        video_mime=video_mime,
        caption=message.caption,
        cron_context=_extract_cron_context(update),
        user_id=update.effective_user.id if update.effective_user else None,
    )
```

**Add `_download_video` helper** after `_download_photo`:

```python
async def _download_video(message: Any) -> tuple[bytes | None, str]:
    """Download video or video_note bytes from Telegram.

    Returns (video_bytes, mime_type). Video notes are always MP4.
    """
    if message.video:
        try:
            video_file = await message.video.get_file()
            data = bytes(await video_file.download_as_bytearray())
            mime = getattr(message.video, "mime_type", "video/mp4") or "video/mp4"
            return data, mime
        except Exception:
            logger.exception("Failed to download video")
            return None, "video/mp4"

    if message.video_note:
        try:
            vn_file = await message.video_note.get_file()
            data = bytes(await vn_file.download_as_bytearray())
            return data, "video/mp4"
        except Exception:
            logger.exception("Failed to download video note")
            return None, "video/mp4"

    return None, "video/mp4"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/zeeshans/myprojects/spare-paw && .venv/bin/python -m pytest tests/test_handler.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/spare_paw/bot/handler.py tests/test_handler.py
git commit -m "feat: add video and video_note handlers to Telegram bot"
```

---

### Task 6: Add video support to webhook backend

**Files:**
- Modify: `src/spare_paw/webhook/backend.py:85-117`
- Test: `tests/test_webhook_backend.py`

- [ ] **Step 1: Write the failing test**

Check existing test patterns first by reading `tests/test_webhook_backend.py`, then add:

```python
class TestWebhookVideoMessage:
    @pytest.mark.asyncio
    async def test_video_field_decoded(self):
        """Webhook message with 'video' field decodes to video_bytes."""
        import base64
        from spare_paw.webhook.backend import WebhookBackend

        backend = WebhookBackend(port=0, secret="test-secret", app_state=MagicMock())

        video_b64 = base64.b64encode(b"\x00\x01\x02").decode()
        # We need to test the message construction in _handle_message
        # Mock the request
        request = MagicMock()
        request.json = AsyncMock(return_value={
            "text": "check this video",
            "video": video_b64,
        })
        request.headers = {"Authorization": "Bearer test-secret", "X-Session-Id": "s1"}

        with patch.object(backend, "_process_message", new_callable=AsyncMock) as mock_process:
            response = await backend._handle_message(request)

        msg = mock_process.call_args.args[0]
        assert msg.video_bytes == b"\x00\x01\x02"
        assert msg.video_mime == "video/mp4"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/zeeshans/myprojects/spare-paw && .venv/bin/python -m pytest tests/test_webhook_backend.py::TestWebhookVideoMessage -v`
Expected: FAIL — `IncomingMessage` not getting `video_bytes`

- [ ] **Step 3: Write minimal implementation**

In `src/spare_paw/webhook/backend.py`, in `_handle_message()`, after line 96 (`voice_b64 = data.get("voice")`), add:

```python
        video_b64 = data.get("video")
```

After line 99 (`voice_bytes = ...`), add:

```python
        video_bytes = base64.b64decode(video_b64) if video_b64 else None
```

Update the `IncomingMessage` construction (lines 101-105) to include video:

```python
        msg = IncomingMessage(
            text=text,
            image_bytes=image_bytes,
            voice_bytes=voice_bytes,
            video_bytes=video_bytes,
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/zeeshans/myprojects/spare-paw && .venv/bin/python -m pytest tests/test_webhook_backend.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/spare_paw/webhook/backend.py tests/test_webhook_backend.py
git commit -m "feat: add video support to webhook backend"
```

---

### Task 7: Full test suite + lint

**Files:** All modified files

- [ ] **Step 1: Run full test suite (excluding slow tests)**

Run: `cd /Users/zeeshans/myprojects/spare-paw && .venv/bin/python -m pytest tests/ -v --ignore=tests/test_pipe_integration.py -x`
Expected: All PASS

- [ ] **Step 2: Run linter**

Run: `cd /Users/zeeshans/myprojects/spare-paw && .venv/bin/ruff check src/ tests/`
Expected: No errors

- [ ] **Step 3: Fix any failures**

If any tests fail or lint errors appear, fix them and re-run.

- [ ] **Step 4: Commit any fixes**

```bash
git add -u
git commit -m "fix: address test/lint issues from vision preprocessing"
```

---

### Task 8: Deploy to phone and verify

**Files:** Changed source files need to be SCP'd to `termux-phone:~/claw-phone/`

- [ ] **Step 1: SCP changed files to phone**

```bash
scp src/spare_paw/config.py termux-phone:~/claw-phone/src/spare_paw/config.py
scp src/spare_paw/backend.py termux-phone:~/claw-phone/src/spare_paw/backend.py
scp src/spare_paw/core/vision.py termux-phone:~/claw-phone/src/spare_paw/core/vision.py
scp src/spare_paw/core/engine.py termux-phone:~/claw-phone/src/spare_paw/core/engine.py
scp src/spare_paw/bot/handler.py termux-phone:~/claw-phone/src/spare_paw/bot/handler.py
scp src/spare_paw/webhook/backend.py termux-phone:~/claw-phone/src/spare_paw/webhook/backend.py
```

- [ ] **Step 2: Restart the bot**

```bash
ssh termux-phone "pkill -f 'spare_paw'"
ssh termux-phone "cd ~/claw-phone && nohup python -m spare_paw gateway > /data/data/com.termux/files/usr/tmp/spare-paw.log 2>&1 &"
```

- [ ] **Step 3: Verify with a photo**

Send a photo to the bot via Telegram with a caption. Check logs to confirm `describe_media` was called and the main agent received the text description.

```bash
ssh termux-phone "tail -50 /data/data/com.termux/files/usr/tmp/spare-paw.log | grep -v 'getUpdates'"
```

- [ ] **Step 4: Verify with a video**

Send a short video to the bot via Telegram. Check logs to confirm video was processed.

- [ ] **Step 5: Update phone config (if needed)**

If `models.vision` needs to be set explicitly:
```bash
ssh termux-phone "cat ~/.spare-paw/config.yaml"
```

The default (`google/gemini-3.1-flash-lite-preview`) should be fine since it's already used as the summary model.
