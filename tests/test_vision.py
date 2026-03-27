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
        call_args = client.chat.call_args
        messages = call_args.kwargs["messages"]
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
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
