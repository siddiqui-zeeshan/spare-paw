"""Tests for TelegramBackend — Telegram implementation of MessageBackend."""

from __future__ import annotations

import re
from unittest.mock import AsyncMock, MagicMock

import pytest

from spare_paw.backend import MessageBackend
from spare_paw.bot.backend import TelegramBackend, md_to_html


# ---------------------------------------------------------------------------
# md_to_html tests (moved from test_telegram_format.py)
# ---------------------------------------------------------------------------


class TestMdToHtml:
    def test_bold(self):
        assert md_to_html("**hello**") == "<b>hello</b>"

    def test_italic(self):
        assert md_to_html("*hello*") == "<i>hello</i>"

    def test_inline_code(self):
        assert md_to_html("`foo()`") == "<code>foo()</code>"

    def test_code_block(self):
        md = "```python\nprint('hi')\n```"
        expected = '<pre><code class="language-python">print(\'hi\')</code></pre>'
        assert md_to_html(md) == expected

    def test_code_block_no_language(self):
        md = "```\nprint('hi')\n```"
        expected = "<pre><code>print('hi')</code></pre>"
        assert md_to_html(md) == expected

    def test_link(self):
        assert md_to_html("[click](https://example.com)") == '<a href="https://example.com">click</a>'

    def test_strikethrough(self):
        assert md_to_html("~~deleted~~") == "<s>deleted</s>"

    def test_html_escaping(self):
        result = md_to_html("<script>")
        assert result == "&lt;script&gt;"

    def test_code_block_preserves_content(self):
        md = "```\n**not bold** and *not italic*\n```"
        expected = "<pre><code>**not bold** and *not italic*</code></pre>"
        assert md_to_html(md) == expected

    def test_plain_text_unchanged(self):
        assert md_to_html("hello world") == "hello world"

    def test_mixed_formatting(self):
        md = "Use **bold** and `code` and [link](https://example.com) together"
        result = md_to_html(md)
        assert "<b>bold</b>" in result
        assert "<code>code</code>" in result
        assert '<a href="https://example.com">link</a>' in result

    def test_heading_h3(self):
        assert md_to_html("### Time Complexity") == "<b>Time Complexity</b>"

    def test_heading_h2(self):
        assert md_to_html("## Section Title") == "<b>Section Title</b>"

    def test_heading_h1(self):
        assert md_to_html("# Main Title") == "<b>Main Title</b>"

    def test_bullet_list_not_italic(self):
        md = "*   Item one\n*   Item two"
        result = md_to_html(md)
        assert "<i>" not in result
        assert "Item one" in result
        assert "Item two" in result

    def test_latex_dollar_signs(self):
        md = "Time complexity is $O(n)$ and space is $O(1)$."
        result = md_to_html(md)
        assert "O(n)" in result
        assert "O(1)" in result

    def test_heading_with_inline_bold(self):
        md = "### Time Complexity: **O(n)**"
        result = md_to_html(md)
        assert "<b>" in result
        assert "O(n)" in result

    def test_simple_table(self):
        md = "| Model | Price |\n|-------|-------|\n| GPT-4 | $10 |\n| Claude | $8 |"
        result = md_to_html(md)
        assert "GPT-4" in result
        assert "Claude" in result
        assert "<pre>" in result

    def test_table_preserves_content(self):
        md = "| Name | Value |\n|------|-------|\n| foo | 42 |"
        result = md_to_html(md)
        assert "foo" in result
        assert "42" in result

    def test_code_block_escapes_html_chars(self):
        md = "```python\nif n <= 0:\n    return n > 1\n```"
        result = md_to_html(md)
        assert "&lt;" in result
        assert "&gt;" in result
        assert "<=" not in result.replace("&lt;=", "")

    def test_inline_code_escapes_html_chars(self):
        result = md_to_html("`a < b && c > d`")
        assert "<code>" in result
        assert "&lt;" in result
        assert "&gt;" in result
        assert "&amp;" in result

    def test_multiple_code_blocks(self):
        md = "First:\n```python\nx = 1\n```\nSecond:\n```js\ny = 2\n```"
        result = md_to_html(md)
        assert "x = 1" in result
        assert "y = 2" in result
        assert result.count("<pre>") == 2

    def test_empty_input(self):
        assert md_to_html("") == ""

    def test_ampersand_in_text(self):
        result = md_to_html("AT&T and H&M")
        assert "AT&amp;T" in result
        assert "H&amp;M" in result

    def test_numbered_list(self):
        md = "1. First item\n2. Second item\n3. Third item"
        result = md_to_html(md)
        assert "1. First item" in result
        assert "3. Third item" in result

    def test_bold_italic_combined(self):
        result = md_to_html("***hello***")
        assert "<b>" in result or "<i>" in result
        assert "hello" in result

    def test_inline_code_not_formatted(self):
        result = md_to_html("`**not bold**`")
        assert "<b>" not in result
        assert "<code>" in result

    def test_real_world_llm_response(self):
        md = (
            "## Summary\n\n"
            "Here's what I found:\n\n"
            "**Model A** costs `$0.50` per million tokens. "
            "See [pricing](https://example.com/pricing).\n\n"
            "```python\ndef calc(n):\n    return n * 0.50\n```\n\n"
            "| Model | Cost |\n|-------|------|\n| A | $0.50 |\n| B | $1.00 |\n\n"
            "~~Old pricing~~ is no longer valid."
        )
        result = md_to_html(md)
        assert "<b>Summary</b>" in result
        assert "<b>Model A</b>" in result
        assert "<code>$0.50</code>" in result
        assert '<a href="https://example.com/pricing">' in result
        assert "<pre><code" in result
        assert "<s>Old pricing</s>" in result
        stripped = re.sub(r"<[a-z/][^>]*>", "", result)
        assert "<" not in stripped, f"Unescaped < found in: {stripped}"


# ---------------------------------------------------------------------------
# TelegramBackend tests
# ---------------------------------------------------------------------------


def _make_backend() -> tuple[TelegramBackend, MagicMock]:
    """Create a TelegramBackend with a mocked bot and application."""
    mock_bot = AsyncMock()
    mock_application = MagicMock()
    mock_application.bot = mock_bot
    mock_application.updater = MagicMock()
    mock_application.updater.start_polling = AsyncMock()
    mock_application.updater.stop = AsyncMock()
    mock_application.initialize = AsyncMock()
    mock_application.start = AsyncMock()
    mock_application.stop = AsyncMock()
    mock_application.shutdown = AsyncMock()
    mock_application.bot_data = {}

    backend = TelegramBackend(mock_application, chat_id=12345)
    return backend, mock_bot


class TestTelegramBackendProtocol:
    def test_satisfies_message_backend(self):
        backend, _ = _make_backend()
        assert isinstance(backend, MessageBackend)


class TestSendText:
    @pytest.mark.asyncio
    async def test_sends_bold_as_html(self):
        backend, bot = _make_backend()
        await backend.send_text("**bold**")
        bot.send_message.assert_called()
        call_kwargs = bot.send_message.call_args
        assert "<b>bold</b>" in call_kwargs.kwargs.get("text", call_kwargs[1].get("text", ""))

    @pytest.mark.asyncio
    async def test_parse_mode_html(self):
        backend, bot = _make_backend()
        await backend.send_text("hello")
        call_kwargs = bot.send_message.call_args
        # Check parse_mode is HTML
        pm = call_kwargs.kwargs.get("parse_mode") or call_kwargs[1].get("parse_mode")
        assert pm == "HTML"

    @pytest.mark.asyncio
    async def test_chunks_long_text(self):
        backend, bot = _make_backend()
        long_text = "a" * 5000
        await backend.send_text(long_text)
        assert bot.send_message.call_count >= 2

    @pytest.mark.asyncio
    async def test_falls_back_to_plain_on_error(self):
        backend, bot = _make_backend()
        # First call raises, second succeeds (plain text fallback)
        bot.send_message.side_effect = [Exception("HTML parse error"), None]
        await backend.send_text("hello")
        assert bot.send_message.call_count == 2
        second_call = bot.send_message.call_args_list[1]
        # Fallback should not have parse_mode=HTML
        pm = second_call.kwargs.get("parse_mode") or second_call[1].get("parse_mode")
        assert pm is None

    @pytest.mark.asyncio
    async def test_empty_text(self):
        backend, bot = _make_backend()
        await backend.send_text("")
        bot.send_message.assert_called()
        call_kwargs = bot.send_message.call_args
        text = call_kwargs.kwargs.get("text", call_kwargs[1].get("text", ""))
        assert text  # should send non-empty fallback


class TestSendFile:
    @pytest.mark.asyncio
    async def test_send_photo(self, tmp_path):
        backend, bot = _make_backend()
        photo = tmp_path / "photo.jpg"
        photo.write_bytes(b"\xff\xd8\xff")
        await backend.send_file(str(photo))
        bot.send_photo.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_document(self, tmp_path):
        backend, bot = _make_backend()
        doc = tmp_path / "doc.pdf"
        doc.write_bytes(b"%PDF")
        await backend.send_file(str(doc))
        bot.send_document.assert_called_once()


class TestSendTyping:
    @pytest.mark.asyncio
    async def test_send_typing(self):
        backend, bot = _make_backend()
        await backend.send_typing()
        bot.send_chat_action.assert_called_once()


class TestSendNotification:
    @pytest.mark.asyncio
    async def test_with_actions_creates_keyboard(self):
        backend, bot = _make_backend()
        actions = [
            {"label": "Approve", "callback_data": "approve:test"},
            {"label": "Reject", "callback_data": "reject:test"},
        ]
        await backend.send_notification("Choose:", actions=actions)
        bot.send_message.assert_called_once()
        call_kwargs = bot.send_message.call_args
        assert call_kwargs.kwargs.get("reply_markup") is not None

    @pytest.mark.asyncio
    async def test_without_actions_plain_text(self):
        backend, bot = _make_backend()
        await backend.send_notification("Alert!")
        bot.send_message.assert_called_once()
        call_kwargs = bot.send_message.call_args
        assert call_kwargs.kwargs.get("reply_markup") is None


class TestSetAppState:
    def test_stores_app_state(self):
        backend, _ = _make_backend()
        mock_app_state = MagicMock()
        backend.set_app_state(mock_app_state)
        assert backend._application.bot_data["app_state"] is mock_app_state
