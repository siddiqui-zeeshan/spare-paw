"""TelegramBackend — Telegram implementation of MessageBackend.

Converts Markdown to Telegram HTML, chunks long messages, and delegates
all I/O to the python-telegram-bot library.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

from telegram import InlineKeyboardButton, InlineKeyboardMarkup
from telegram.constants import ChatAction, ParseMode

from spare_paw.core.engine import split_text

logger = logging.getLogger(__name__)

_MAX_MESSAGE_LENGTH = 4096

# ---------------------------------------------------------------------------
# Markdown → Telegram HTML conversion
# ---------------------------------------------------------------------------

_CODE_BLOCK_RE = re.compile(r"```(\w*)\n(.*?)```", re.DOTALL)
_INLINE_CODE_RE = re.compile(r"`([^`\n]+)`")
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
_BOLD_RE = re.compile(r"\*\*(.+?)\*\*", re.DOTALL)
_ITALIC_RE = re.compile(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)")
_STRIKE_RE = re.compile(r"~~(.+?)~~")
_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")

_TABLE_RE = re.compile(
    r"((?:^\|.+\|$\n?)+)",
    re.MULTILINE,
)


def convert_tables(text: str) -> str:
    """Convert Markdown tables to monospace <pre> blocks for Telegram."""

    def _render_table(m: re.Match) -> str:
        lines = m.group(1).strip().split("\n")
        rows: list[list[str]] = []
        for line in lines:
            cells = [c.strip() for c in line.strip("|").split("|")]
            if all(c.replace("-", "").replace(":", "") == "" for c in cells):
                continue
            rows.append(cells)
        if not rows:
            return m.group(0)
        num_cols = max(len(r) for r in rows)
        widths = [0] * num_cols
        for row in rows:
            for i, cell in enumerate(row):
                if i < num_cols:
                    widths[i] = max(widths[i], len(cell))
        output_lines = []
        for row in rows:
            parts = []
            for i in range(num_cols):
                cell = row[i] if i < len(row) else ""
                parts.append(cell.ljust(widths[i]))
            output_lines.append("  ".join(parts))
        return "<pre>" + "\n".join(output_lines) + "</pre>"

    return _TABLE_RE.sub(_render_table, text)


def md_to_html(text: str) -> str:
    """Convert standard Markdown to Telegram-supported HTML."""
    code_blocks: list[str] = []

    def _save_code_block(m: re.Match) -> str:
        lang = m.group(1)
        code = m.group(2).rstrip("\n")
        code = code.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        idx = len(code_blocks)
        if lang:
            code_blocks.append(
                f'<pre><code class="language-{lang}">{code}</code></pre>'
            )
        else:
            code_blocks.append(f"<pre><code>{code}</code></pre>")
        return f"\x00CODEBLOCK{idx}\x00"

    text = _CODE_BLOCK_RE.sub(_save_code_block, text)

    inline_codes: list[str] = []

    def _save_inline_code(m: re.Match) -> str:
        idx = len(inline_codes)
        escaped_content = (
            m.group(1).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        )
        inline_codes.append(f"<code>{escaped_content}</code>")
        return f"\x00INLINE{idx}\x00"

    text = _INLINE_CODE_RE.sub(_save_inline_code, text)

    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    text = convert_tables(text)

    text = _HEADING_RE.sub(r"<b>\2</b>", text)
    text = _BOLD_RE.sub(r"<b>\1</b>", text)
    text = _ITALIC_RE.sub(r"<i>\1</i>", text)
    text = _STRIKE_RE.sub(r"<s>\1</s>", text)
    text = _LINK_RE.sub(r'<a href="\2">\1</a>', text)

    for idx, code in enumerate(inline_codes):
        text = text.replace(f"\x00INLINE{idx}\x00", code)
    for idx, block in enumerate(code_blocks):
        text = text.replace(f"\x00CODEBLOCK{idx}\x00", block)

    return text


# ---------------------------------------------------------------------------
# TelegramBackend class
# ---------------------------------------------------------------------------


class TelegramBackend:
    """Telegram implementation of MessageBackend."""

    def __init__(self, application: Any, chat_id: int) -> None:
        self._application = application
        self._chat_id = chat_id

    @property
    def bot(self) -> Any:
        return self._application.bot

    def set_app_state(self, app_state: Any) -> None:
        self._application.bot_data["app_state"] = app_state

    async def send_text(self, text: str) -> None:
        if not text:
            text = "(empty response)"

        chunks = split_text(text, _MAX_MESSAGE_LENGTH)
        for chunk in chunks:
            try:
                html = md_to_html(chunk)
                await self.bot.send_message(
                    chat_id=self._chat_id,
                    text=html,
                    parse_mode=ParseMode.HTML,
                )
            except Exception:
                await self.bot.send_message(
                    chat_id=self._chat_id,
                    text=chunk,
                )

    async def send_file(self, path: str, caption: str = "") -> None:
        fpath = Path(path)
        suffix = fpath.suffix.lower()
        with open(fpath, "rb") as f:
            if suffix in (".jpg", ".jpeg", ".png", ".gif", ".webp"):
                await self.bot.send_photo(
                    chat_id=self._chat_id, photo=f, caption=caption or None,
                )
            elif suffix in (".mp4", ".mov", ".avi"):
                await self.bot.send_video(
                    chat_id=self._chat_id, video=f, caption=caption or None,
                )
            elif suffix in (".mp3", ".ogg", ".m4a", ".wav"):
                await self.bot.send_audio(
                    chat_id=self._chat_id, audio=f, caption=caption or None,
                )
            else:
                await self.bot.send_document(
                    chat_id=self._chat_id, document=f, caption=caption or None,
                )

    async def send_typing(self) -> None:
        await self.bot.send_chat_action(
            chat_id=self._chat_id, action=ChatAction.TYPING,
        )

    async def send_notification(
        self, text: str, actions: list[dict] | None = None,
    ) -> None:
        reply_markup = None
        if actions:
            buttons = [
                InlineKeyboardButton(
                    a["label"], callback_data=a["callback_data"],
                )
                for a in actions
            ]
            reply_markup = InlineKeyboardMarkup([buttons])

        await self.bot.send_message(
            chat_id=self._chat_id,
            text=text,
            reply_markup=reply_markup,
        )

    async def start(self) -> None:
        from telegram import BotCommand

        await self._application.initialize()
        await self._application.bot.set_my_commands([
            BotCommand("cron", "Manage scheduled tasks (list, remove, pause, resume, info)"),
            BotCommand("config", "Show or change runtime config"),
            BotCommand("status", "Uptime, memory, DB size, active crons"),
            BotCommand("search", "Full-text search over conversation history"),
            BotCommand("forget", "Start a new conversation"),
            BotCommand("model", "Switch the active model"),
            BotCommand("mcp", "List connected MCP servers and tools"),
        ])
        await self._application.start()
        if self._application.updater is not None:
            await self._application.updater.start_polling(drop_pending_updates=True)

    async def stop(self) -> None:
        if self._application.updater is not None:
            await self._application.updater.stop()
        await self._application.stop()
        await self._application.shutdown()
