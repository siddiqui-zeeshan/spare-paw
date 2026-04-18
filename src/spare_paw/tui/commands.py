"""Slash-command router for the TUI.

State (app_state) is injected explicitly rather than accessed via
``self.<anything>``, making every branch trivially unit-testable and
preventing attribute-typo bugs.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".gif", ".webp"})

HELP_TEXT = (
    "**Commands:** /help, /exit, /forget, /status, /model, /models, "
    "/roles, /image /path, /plan <prompt>\n"
    "**Keys:** Ctrl+C Exit, Ctrl+L Clear, Ctrl+N New, Esc Cancel, F1 Help, "
    "Ctrl+F Search, PgUp/PgDn Scroll, ↑/↓ Input history, Shift+Enter Newline"
)


@dataclass
class CommandResult:
    kind: str  # "send" | "send_image" | "text" | "quit" | "forget"
    content: str = ""        # rich text or markdown to display
    text: str = ""           # user text to send to engine
    plan: bool = False
    image_b64: str | None = None
    caption: str = ""


class SlashCommandRouter:
    """Dispatches user input to the appropriate action."""

    def __init__(self, app_state: Any | None) -> None:
        self._app_state = app_state

    async def dispatch(self, raw: str) -> CommandResult:
        text = raw.strip()
        if not text:
            return CommandResult(kind="text", content="")

        lower = text.lower()

        if lower in ("/exit", "/quit"):
            return CommandResult(kind="quit")

        if lower == "/help":
            return CommandResult(kind="text", content=HELP_TEXT)

        if lower == "/forget":
            return CommandResult(kind="forget")

        if lower == "/roles":
            return await self._roles()

        if lower.startswith("/models"):
            return await self._models(text[len("/models"):].strip() or None)

        if lower.startswith("/model"):
            parts = text.split()
            return await self._model(parts[1:] if len(parts) > 1 else None)

        if lower.startswith("/image "):
            return self._image(text[len("/image "):].strip())

        if lower == "/plan" or lower.startswith("/plan "):
            body = text[len("/plan"):].strip()
            if not body:
                return CommandResult(kind="text", content="Usage: /plan <prompt>")
            return CommandResult(kind="send", text=body, plan=True)

        if text.startswith("/"):
            return CommandResult(
                kind="text",
                content=f"Unknown command: `{text.split()[0]}`. Type /help.",
            )

        return CommandResult(kind="send", text=text, plan=False)

    async def _roles(self) -> CommandResult:
        if self._app_state is None:
            return CommandResult(kind="text", content="(no app state — /roles unavailable)")
        from spare_paw.core.commands import cmd_roles
        content = await cmd_roles(self._app_state)
        return CommandResult(kind="text", content=content)

    async def _models(self, query: str | None) -> CommandResult:
        if self._app_state is None:
            return CommandResult(kind="text", content="(no app state — /models unavailable)")
        from spare_paw.core.commands import cmd_models
        content = await cmd_models(self._app_state, query)
        return CommandResult(kind="text", content=content)

    async def _model(self, args: list[str] | None) -> CommandResult:
        if self._app_state is None:
            return CommandResult(kind="text", content="(no app state — /model unavailable)")
        from spare_paw.core.commands import cmd_model
        content = await cmd_model(self._app_state, args)
        return CommandResult(kind="text", content=content)

    def _image(self, rest: str) -> CommandResult:
        if not rest:
            return CommandResult(kind="text", content="Usage: /image <path> [caption]")
        parts = rest.split(None, 1)
        path = parts[0]
        caption = parts[1] if len(parts) > 1 else "What do you see in this image?"
        fpath = Path(path).expanduser()
        if not fpath.exists():
            return CommandResult(kind="text", content=f"[red]File not found: {path}[/red]")
        if fpath.suffix.lower() not in _IMAGE_EXTENSIONS:
            return CommandResult(
                kind="text",
                content=f"[red]Unsupported image format: {fpath.suffix}[/red]",
            )
        image_b64 = base64.b64encode(fpath.read_bytes()).decode("ascii")
        return CommandResult(
            kind="send_image", image_b64=image_b64, caption=caption, text=caption,
        )
