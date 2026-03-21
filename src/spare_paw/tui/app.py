"""Textual TUI application for spare-paw."""

from __future__ import annotations

import asyncio
import base64
import random
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    from textual.app import App, ComposeResult
    from textual.binding import Binding
    from textual.containers import Vertical
    from textual.message import Message
    from textual.widgets import Footer, Header, Input, RichLog, Static

    HAS_TEXTUAL = True
except ImportError:
    HAS_TEXTUAL = False
    Message = object  # type: ignore[assignment,misc]

_IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".gif", ".webp"})

SPINNER_FRAMES = ("·", "✢", "✳", "∗", "✻", "✽", "✻", "∗", "✳", "✢")

THINKING_VERBS = (
    "Purring",
    "Pawing",
    "Whisker-twitching",
    "Tail-swishing",
    "Hunting",
    "Pouncing",
    "Grooming",
    "Kneading",
    "Stretching",
    "Sniffing",
    "Stalking",
    "Conjuring",
    "Crystallizing",
    "Calibrating",
)

TOOL_ICONS = {
    "read_file": "⬚",
    "write_file": "□",
    "edit_file": "□",
    "shell": "⟩",
    "web_search": "◎",
    "web_scrape": "◎",
}

SLASH_COMMANDS = [
    "/help",
    "/exit",
    "/quit",
    "/forget",
    "/status",
    "/model",
    "/models",
    "/roles",
    "/image",
]


def _format_timestamp(dt: datetime | None = None) -> str:
    if dt is None:
        dt = datetime.now()
    return dt.strftime("%-I:%M %p")


class AppendLog(Message):
    """Thread-safe message to write content to the chat log."""

    def __init__(self, content: Any) -> None:
        super().__init__()
        self.content = content


class UpdateStatus(Message):
    """Thread-safe message to update the status bar."""

    def __init__(self, text: str) -> None:
        super().__init__()
        self.text = text


class StreamToken(Message):
    """A single streaming token from the model."""

    def __init__(self, token: str) -> None:
        super().__init__()
        self.token = token


class StreamEnd(Message):
    """Signal that streaming has finished."""

    def __init__(self) -> None:
        super().__init__()


class ToolCallEvent(Message):
    """A tool call to display."""

    def __init__(self, tool: str, args: str) -> None:
        super().__init__()
        self.tool = tool
        self.args = args


class TUIBackend:
    """MessageBackend that updates TUI widgets via thread-safe messages."""

    def __init__(self, app: Any) -> None:
        self._app = app

    async def send_text(self, text: str) -> None:
        from rich.markdown import Markdown

        if hasattr(self._app, "_hide_thinking"):
            self._app._hide_thinking()
        self._app.post_message(AppendLog(Markdown(text)))
        self._app.post_message(AppendLog(""))
        if hasattr(self._app, "_last_role"):
            self._app._last_role = "assistant"

    async def send_file(self, path: str, caption: str = "") -> None:
        msg = f"File: {path}"
        if caption:
            msg += f" — {caption}"
        self._app.post_message(AppendLog(f"[dim]{msg}[/dim]"))

    async def send_typing(self) -> None:
        pass

    async def send_notification(
        self, text: str, actions: list[dict] | None = None
    ) -> None:
        self._app.post_message(AppendLog(f"[yellow]{text}[/yellow]"))

    def on_tool_event(self, event: Any) -> None:
        if event.kind == "tool_start" and event.tool_name:
            args_str = ""
            if event.tool_args:
                args_str = ", ".join(
                    f"{k}={v!r}" for k, v in list(event.tool_args.items())[:3]
                )
            self._app.post_message(ToolCallEvent(event.tool_name, args_str))

    def on_token(self, token: str) -> None:
        self._app.post_message(StreamToken(token))

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass


if HAS_TEXTUAL:
    from textual.suggester import SuggestFromList

    class SparePawTUI(App):
        """TUI interface for spare-paw."""

        TITLE = "spare-paw"

        CSS = """
        Screen {
            background: $surface;
        }
        #chat-log {
            height: 1fr;
            padding: 1 0;
            background: $surface;
        }
        #stream-buffer {
            padding: 0 2;
            color: $text-muted;
            text-style: italic;
            display: none;
        }
        #tool-display {
            padding: 0 2;
            opacity: 60%;
            display: none;
        }
        #thinking {
            padding: 0 2;
            color: $text-muted;
            display: none;
        }
        #input {
            border-top: solid $primary-darken-2;
            padding: 0 1;
        }
        #status-bar {
            height: 1;
            background: $panel;
            color: $text-muted;
            padding: 0 1;
            dock: bottom;
        }
        """

        BINDINGS = [
            Binding("ctrl+c", "quit", "Exit"),
            Binding("ctrl+l", "clear", "Clear"),
            Binding("ctrl+n", "new_conversation", "New"),
            Binding("escape", "cancel_request", "Cancel"),
            Binding("f1", "help", "Help"),
        ]

        def __init__(
            self,
            client: Any | None = None,
            app_state: Any | None = None,
            **kwargs: Any,
        ) -> None:
            super().__init__(**kwargs)
            self._client = client
            self._app_state = app_state
            self._backend: TUIBackend | None = None
            self._current_task: asyncio.Task | None = None
            self._msg_count: int = 0
            self._last_role: str = ""
            self._tool_count: int = 0
            self._pending_tools: list[str] = []
            self._stream_text: str = ""
            self._model: str = "unknown"
            self._url: str = "local"
            self._spinner_index: int = 0
            self._spinner_timer: Any | None = None
            self._thinking_verb: str = "Thinking"

        def compose(self) -> ComposeResult:
            yield Header()
            with Vertical():
                yield RichLog(id="chat-log", markup=True, wrap=True)
                yield Static("", id="stream-buffer")
                yield Static("", id="tool-display")
                yield Static("", id="thinking")
                yield Input(
                    id="input",
                    placeholder="> Type a message... (/ for commands)",
                    suggester=SuggestFromList(SLASH_COMMANDS, case_sensitive=False),
                )
            yield Static("", id="status-bar")
            yield Footer()

        def _format_user_msg(self, _text: str = "", dt: Any | None = None) -> Any:
            from rich.text import Text

            ts = _format_timestamp(dt)
            line = Text()
            line.append("You", style="bold green")
            line.append(" " * max(1, 60 - 3 - len(ts)))
            line.append(ts, style="dim")
            return line

        def _format_bot_label(self, dt: Any | None = None) -> Any:
            from rich.text import Text

            ts = _format_timestamp(dt)
            line = Text()
            line.append("spare-paw", style="bold cyan")
            line.append(" " * max(1, 60 - 9 - len(ts)))
            line.append(ts, style="dim")
            return line

        def _write_divider(self) -> None:
            from rich.rule import Rule

            log = self.query_one("#chat-log", RichLog)
            log.write(Rule(characters="━", style="bright_black"))

        def _show_thinking(self) -> None:
            thinking = self.query_one("#thinking", Static)
            self._spinner_index = 0
            self._thinking_verb = random.choice(THINKING_VERBS)
            thinking.update(
                f"[dim italic]{SPINNER_FRAMES[0]} {self._thinking_verb}…[/dim italic]"
            )
            thinking.display = True
            self._spinner_timer = self.set_interval(
                0.12, self._advance_spinner
            )

        def _advance_spinner(self) -> None:
            self._spinner_index = (self._spinner_index + 1) % len(SPINNER_FRAMES)
            if self._spinner_index == 0:
                self._thinking_verb = random.choice(THINKING_VERBS)
            frame = SPINNER_FRAMES[self._spinner_index]
            self.query_one("#thinking", Static).update(
                f"[dim italic]{frame} {self._thinking_verb}…[/dim italic]"
            )

        def _hide_thinking(self) -> None:
            if self._spinner_timer is not None:
                self._spinner_timer.stop()
                self._spinner_timer = None
            thinking = self.query_one("#thinking", Static)
            thinking.display = False

        def _show_tool_display(self) -> None:
            from rich.panel import Panel

            if not self._pending_tools:
                return
            content = "\n".join(self._pending_tools)
            panel = Panel(
                content,
                title="tools",
                border_style="dim",
                expand=True,
                padding=(0, 1),
            )
            td = self.query_one("#tool-display", Static)
            td.update(panel)
            td.display = True

        def _hide_tool_display(self) -> None:
            td = self.query_one("#tool-display", Static)
            td.display = False
            td.update("")

        def _flush_tool_calls(self) -> None:
            self._hide_tool_display()
            self._pending_tools = []

        def _update_status_bar(self) -> None:
            connected = self._url != "local"
            dot = "[green]◉[/green]" if connected else "[red]◉[/red]"
            parts = [
                f"{dot} {self._url}",
                self._model,
                f"{self._msg_count} msgs",
                f"↑{self._tool_count} tools",
            ]
            self.post_message(UpdateStatus("  │  ".join(parts)))

        def on_append_log(self, message: AppendLog) -> None:
            self.query_one("#chat-log", RichLog).write(message.content)

        def on_update_status(self, message: UpdateStatus) -> None:
            self.query_one("#status-bar", Static).update(message.text)

        def on_stream_token(self, message: StreamToken) -> None:
            self._stream_text += message.token
            buf = self.query_one("#stream-buffer", Static)
            buf.display = True
            buf.update(self._stream_text)

        def on_stream_end(self, _message: StreamEnd) -> None:
            buf = self.query_one("#stream-buffer", Static)
            buf.display = False
            self._stream_text = ""
            buf.update("")

        def on_tool_call_event(self, message: ToolCallEvent) -> None:
            self._tool_count += 1
            icon = TOOL_ICONS.get(message.tool, "○")
            call_str = f"{icon} {message.tool}({message.args})"
            self._pending_tools.append(call_str)
            self._show_tool_display()
            self._update_status_bar()

        async def on_mount(self) -> None:
            self._backend = TUIBackend(self)

            if self._client:
                try:
                    info = await self._client.status()
                    self._model = info.get("model", "unknown")
                    self._url = self._client._url
                except Exception:
                    self._url = self._client._url

                self.query_one("#chat-log", RichLog).write(
                    "[dim]Connected to remote spare-paw instance[/dim]\n"
                )
                await self._load_history_remote()
            else:
                if self._app_state is not None:
                    self._model = self._app_state.config.get("models.main_agent", "unknown")

                self.query_one("#chat-log", RichLog).write(
                    "[dim]Running in standalone local mode[/dim]\n"
                )

                if self._app_state is not None:
                    from spare_paw.core.engine import start_queue_processor

                    self._app_state.backend = self._backend
                    start_queue_processor(self._app_state, self._backend)
                    await self._load_history_local()

            self._update_status_bar()
            self.query_one("#input", Input).focus()

        async def on_input_submitted(self, event: Input.Submitted) -> None:
            text = event.value.strip()
            if not text:
                return

            if text.endswith("\\"):
                return

            event.input.clear()
            log_widget = self.query_one("#chat-log", RichLog)

            self._msg_count += 1

            if self._last_role == "assistant":
                self._write_divider()

            log_widget.write(self._format_user_msg(text))
            log_widget.write(text)
            self._last_role = "user"

            self._update_status_bar()

            if text.lower() in ("/exit", "/quit"):
                self.exit()
                return

            if text.lower() == "/help":
                from rich.markdown import Markdown

                log_widget.write(Markdown(
                    "**Commands:** /help, /exit, /forget, /status, /model, /models, /image /path\n"
                    "**Keys:** Ctrl+C Exit, Ctrl+L Clear, Ctrl+N New, Esc Cancel, F1 Help"
                ))
                return

            if text.lower() == "/roles":
                from spare_paw.core.commands import cmd_roles
                result = await cmd_roles()
                log_widget.write(result)
                return

            if text.lower().startswith("/model"):
                await self._handle_model_command(text, log_widget)
                return

            if text.startswith("/image "):
                rest = text[7:].strip()
                if rest:
                    parts = rest.split(None, 1)
                    path = parts[0]
                    caption = parts[1] if len(parts) > 1 else "What do you see in this image?"
                    fpath = Path(path).expanduser()
                    if not fpath.exists():
                        log_widget.write(f"[red]File not found: {path}[/red]")
                        return
                    if fpath.suffix.lower() not in _IMAGE_EXTENSIONS:
                        log_widget.write(f"[red]Unsupported image format: {fpath.suffix}[/red]")
                        return
                    image_b64 = base64.b64encode(fpath.read_bytes()).decode("ascii")
                    if self._client:
                        self._current_task = asyncio.create_task(
                            self._send_remote(caption, image_b64=image_b64)
                        )
                    else:
                        await self._send_local_image(fpath.read_bytes(), caption)
                    return

            # /plan prefix — deep thinking mode
            plan_mode = False
            if text.lower().startswith("/plan "):
                plan_mode = True
                text = text[6:].strip()
                if not text:
                    log_widget.write("[dim]Usage: /plan <prompt>[/dim]")
                    return

            self._show_thinking()

            if self._client:
                self._current_task = asyncio.create_task(self._send_remote(text))
            else:
                await self._send_local(text, plan=plan_mode)

        async def _send_remote(self, text: str, image_b64: str | None = None) -> None:
            try:
                await self._client.send_message(text, image_b64=image_b64)
            except asyncio.CancelledError:
                self._hide_thinking()
                self.post_message(AppendLog("[dim]Cancelled[/dim]"))
                return
            except Exception as e:
                self._hide_thinking()
                self.post_message(AppendLog(f"[red]Error: {e}[/red]"))
                return

            try:
                async for event in self._client.stream_response():
                    etype = event.get("type", "")
                    if etype == "text":
                        self._hide_thinking()
                        self.post_message(StreamEnd())
                        self._flush_tool_calls()

                        log = self.query_one("#chat-log", RichLog)
                        log.write(self._format_bot_label())

                        from rich.markdown import Markdown

                        self.post_message(AppendLog(Markdown(event.get("text", ""))))
                        self.post_message(AppendLog(""))
                        self._msg_count += 1
                        self._last_role = "assistant"
                        self._update_status_bar()
                    elif etype == "tool_call":
                        tool = event.get("tool", "")
                        args = event.get("args", "")
                        self.post_message(ToolCallEvent(tool, args))
                    elif etype == "token":
                        self.post_message(StreamToken(event.get("token", "")))
            except asyncio.CancelledError:
                self._hide_thinking()
                self.post_message(AppendLog("[dim]Cancelled[/dim]"))
            except Exception as e:
                self._hide_thinking()
                self.post_message(AppendLog(f"[red]Connection error: {e}[/red]"))
                self._url = "disconnected"
                self._update_status_bar()
            finally:
                self._current_task = None

        def _render_history(self, messages: list[dict]) -> None:
            from datetime import datetime as dt

            from rich.markdown import Markdown

            log = self.query_one("#chat-log", RichLog)
            for msg in messages:
                role = msg.get("role", "")
                content = msg.get("content", "")
                created_at = msg.get("created_at", "")

                ts_dt = None
                if created_at:
                    try:
                        ts_dt = dt.fromisoformat(created_at)
                    except (ValueError, TypeError):
                        pass

                if role == "user" and self._last_role == "assistant":
                    self._write_divider()

                if role == "user":
                    log.write(self._format_user_msg(content, ts_dt))
                    log.write(content)
                else:
                    log.write(self._format_bot_label(ts_dt))
                    log.write(Markdown(content))

                self._msg_count += 1
                self._last_role = role

            if messages:
                self._write_divider()

        async def _load_history_remote(self) -> None:
            try:
                messages = await self._client.history(limit=10)
                self._render_history(messages)
            except Exception:
                pass

        async def _load_history_local(self) -> None:
            try:
                from spare_paw.context import get_or_create_conversation, recent

                conversation_id = await get_or_create_conversation()
                messages = await recent(conversation_id, limit=10)
                self._render_history(messages)
            except Exception:
                pass

        async def _send_local(self, text: str, plan: bool = False) -> None:
            from spare_paw.backend import IncomingMessage
            from spare_paw.core.engine import enqueue

            msg = IncomingMessage(text=text, plan=plan)
            await enqueue(msg)

        async def _send_local_image(self, image_bytes: bytes, caption: str) -> None:
            from spare_paw.backend import IncomingMessage
            from spare_paw.core.engine import enqueue

            msg = IncomingMessage(image_bytes=image_bytes, caption=caption)
            await enqueue(msg)

        def action_clear(self) -> None:
            self.query_one("#chat-log", RichLog).clear()

        def action_cancel_request(self) -> None:
            if self._current_task and not self._current_task.done():
                self._current_task.cancel()
                self.query_one("#chat-log", RichLog).write("[dim]Cancelled[/dim]")

        def action_new_conversation(self) -> None:
            self._msg_count = 0
            self._tool_count = 0
            self._last_role = ""
            self._pending_tools = []
            self._flush_tool_calls()
            self.query_one("#chat-log", RichLog).clear()
            self._update_status_bar()
            if self._client:
                self._current_task = asyncio.create_task(self._send_remote("/forget"))
            else:
                asyncio.create_task(self._send_local("/forget"))

        def action_help(self) -> None:
            from rich.markdown import Markdown

            self.query_one("#chat-log", RichLog).write(Markdown(
                "**Commands:** /help, /exit, /forget, /status, /model, /models, /image /path\n"
                "**Keys:** Ctrl+C Exit, Ctrl+L Clear, Ctrl+N New, Esc Cancel, F1 Help"
            ))

        async def _handle_model_command(self, text: str, log_widget: Any) -> None:
            """Handle /model and /models commands locally."""
            from spare_paw.core.commands import cmd_model, cmd_models

            stripped = text.strip()

            # /models [filter]
            if stripped.lower().startswith("/models"):
                query = stripped[7:].strip() or None
                result = await cmd_models(self._app_state, query)
                log_widget.write(result)
                return

            # /model [role] [model_id]
            parts = stripped.split()
            args = parts[1:] if len(parts) > 1 else None
            result = await cmd_model(self._app_state, args)
            log_widget.write(result)
            if args and self._app_state:
                self._model = self._app_state.config.get("models.main_agent", "unknown")
                self._update_status_bar()


async def run_tui(
    client: Any | None = None, app_state: Any | None = None
) -> None:
    """Launch the TUI app."""
    if not HAS_TEXTUAL:
        raise ImportError(
            "textual is required for TUI mode: pip install spare-paw[tui]"
        )
    app = SparePawTUI(client=client, app_state=app_state)
    await app.run_async()
