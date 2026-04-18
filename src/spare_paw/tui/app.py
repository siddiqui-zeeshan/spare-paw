"""TUI app shell — wires backend, commands, streaming, and widgets."""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.widgets import Footer, Header

from spare_paw.tui.backend import TUIBackend
from spare_paw.tui.commands import SlashCommandRouter
from spare_paw.tui.events import (
    AppendError,
    AppendLog,
    ConnectionStateChanged,
    StreamEnd,
    StreamToken,
    ToolCallEnd,
    ToolCallStart,
    UpdateStatus,
)
from spare_paw.tui.streaming import StreamSession
from spare_paw.tui.theme import APP_CSS, STREAM_COALESCE_MS
from spare_paw.tui.widgets.chat_log import ChatLog
from spare_paw.tui.widgets.composer import Composer, ComposerSubmitted
from spare_paw.tui.widgets.message_view import MessageView
from spare_paw.tui.widgets.status_bar import StatusBar


class SparePawTUI(App):
    TITLE = "spare-paw"
    CSS = APP_CSS

    BINDINGS = [
        Binding("ctrl+c", "quit", "Exit"),
        Binding("ctrl+l", "clear", "Clear"),
        Binding("ctrl+n", "new_conversation", "New"),
        Binding("escape", "cancel_request", "Cancel"),
        Binding("ctrl+f", "find", "Find"),
        Binding("ctrl+y", "copy_last", "Copy last"),
        Binding("pageup", "scroll_page_up", "PgUp"),
        Binding("pagedown", "scroll_page_down", "PgDn"),
        Binding("home", "scroll_home", "Top"),
        Binding("end", "scroll_end", "Bottom"),
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
        self._commands = SlashCommandRouter(app_state=app_state)
        self._current_task: asyncio.Task | None = None
        self._msg_count = 0
        self._tool_count = 0
        self._model = "unknown"
        self._url = "local"
        self._connection = "local"
        self._stream: StreamSession | None = None

    def compose(self) -> ComposeResult:
        yield Header()
        yield ChatLog(id="chat-log")
        yield Composer(id="composer")
        yield StatusBar(id="status-bar")
        yield Footer()

    async def on_mount(self) -> None:
        self._backend = TUIBackend(self)
        self._update_status()

        if self._client is not None:
            try:
                info = await self._client.status()
                self._model = info.get("model", "unknown")
                self._url = self._client._url
                self._connection = "connected"
            except Exception as exc:
                self._url = self._client._url
                self._connection = "disconnected"
                self.query_one(ChatLog).append_error(f"Could not reach remote: {exc}")
            await self._load_history_remote()
        else:
            if self._app_state is not None:
                self._model = self._app_state.config.get("models.main_agent", "unknown")
                from spare_paw.core.engine import start_queue_processor
                self._app_state.backend = self._backend
                start_queue_processor(self._app_state, self._backend)
            self._connection = "local"
            await self._load_history_local()

        self._update_status()
        self.query_one(Composer).focus()

    def on_composer_submitted(self, msg: ComposerSubmitted) -> None:
        self.run_worker(self._handle_input(msg.text), exclusive=False)

    async def _handle_input(self, text: str) -> None:
        log = self.query_one(ChatLog)
        log.mount_turn(MessageView(role="user", initial_text=text))
        self._msg_count += 1
        self._update_status()

        result = await self._commands.dispatch(text)

        if result.kind == "quit":
            self.exit()
            return
        if result.kind == "find":
            hits = log.search(result.content)
            if not hits:
                log.append_error(f"No matches for '{result.content}'")
            else:
                hits[-1].scroll_visible()
                log.append_error(f"{len(hits)} matches — showing last")
            return
        if result.kind == "text":
            log.mount_turn(MessageView(role="assistant", initial_text=result.content, historical=True))
            return
        if result.kind == "forget":
            self._clear_conversation()
            if self._client:
                await self._client.send_message("/forget")
            else:
                await self._send_local("/forget")
            return
        if result.kind == "send":
            await self._start_assistant_turn(result.text, plan=result.plan)
            return
        if result.kind == "send_image":
            await self._start_assistant_turn(
                result.text, image_b64=result.image_b64, plan=False,
            )
            return

    async def _start_assistant_turn(
        self, text: str, plan: bool = False, image_b64: str | None = None,
    ) -> None:
        log = self.query_one(ChatLog)
        turn = MessageView(role="assistant")
        log.mount_turn(turn)
        self._stream = StreamSession(
            on_flush=turn.append_stream,
            coalesce_ms=STREAM_COALESCE_MS,
        )

        if self._client is not None:
            self._current_task = asyncio.create_task(
                self._run_remote(text, image_b64=image_b64)
            )
        else:
            self._current_task = asyncio.create_task(
                self._send_local(text, plan=plan, image_b64=image_b64)
            )

    async def _run_remote(self, text: str, image_b64: str | None = None) -> None:
        try:
            await self._client.send_message(text, image_b64=image_b64)
            async for event in self._client.stream_response():
                etype = event.get("type")
                if etype == "token":
                    self.post_message(StreamToken(event.get("token", "")))
                elif etype == "tool_call":
                    self.post_message(ToolCallStart(
                        call_id=event.get("call_id", event.get("tool", "")),
                        tool=event.get("tool", ""),
                        args=event.get("args", {}) or {},
                    ))
                elif etype == "tool_end":
                    self.post_message(ToolCallEnd(
                        call_id=event.get("call_id", event.get("tool", "")),
                        success=event.get("success", True),
                        duration_ms=event.get("duration_ms", 0),
                        preview=event.get("preview", ""),
                    ))
                elif etype == "text":
                    final = event.get("text", "")
                    if final:
                        self.post_message(StreamToken(final))
                    self.post_message(StreamEnd())
                    return
        except asyncio.CancelledError:
            self.post_message(StreamEnd())
            raise
        except Exception as exc:
            self.post_message(AppendError(f"Connection error: {exc}"))
            self._connection = "disconnected"
            self._update_status()
        finally:
            self._current_task = None

    async def _send_local(
        self, text: str, plan: bool = False, image_b64: str | None = None,
    ) -> None:
        from spare_paw.backend import IncomingMessage
        from spare_paw.core.engine import enqueue

        if image_b64 is not None:
            import base64
            image_bytes = base64.b64decode(image_b64)
            await enqueue(IncomingMessage(image_bytes=image_bytes, caption=text))
        else:
            await enqueue(IncomingMessage(text=text, plan=plan))

    async def _load_history_local(self) -> None:
        try:
            from spare_paw.context import get_or_create_conversation, recent
            cid = await get_or_create_conversation()
            messages = await recent(cid, limit=10)
            self._render_history(messages)
        except Exception as exc:
            self.query_one(ChatLog).append_error(f"Could not load history: {exc}")

    async def _load_history_remote(self) -> None:
        if self._client is None:
            return
        try:
            messages = await self._client.history(limit=10)
            self._render_history(messages)
        except Exception as exc:
            self.query_one(ChatLog).append_error(f"Could not load history: {exc}")

    def _render_history(self, messages: list[dict]) -> None:
        log = self.query_one(ChatLog)
        for msg in messages:
            role = msg.get("role", "assistant")
            content = msg.get("content", "")
            created_at = msg.get("created_at")
            ts = None
            if created_at:
                try:
                    ts = datetime.fromisoformat(created_at)
                except (ValueError, TypeError):
                    ts = None
            log.mount_turn(MessageView(
                role=role if role in ("user", "assistant") else "assistant",
                initial_text=content,
                timestamp=ts,
                historical=True,
            ))
            self._msg_count += 1

    def on_stream_token(self, msg: StreamToken) -> None:
        if self._stream is not None:
            self._stream.append(msg.token)
        else:
            # Fallback: append directly to active assistant if no StreamSession active
            turn = self.query_one(ChatLog).active_assistant()
            if turn is not None:
                turn.append_stream(msg.token)

    def on_stream_end(self, _msg: StreamEnd) -> None:
        log = self.query_one(ChatLog)
        turn = log.active_assistant()
        if self._stream is not None:
            self._stream.finalize()
            self._stream = None
        if turn is not None:
            turn.finalize()
            self._msg_count += 1
        self._update_status()

    def on_tool_call_start(self, msg: ToolCallStart) -> None:
        log = self.query_one(ChatLog)
        turn = log.active_assistant() or log.mount_turn(MessageView(role="assistant"))
        turn.add_tool_call(call_id=msg.call_id, tool=msg.tool, args=msg.args)
        self._tool_count += 1
        self._update_status()

    def on_tool_call_end(self, msg: ToolCallEnd) -> None:
        log = self.query_one(ChatLog)
        turn = log.active_assistant()
        if turn is not None:
            turn.complete_tool_call(
                call_id=msg.call_id,
                success=msg.success,
                duration_ms=msg.duration_ms,
                preview=msg.preview,
            )

    def on_append_log(self, msg: AppendLog) -> None:
        from textual.widgets import Static
        self.query_one(ChatLog).mount(Static(msg.content))

    def on_append_error(self, msg: AppendError) -> None:
        self.query_one(ChatLog).append_error(msg.text)

    def on_update_status(self, msg: UpdateStatus) -> None:
        bar = self.query_one(StatusBar)
        bar.update(msg.text)

    def on_connection_state_changed(self, msg: ConnectionStateChanged) -> None:
        self._connection = msg.state
        self._update_status()
        if msg.detail:
            self.query_one(ChatLog).append_error(msg.detail)

    def _update_status(self) -> None:
        self.query_one(StatusBar).set_state(
            connection=self._connection,
            url=self._url,
            model=self._model,
            msg_count=self._msg_count,
            tool_count=self._tool_count,
        )

    def _clear_conversation(self) -> None:
        self._msg_count = 0
        self._tool_count = 0
        self.query_one(ChatLog).remove_children()
        self._update_status()

    # Actions
    def action_clear(self) -> None:
        self.query_one(ChatLog).remove_children()

    def action_cancel_request(self) -> None:
        if self._current_task and not self._current_task.done():
            self._current_task.cancel()
            turn = self.query_one(ChatLog).active_assistant()
            if turn is not None:
                turn.mark_cancelled()
            if self._stream is not None:
                self._stream.cancel()
                self._stream = None

    def action_new_conversation(self) -> None:
        self._clear_conversation()
        if self._client:
            self._current_task = asyncio.create_task(self._client.send_message("/forget"))
        else:
            asyncio.create_task(self._send_local("/forget"))

    def action_help(self) -> None:
        from spare_paw.tui.commands import HELP_TEXT
        self.query_one(ChatLog).mount_turn(
            MessageView(role="assistant", initial_text=HELP_TEXT, historical=True)
        )

    def action_scroll_page_up(self) -> None:
        self.query_one(ChatLog).scroll_page_up()

    def action_scroll_page_down(self) -> None:
        self.query_one(ChatLog).scroll_page_down()

    def action_scroll_home(self) -> None:
        self.query_one(ChatLog).scroll_home()

    def action_scroll_end(self) -> None:
        self.query_one(ChatLog).scroll_end()

    def action_find(self) -> None:
        # MVP placeholder — /find is routed via SlashCommandRouter in Task 16.
        pass

    def action_copy_last(self) -> None:
        views = list(self.query_one(ChatLog).query(MessageView))
        for view in reversed(views):
            if view.role == "assistant" and view.live_text:
                _copy_to_clipboard(view.live_text)
                self.query_one(ChatLog).append_error(
                    f"Copied {len(view.live_text)} chars"
                )
                return


def _copy_to_clipboard(text: str) -> None:
    """Best-effort clipboard copy. Uses pbcopy on macOS, xclip/wl-copy on Linux.

    Silently does nothing if no clipboard tool is available.
    """
    import shutil
    import subprocess

    for cmd in (["pbcopy"], ["wl-copy"], ["xclip", "-selection", "clipboard"]):
        if shutil.which(cmd[0]):
            try:
                subprocess.run(cmd, input=text.encode(), check=True)
                return
            except subprocess.SubprocessError:
                continue


async def run_tui(client=None, app_state=None) -> None:
    app = SparePawTUI(client=client, app_state=app_state)
    await app.run_async()
