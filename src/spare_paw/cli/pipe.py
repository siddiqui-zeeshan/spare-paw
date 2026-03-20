"""Non-interactive pipe mode for CLI."""

from __future__ import annotations

import sys
from typing import Any

from rich.console import Console


async def run_pipe(
    client: Any | None = None,
    app_state: Any | None = None,
) -> None:
    """Read stdin, process once, print response, exit."""
    text = sys.stdin.read().strip()
    if not text:
        return

    console = Console(stderr=True)

    if client is not None:
        # Remote mode
        try:
            await client.send_message(text)
        except PermissionError as e:
            console.print(f"[red]{e}[/red]")
            return

        while True:
            try:
                events = await client.poll(timeout=60)
            except Exception as e:
                console.print(f"[red]Connection error: {e}[/red]")
                return

            for event in events:
                if event.get("type") == "text":
                    print(event.get("text", ""))
                    return
    else:
        # Standalone mode
        from spare_paw.backend import IncomingMessage
        from spare_paw.core.engine import process_message

        class _CollectBackend:
            """Minimal backend that collects text output."""
            def __init__(self) -> None:
                self.output: list[str] = []

            async def send_text(self, text: str) -> None:
                self.output.append(text)

            async def send_file(self, path: str, caption: str = "") -> None:
                pass

            async def send_typing(self) -> None:
                pass

            async def send_notification(
                self, text: str, actions: list[dict] | None = None
            ) -> None:
                pass

            def on_tool_event(self, event: Any) -> None:
                pass

            def on_token(self, token: str) -> None:
                pass

            async def start(self) -> None:
                pass

            async def stop(self) -> None:
                pass

        backend = _CollectBackend()
        msg = IncomingMessage(text=text)
        await process_message(app_state, msg, backend)

        for chunk in backend.output:
            print(chunk)
