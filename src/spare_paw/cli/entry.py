"""Entry point for the ``spare-paw chat`` command."""

from __future__ import annotations

import asyncio
import sys

from rich.console import Console


async def _cleanup_local(app_state: object, backend: object | None = None) -> None:
    """Clean up local engine resources."""
    from spare_paw.core import engine as engine_mod

    # Cancel queue processor task
    if engine_mod._queue_task is not None:
        engine_mod._queue_task.cancel()
        try:
            await engine_mod._queue_task
        except asyncio.CancelledError:
            pass

    # Stop backend
    if backend is not None and hasattr(backend, "stop"):
        await backend.stop()

    # Close router client
    if hasattr(app_state, "router_client") and app_state.router_client is not None:
        await app_state.router_client.close()

    # Shutdown executor
    if hasattr(app_state, "executor") and app_state.executor is not None:
        app_state.executor.shutdown(wait=False)


async def _async_chat() -> None:
    from spare_paw.config import config

    config.load()

    remote_url = config.get("remote.url")
    client = None
    app_state = None

    if remote_url:
        from spare_paw.cli.client import RemoteClient

        client = RemoteClient(
            url=remote_url,
            secret=config.get("remote.secret", ""),
        )
        if not await client.health():
            console = Console()
            console.print(
                f"[red]Cannot connect to {remote_url}. Is the server running?[/red]"
            )
            await client.close()
            return
    else:
        from spare_paw.gateway import init_app_state

        try:
            app_state = await init_app_state()
        except RuntimeError as e:
            Console().print(f"[red]{e}[/red]")
            return

    try:
        # Pipe mode: non-interactive stdin
        if not sys.stdin.isatty():
            from spare_paw.cli.pipe import run_pipe

            await run_pipe(client=client, app_state=app_state)
            return

        # TUI mode (default interactive)
        from spare_paw.tui.app import run_tui

        await run_tui(client=client, app_state=app_state)
    finally:
        if client is not None:
            await client.close()
        if app_state is not None:
            await _cleanup_local(app_state)


def run_chat() -> None:
    """Synchronous entry point for ``spare-paw chat``."""
    asyncio.run(_async_chat())
