"""StreamSession: token-coalescing buffer with scheduled flush.

Streamed tokens arrive at ~ms granularity. Re-rendering a widget on every
token is expensive. StreamSession buffers tokens and flushes them in a
coalescing window (default 16ms) on the asyncio event loop.
"""

from __future__ import annotations

import asyncio
from typing import Callable


class StreamSession:
    def __init__(
        self,
        on_flush: Callable[[str], None],
        coalesce_ms: int = 16,
    ) -> None:
        self._on_flush = on_flush
        self._coalesce_ms = coalesce_ms
        self._buffer: list[str] = []
        self._pending: asyncio.TimerHandle | None = None
        self._cancelled = False

    def append(self, token: str) -> None:
        if self._cancelled or not token:
            return
        self._buffer.append(token)
        if self._pending is None:
            loop = asyncio.get_event_loop()
            self._pending = loop.call_later(
                self._coalesce_ms / 1000.0, self._flush,
            )

    def _flush(self) -> None:
        self._pending = None
        if self._cancelled:
            return
        if not self._buffer:
            return
        chunk = "".join(self._buffer)
        self._buffer.clear()
        try:
            self._on_flush(chunk)
        except Exception:
            pass

    def finalize(self) -> None:
        """Force-flush pending tokens immediately."""
        if self._pending is not None:
            self._pending.cancel()
            self._pending = None
        self._flush()

    def cancel(self) -> None:
        """Stop flushing; future tokens are dropped."""
        self._cancelled = True
        if self._pending is not None:
            self._pending.cancel()
            self._pending = None
        self._buffer.clear()
