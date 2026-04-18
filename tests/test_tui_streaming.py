from __future__ import annotations

import asyncio
import pytest

from spare_paw.tui.streaming import StreamSession


@pytest.mark.asyncio
async def test_tokens_flushed_after_coalesce_window():
    flushed: list[str] = []
    session = StreamSession(
        on_flush=lambda text: flushed.append(text),
        coalesce_ms=10,
    )
    session.append("hel")
    session.append("lo ")
    session.append("world")
    await asyncio.sleep(0.03)
    assert "".join(flushed) == "hello world"
    assert len(flushed) >= 1


@pytest.mark.asyncio
async def test_finalize_flushes_pending_tokens():
    flushed: list[str] = []
    session = StreamSession(on_flush=flushed.append, coalesce_ms=1000)
    session.append("hi")
    session.finalize()
    assert "".join(flushed) == "hi"


@pytest.mark.asyncio
async def test_cancel_stops_future_flushes():
    flushed: list[str] = []
    session = StreamSession(on_flush=flushed.append, coalesce_ms=5)
    session.append("part1 ")
    session.cancel()
    session.append("part2")
    await asyncio.sleep(0.02)
    assert "part2" not in "".join(flushed)
