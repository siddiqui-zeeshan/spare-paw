from __future__ import annotations

import pytest

from spare_paw.cli.client import ConnectionState, RemoteClient


@pytest.mark.asyncio
async def test_initial_state_is_connected_false():
    rc = RemoteClient("http://nowhere:9999", secret="")
    assert rc.connection_state == ConnectionState.DISCONNECTED


@pytest.mark.asyncio
async def test_state_transitions_recorded(monkeypatch):
    rc = RemoteClient("http://nowhere:9999", secret="")
    states: list[ConnectionState] = []
    rc.subscribe_state(lambda s: states.append(s))
    rc._set_state(ConnectionState.RECONNECTING)
    rc._set_state(ConnectionState.CONNECTED)
    assert states == [ConnectionState.RECONNECTING, ConnectionState.CONNECTED]


def test_backoff_sequence_capped():
    rc = RemoteClient("http://nowhere:9999")
    delays = [rc._next_backoff() for _ in range(8)]
    assert delays[:5] == [1.0, 2.0, 4.0, 8.0, 16.0]
    assert all(d <= 30.0 for d in delays)


def test_backoff_resets_on_connected():
    rc = RemoteClient("http://nowhere:9999")
    rc._next_backoff()
    rc._next_backoff()
    rc._next_backoff()
    rc._set_state(ConnectionState.CONNECTED)
    assert rc._next_backoff() == 1.0
