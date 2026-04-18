from spare_paw.tui.events import (
    AppendLog, UpdateStatus, StreamToken, StreamEnd,
    ToolCallStart, ToolCallEnd, AppendError, ConnectionStateChanged,
)


def test_events_importable():
    assert AppendLog is not None
    assert UpdateStatus is not None
    assert StreamToken is not None
    assert StreamEnd is not None
    assert ToolCallStart is not None
    assert ToolCallEnd is not None
    assert AppendError is not None
    assert ConnectionStateChanged is not None


def test_stream_token_carries_text():
    evt = StreamToken("hello")
    assert evt.token == "hello"


def test_tool_call_start_payload():
    evt = ToolCallStart(call_id="c1", tool="read_file", args={"path": "foo"})
    assert evt.call_id == "c1"
    assert evt.tool == "read_file"
    assert evt.args == {"path": "foo"}


def test_tool_call_end_payload():
    evt = ToolCallEnd(call_id="c1", success=True, duration_ms=320, preview="...")
    assert evt.call_id == "c1"
    assert evt.success is True
    assert evt.duration_ms == 320
    assert evt.preview == "..."


def test_connection_state_event():
    evt = ConnectionStateChanged(state="reconnecting", detail="SSE closed")
    assert evt.state == "reconnecting"
    assert evt.detail == "SSE closed"
