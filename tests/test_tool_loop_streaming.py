from __future__ import annotations

import pytest

from spare_paw.router.openrouter import StreamChunk
from spare_paw.router.tool_loop import _stream_and_assemble, run_tool_loop


class _FakeClient:
    def __init__(self, chunks: list[StreamChunk]) -> None:
        self._chunks = chunks

    async def chat_stream(self, messages, model, tools):
        for c in self._chunks:
            yield c


@pytest.mark.asyncio
async def test_assemble_text_only():
    chunks = [
        StreamChunk(kind="text_delta", content="Hel"),
        StreamChunk(kind="text_delta", content="lo "),
        StreamChunk(kind="text_delta", content="world"),
        StreamChunk(kind="done", finish_reason="stop",
                    usage={"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}),
    ]
    tokens: list[str] = []
    msg, usage = await _stream_and_assemble(_FakeClient(chunks), [], "m", None, tokens.append)
    assert msg == {"role": "assistant", "content": "Hello world"}
    assert tokens == ["Hel", "lo ", "world"]
    assert usage == {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}


@pytest.mark.asyncio
async def test_assemble_tool_calls_from_deltas():
    chunks = [
        StreamChunk(kind="tool_call_delta", tool_index=0, tool_id="call_1",
                    tool_name="read_file"),
        StreamChunk(kind="tool_call_delta", tool_index=0,
                    arguments_fragment='{"path": "'),
        StreamChunk(kind="tool_call_delta", tool_index=0,
                    arguments_fragment='foo.py"}'),
        StreamChunk(kind="done", finish_reason="tool_calls"),
    ]
    msg, _ = await _stream_and_assemble(_FakeClient(chunks), [], "m", None, None)
    assert msg["role"] == "assistant"
    assert msg["content"] is None
    assert msg["tool_calls"] == [{
        "id": "call_1",
        "type": "function",
        "function": {"name": "read_file", "arguments": '{"path": "foo.py"}'},
    }]


@pytest.mark.asyncio
async def test_assemble_text_then_tool_call():
    chunks = [
        StreamChunk(kind="text_delta", content="Let me check."),
        StreamChunk(kind="tool_call_delta", tool_index=0, tool_id="c1",
                    tool_name="shell", arguments_fragment='{"cmd":"ls"}'),
        StreamChunk(kind="done", finish_reason="tool_calls"),
    ]
    tokens: list[str] = []
    msg, _ = await _stream_and_assemble(_FakeClient(chunks), [], "m", None, tokens.append)
    assert tokens == ["Let me check."]
    assert msg["content"] == "Let me check."
    assert msg["tool_calls"][0]["function"]["name"] == "shell"


class _StreamingFakeClient:
    """Fake client that responds via chat_stream only; chat() should never be called."""
    def __init__(self, responses: list[list[StreamChunk]]) -> None:
        self._responses = list(responses)

    async def chat_stream(self, messages, model, tools):
        chunks = self._responses.pop(0)
        for c in chunks:
            yield c

    async def chat(self, *a, **kw):
        raise AssertionError("run_tool_loop must not call chat() — it should stream")


@pytest.mark.asyncio
async def test_run_tool_loop_emits_tokens_live_not_retroactively():
    """Verify on_token fires during streaming, not by splitting final content after."""
    chunks = [
        StreamChunk(kind="text_delta", content="one "),
        StreamChunk(kind="text_delta", content="two "),
        StreamChunk(kind="text_delta", content="three"),
        StreamChunk(kind="done", finish_reason="stop",
                    usage={"prompt_tokens": 1, "completion_tokens": 3, "total_tokens": 4}),
    ]

    class _EmptyRegistry:
        def get_schemas(self): return []

    tokens: list[str] = []
    result = await run_tool_loop(
        client=_StreamingFakeClient([chunks]),
        messages=[{"role": "user", "content": "hi"}],
        model="m",
        tools=[],
        tool_registry=_EmptyRegistry(),
        on_token=tokens.append,
    )

    assert tokens == ["one ", "two ", "three"]
    assert result == "one two three"
