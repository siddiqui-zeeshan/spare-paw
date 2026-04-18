from __future__ import annotations

import pytest

from spare_paw.router.openrouter import StreamChunk
from spare_paw.router.tool_loop import _stream_and_assemble


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
