from __future__ import annotations

import asyncio
import json
from unittest.mock import MagicMock

import pytest

from spare_paw.router.openrouter import OpenRouterClient, StreamChunk


class _FakeLineIter:
    def __init__(self, lines: list[bytes]) -> None:
        self._lines = lines
    def __aiter__(self):
        return self._iter()
    async def _iter(self):
        for line in self._lines:
            yield line


class _FakeResp:
    def __init__(self, lines: list[bytes], status: int = 200) -> None:
        self.content = _FakeLineIter(lines)
        self.status = status
    async def __aenter__(self): return self
    async def __aexit__(self, *a): return False
    async def text(self): return ""


def _sse(payload: dict) -> bytes:
    return ("data: " + json.dumps(payload) + "\n").encode()


@pytest.mark.asyncio
async def test_chat_stream_yields_text_tool_and_done_chunks():
    lines = [
        _sse({"choices": [{"delta": {"content": "Hello"}}]}),
        _sse({"choices": [{"delta": {"content": " world"}}]}),
        _sse({"choices": [{"delta": {"tool_calls": [
            {"index": 0, "id": "call_1", "function": {"name": "read_file"}}
        ]}}]}),
        _sse({"choices": [{"delta": {"tool_calls": [
            {"index": 0, "function": {"arguments": "{\"path\": \""}}
        ]}}]}),
        _sse({"choices": [{"delta": {"tool_calls": [
            {"index": 0, "function": {"arguments": "foo.py\"}"}}
        ]}}]}),
        _sse({"choices": [{"finish_reason": "tool_calls"}],
              "usage": {"prompt_tokens": 5, "completion_tokens": 9, "total_tokens": 14}}),
        b"data: [DONE]\n",
    ]
    sem = asyncio.Semaphore(1)
    client = OpenRouterClient("key", sem)
    client._get_session = MagicMock(return_value=MagicMock(post=MagicMock(return_value=_FakeResp(lines))))

    chunks: list[StreamChunk] = []
    async for ch in client.chat_stream([{"role": "user", "content": "hi"}], "m"):
        chunks.append(ch)

    kinds = [c.kind for c in chunks]
    assert kinds == ["text_delta", "text_delta", "tool_call_delta",
                     "tool_call_delta", "tool_call_delta", "done"]
    assert chunks[0].content == "Hello"
    assert chunks[1].content == " world"
    assert chunks[2].tool_index == 0
    assert chunks[2].tool_id == "call_1"
    assert chunks[2].tool_name == "read_file"
    assert chunks[3].arguments_fragment == '{"path": "'
    assert chunks[4].arguments_fragment == 'foo.py"}'
    assert chunks[5].finish_reason == "tool_calls"
    assert chunks[5].usage == {"prompt_tokens": 5, "completion_tokens": 9, "total_tokens": 14}
