"""Helpers for converting old-style OpenRouter response dicts into StreamChunk
sequences used by fake streaming clients in tests.

Task 3 of the TUI 10/10 plan replaced run_tool_loop's non-streaming
``client.chat()`` call with ``client.chat_stream()``. Pre-existing tests
were written against the non-streaming shape — these helpers convert
their response fixtures into equivalent StreamChunk sequences so the
behavioural assertions still exercise run_tool_loop end-to-end.
"""

from __future__ import annotations

from typing import Any

from spare_paw.router.openrouter import StreamChunk


def response_to_chunks(
    content: str | None = None,
    tool_calls: list[dict[str, Any]] | None = None,
    usage: dict[str, int] | None = None,
) -> list[StreamChunk]:
    """Convert a non-streaming response (content + tool_calls + usage) into
    an equivalent list of ``StreamChunk`` deltas ending with ``done``.

    - ``content`` becomes a single ``text_delta``.
    - Each tool call becomes a single ``tool_call_delta`` carrying id, name,
      and a full arguments fragment.
    - A final ``done`` chunk carries the finish_reason (tool_calls if any
      tool calls were present, else stop) and the usage dict.
    """
    chunks: list[StreamChunk] = []
    if content:
        chunks.append(StreamChunk(kind="text_delta", content=content))
    for i, tc in enumerate(tool_calls or []):
        fn = tc.get("function", {})
        chunks.append(
            StreamChunk(
                kind="tool_call_delta",
                tool_index=i,
                tool_id=tc.get("id"),
                tool_name=fn.get("name"),
                arguments_fragment=fn.get("arguments"),
            )
        )
    chunks.append(
        StreamChunk(
            kind="done",
            finish_reason="tool_calls" if tool_calls else "stop",
            usage=usage
            or {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        )
    )
    return chunks


def response_dict_to_chunks(response: dict[str, Any]) -> list[StreamChunk]:
    """Convert a full OpenAI-format response dict (``{"choices": [...], "usage": ...}``)
    into a StreamChunk sequence. Shortcut for call sites that already build
    response dicts with nested ``message.content`` / ``message.tool_calls``.
    """
    choices = response.get("choices") or []
    message = choices[0].get("message", {}) if choices else {}
    return response_to_chunks(
        content=message.get("content"),
        tool_calls=message.get("tool_calls"),
        usage=response.get("usage"),
    )


class FakeStreamingClient:
    """Fake OpenRouterClient that only supports ``chat_stream``.

    Pass a list of StreamChunk sequences; each call to ``chat_stream``
    pops and yields the next sequence.

    ``chat()`` is provided as a fallback because run_tool_loop calls it
    after exhausting ``max_iterations`` to request a final text summary.
    """

    def __init__(self, responses: list[list[StreamChunk]]) -> None:
        self._responses = list(responses)
        self.stream_call_count = 0
        self.chat_call_count = 0
        self.last_messages: list[dict[str, Any]] | None = None
        self.last_tools: list[dict[str, Any]] | None = None

    async def chat_stream(self, messages, model, tools=None):
        self.stream_call_count += 1
        self.last_messages = list(messages)
        self.last_tools = tools
        chunks = self._responses.pop(0)
        for c in chunks:
            yield c

    async def chat(self, messages, model, tools=None):
        self.chat_call_count += 1
        return {
            "choices": [
                {"message": {"role": "assistant", "content": "fallback"}}
            ],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
        }
