"""Tool-use execution loop for multi-turn function calling."""

from __future__ import annotations

import json
import logging
from concurrent.futures import ProcessPoolExecutor
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from spare_paw.router.openrouter import OpenRouterClient
    from spare_paw.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


async def run_tool_loop(
    client: "OpenRouterClient",
    messages: list[dict[str, Any]],
    model: str,
    tools: list[dict[str, Any]],
    tool_registry: "ToolRegistry",
    max_iterations: int = 20,
    executor: ProcessPoolExecutor | None = None,
    track_usage: bool = False,
) -> str | tuple[str, dict[str, int]]:
    """Run the model in a tool-calling loop until it produces a final text response.

    Each iteration:
      1. Call the model with the current messages and tool definitions.
      2. If the model returns tool_calls, execute each one and append results.
      3. If the model returns plain text (no tool_calls), return it.
      4. Repeat up to ``max_iterations`` times.

    Tool execution errors are caught and returned to the model as error
    strings so it can attempt recovery.

    Args:
        client: OpenRouter API client.
        messages: Conversation messages (mutated in place with tool calls/results).
        model: Model identifier for OpenRouter.
        tools: Tool JSON schemas in OpenAI function-calling format.
        tool_registry: Registry that resolves and executes tool functions.
        max_iterations: Safety cap on tool-call rounds (default 20).
        executor: Optional ProcessPoolExecutor for blocking tool functions.
        track_usage: If True, accumulate token usage across all API calls and
            return ``(response_text, usage_dict)`` instead of just the text.

    Returns:
        The final text content from the model when ``track_usage`` is False.
        A tuple of ``(text, usage_dict)`` when ``track_usage`` is True, where
        *usage_dict* has keys ``prompt_tokens``, ``completion_tokens``, and
        ``total_tokens``.
    """
    total_usage: dict[str, int] = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }

    def _accumulate_usage(resp: dict[str, Any]) -> None:
        usage = resp.get("usage", {})
        for key in total_usage:
            total_usage[key] += usage.get(key, 0)

    def _maybe_with_usage(text: str) -> str | tuple[str, dict[str, int]]:
        if track_usage:
            return (text, total_usage)
        return text

    for iteration in range(1, max_iterations + 1):
        response = await client.chat(messages, model, tools)
        _accumulate_usage(response)

        choice = response["choices"][0]
        assistant_message = choice["message"]

        # Check for tool calls
        tool_calls = assistant_message.get("tool_calls")

        if not tool_calls:
            # No tool calls — return the text content
            content = assistant_message.get("content", "")
            return _maybe_with_usage(content or "")

        # Append the assistant message (with tool_calls) to the conversation
        messages.append(assistant_message)

        # Execute each tool call, deferring stop signals until all are done
        stop_reply: str | None = None
        for tool_call in tool_calls:
            call_id = tool_call["id"]
            function = tool_call["function"]
            name = function["name"]
            raw_args = function.get("arguments", "{}")

            # Parse arguments — models sometimes return a string
            try:
                args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
            except json.JSONDecodeError:
                args = {}
                logger.warning(
                    "Iteration %d: failed to parse args for tool %s: %s",
                    iteration,
                    name,
                    raw_args,
                )

            # Execute the tool, catching any exception
            try:
                result = await tool_registry.execute(name, args, executor)
                result_str = str(result) if not isinstance(result, str) else result
                logger.info(
                    "Iteration %d: tool %s executed successfully", iteration, name
                )
            except Exception as exc:
                result_str = f"Error executing tool {name}: {exc}"
                logger.error(
                    "Iteration %d: tool %s failed: %s",
                    iteration,
                    name,
                    exc,
                    exc_info=True,
                )

            # Append tool result to messages
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": result_str,
                }
            )

            # Check for stop signal but defer until all tool calls are done
            try:
                parsed = json.loads(result_str)
                if isinstance(parsed, dict) and parsed.get("__stop_turn__"):
                    logger.info("Tool %s requested turn stop (deferred)", name)
                    if stop_reply is None:
                        stop_reply = parsed.get("reply", result_str)
            except (json.JSONDecodeError, TypeError):
                pass

        # After all tool calls in this batch, honour the deferred stop
        if stop_reply is not None:
            logger.info("Executing deferred turn stop")
            return _maybe_with_usage(stop_reply)

    # Exhausted max_iterations — make one final call without tools to get a
    # text summary from the model, or return a fallback error.
    logger.warning(
        "Tool loop reached max iterations (%d), requesting final response",
        max_iterations,
    )
    try:
        response = await client.chat(messages, model)
        _accumulate_usage(response)
        choice = response["choices"][0]
        content = choice["message"].get("content", "")
        if content:
            return _maybe_with_usage(content)
    except Exception as exc:
        logger.error("Final call after max iterations failed: %s", exc)

    return _maybe_with_usage(
        f"Reached the maximum of {max_iterations} tool iterations "
        "without a final response."
    )
