"""Tool-use execution loop for multi-turn function calling."""

from __future__ import annotations

import json
import logging
import uuid
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from spare_paw.router.openrouter import OpenRouterClient
    from spare_paw.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


@dataclass
class ToolEvent:
    """Event emitted during tool loop execution for UI feedback."""

    kind: str  # "tool_start" | "tool_end" | "llm_start" | "llm_end"
    tool_name: str | None = None
    tool_args: dict[str, Any] | None = None
    result_preview: str | None = None
    iteration: int = 0

# Per-turn (not per-session) call limits. Tools not listed are unlimited.
DEFAULT_TOOL_LIMITS: dict[str, int] = {
    "web_scrape": 5,
    "web_search": 5,
    "tavily_search": 5,
    "shell": 10,
    "spawn_agent": 3,
}


async def run_tool_loop(
    client: "OpenRouterClient",
    messages: list[dict[str, Any]],
    model: str,
    tools: list[dict[str, Any]],
    tool_registry: "ToolRegistry",
    max_iterations: int = 20,
    executor: ProcessPoolExecutor | None = None,
    track_usage: bool = False,
    tool_limits: dict[str, int | None] | None = None,
    on_event: Callable[[ToolEvent], None] | None = None,
    on_token: Callable[[str], None] | None = None,
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
        tool_limits: Per-tool call limits for this turn. Merged on top of
            ``DEFAULT_TOOL_LIMITS``; set a value to override a default, or
            ``None`` to remove a default limit (making the tool unlimited).

    Returns:
        The final text content from the model when ``track_usage`` is False.
        A tuple of ``(text, usage_dict)`` when ``track_usage`` is True, where
        *usage_dict* has keys ``prompt_tokens``, ``completion_tokens``, and
        ``total_tokens``.
    """
    merged = {**DEFAULT_TOOL_LIMITS, **(tool_limits or {})}
    effective_limits: dict[str, int] = {
        k: v for k, v in merged.items() if v is not None
    }
    call_counts: dict[str, int] = {}

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
        if on_event is not None:
            on_event(ToolEvent(kind="llm_start", iteration=iteration))

        response = await client.chat(messages, model, tools)
        _accumulate_usage(response)
        usage = response.get("usage", {})
        if usage:
            logger.info(
                "Iteration %d: tokens prompt=%d completion=%d total=%d",
                iteration,
                usage.get("prompt_tokens", 0),
                usage.get("completion_tokens", 0),
                usage.get("total_tokens", 0),
            )

        if on_event is not None:
            on_event(ToolEvent(kind="llm_end", iteration=iteration))

        choice = response["choices"][0]
        assistant_message = choice["message"]

        # Check for tool calls
        tool_calls = assistant_message.get("tool_calls")

        if not tool_calls:
            # No tool calls — if streaming requested, re-do final call via stream
            content = assistant_message.get("content", "")
            if on_token is not None and content:
                # Emit tokens in word-sized chunks
                for word in content.split(" "):
                    on_token(word + " ")
            return _maybe_with_usage(content or "")

        # Append the assistant message (with tool_calls) to the conversation
        messages.append(assistant_message)

        # Generate a shared group_id for all spawn_agent calls in this batch
        has_spawn = any(
            tc["function"]["name"] == "spawn_agent" for tc in tool_calls
        )
        batch_group_id = uuid.uuid4().hex[:8] if has_spawn else None

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

            # Inject batch group_id into spawn_agent calls
            if name == "spawn_agent" and batch_group_id is not None:
                args["group_id"] = batch_group_id

            # Enforce per-turn rate limits
            call_counts[name] = call_counts.get(name, 0) + 1
            limit = effective_limits.get(name)
            if limit is not None and call_counts[name] > limit:
                result_str = (
                    f"Rate limit: {name} called {call_counts[name]}/{limit} "
                    "times this turn. Try a different approach."
                )
                logger.warning(
                    "Iteration %d: tool %s exceeded rate limit (%d/%d)",
                    iteration,
                    name,
                    call_counts[name],
                    limit,
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": result_str,
                    }
                )
                continue

            # Fire tool_start event
            if on_event is not None:
                on_event(ToolEvent(
                    kind="tool_start",
                    tool_name=name,
                    tool_args=args,
                    iteration=iteration,
                ))

            # Execute the tool, catching any exception
            try:
                result = await tool_registry.execute(name, args, executor)
                result_str = str(result) if not isinstance(result, str) else result
                preview = result_str[:200] + "..." if len(result_str) > 200 else result_str
                logger.info(
                    "Iteration %d: tool %s executed successfully: %s", iteration, name, preview
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

            # Fire tool_end event
            if on_event is not None:
                on_event(ToolEvent(
                    kind="tool_end",
                    tool_name=name,
                    result_preview=result_str[:200],
                    iteration=iteration,
                ))

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
