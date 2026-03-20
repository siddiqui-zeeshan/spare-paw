"""Deep thinking planning phase.

When invoked via /plan, decomposes the user's request into a structured
execution plan before the main tool loop runs.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from spare_paw.router.openrouter import OpenRouterClient

logger = logging.getLogger(__name__)

PLANNING_SYSTEM_PROMPT = """\
You are a planning assistant. Your job is to analyze a user's request and \
create a clear, actionable execution plan BEFORE any tools are called.

Given the conversation context, output a plan in this format:

## Plan
1. [Step description] → tools: [tool1, tool2] | agent: [researcher/coder/analyst/none]
2. [Step description] → tools: [tool1, tool2] | agent: [type or none]
...

## Parallel groups
- Group 1: steps [1, 2] (independent, can run in parallel)
- Group 2: steps [3] (depends on group 1)

## Notes
[Any important considerations, edge cases, or clarifications]

Rules:
- Be concise — only list steps that require action.
- Identify which steps are independent (parallelizable) vs dependent.
- For each step, name the specific tools or agent type needed.
- If the request is simple (single tool call), say so — don't over-plan.\
"""


async def create_plan(
    messages: list[dict[str, Any]],
    config: Any,
    client: "OpenRouterClient",
) -> str:
    """Generate an execution plan from the conversation context.

    Makes a single LLM call with a planning-specific system prompt and
    no tool schemas. Returns the plan as markdown text, or an empty
    string on failure (graceful degradation).
    """
    model = config.get("planning.model") or config.get(
        "models.default", "google/gemini-2.0-flash"
    )

    plan_messages = [{"role": "system", "content": PLANNING_SYSTEM_PROMPT}]
    for msg in messages:
        if msg["role"] != "system":
            plan_messages.append(msg)

    try:
        response = await client.chat(plan_messages, model)
        return response["choices"][0]["message"].get("content", "")
    except Exception:
        logger.exception("Planning phase failed — proceeding without plan")
        return ""
