"""Model router: OpenRouter API client and tool-use execution loop."""

from spare_paw.router.openrouter import OpenRouterClient, OpenRouterError
from spare_paw.router.tool_loop import run_tool_loop

__all__ = ["OpenRouterClient", "OpenRouterError", "run_tool_loop"]
