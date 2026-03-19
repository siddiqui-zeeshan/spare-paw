"""Utilities for redacting secrets from log output."""

from __future__ import annotations

import re

_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"sk-[a-zA-Z0-9-]{20,}"),                        # OpenAI / OpenRouter
    re.compile(r"ghp_[a-zA-Z0-9]{36}"),                          # GitHub PAT
    re.compile(r"gsk_[a-zA-Z0-9]{20,}"),                         # Groq
    re.compile(r"tvly-[a-zA-Z0-9]{20,}"),                        # Tavily
    re.compile(r"\d{8,10}:[A-Za-z0-9_-]{35}"),                    # Telegram bot token
    re.compile(r"Bearer [a-zA-Z0-9._-]{20,}"),                   # Bearer (min 20 chars)
    re.compile(r"(?i)(password|secret|token|api_?key)=\S+"),     # query params
]

_REPLACEMENT = "[REDACTED]"


def redact_secrets(text: str) -> str:
    """Replace known secret patterns in *text* with ``[REDACTED]``."""
    for pattern in _PATTERNS:
        text = pattern.sub(_REPLACEMENT, text)
    return text
