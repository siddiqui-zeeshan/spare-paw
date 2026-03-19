"""Utilities for redacting secrets from log output."""

from __future__ import annotations

import re

_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"sk-[a-zA-Z0-9-]{20,}"),
    re.compile(r"ghp_[a-zA-Z0-9]{36}"),
    re.compile(r"Bearer [a-zA-Z0-9._-]+"),
    re.compile(r"(?i)(password|token|api_key)=\S+"),
]

_REPLACEMENT = "[REDACTED]"


def redact_secrets(text: str) -> str:
    """Replace known secret patterns in *text* with ``[REDACTED]``."""
    for pattern in _PATTERNS:
        text = pattern.sub(_REPLACEMENT, text)
    return text
