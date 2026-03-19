"""Tests for spare_paw.util.redact — secret redaction from log strings."""

from __future__ import annotations

import pytest

from spare_paw.util.redact import redact_secrets


class TestRedactSecrets:
    @pytest.mark.parametrize(
        "text, expected",
        [
            # OpenRouter / OpenAI key
            (
                "key=sk-abcdefghijklmnopqrstuvwxyz1234",
                "key=[REDACTED]",
            ),
            # GitHub token
            (
                "token ghp_" + "A" * 36,
                "token [REDACTED]",
            ),
            # Groq key
            (
                "key gsk_" + "a" * 30,
                "key [REDACTED]",
            ),
            # Tavily key
            (
                "key tvly-" + "a" * 25,
                "key [REDACTED]",
            ),
            # Telegram bot token
            (
                "bot 8511158159:AAH64DjygEnQ-ruFrsfyXeeOkvhcJAj6AKY",
                "bot [REDACTED]",
            ),
            # Bearer token (long enough)
            (
                "Authorization: Bearer eyJhbGciOiJSUzI1NiJ9.payload.sig",
                "Authorization: [REDACTED]",
            ),
            # Bearer short word is NOT redacted (no false positive)
            (
                "Use a Bearer token for auth",
                "Use a Bearer token for auth",
            ),
            # password= query param
            (
                "https://example.com?password=supersecret",
                "https://example.com?[REDACTED]",
            ),
            # token= query param
            (
                "https://example.com?token=abc123xyz",
                "https://example.com?[REDACTED]",
            ),
            # api_key= query param
            (
                "request?api_key=myrandomapikey",
                "request?[REDACTED]",
            ),
            # secret= query param
            (
                "config?secret=mysecretvalue",
                "config?[REDACTED]",
            ),
            # apikey= (no underscore)
            (
                "url?apikey=abcdef123456",
                "url?[REDACTED]",
            ),
        ],
    )
    def test_pattern_redacted(self, text: str, expected: str) -> None:
        assert redact_secrets(text) == expected

    def test_plain_text_unchanged(self) -> None:
        plain = "Hello, this is a normal log message with no secrets."
        assert redact_secrets(plain) == plain

    def test_empty_string(self) -> None:
        assert redact_secrets("") == ""

    def test_multiple_secrets_in_one_string(self) -> None:
        text = (
            "Using sk-" + "x" * 20 + " and token=abc123"
            " and Bearer " + "a" * 30
        )
        result = redact_secrets(text)
        assert "[REDACTED]" in result
        assert "sk-" not in result
        assert "abc123" not in result
        assert "a" * 30 not in result
