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
            # Bearer token
            (
                "Authorization: Bearer eyJhbGciOiJSUzI1NiJ9.payload.sig",
                "Authorization: [REDACTED]",
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
            "Using sk-" + "x" * 20 + " and token=abc123 and Bearer tok.en-val"
        )
        result = redact_secrets(text)
        assert "[REDACTED]" in result
        assert "sk-" not in result
        assert "abc123" not in result
        assert "tok.en-val" not in result
