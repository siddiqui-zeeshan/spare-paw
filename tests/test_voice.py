"""Tests for core/voice.py — voice transcription via Groq Whisper."""

from __future__ import annotations

import ast
import inspect
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from spare_paw.core.voice import VoiceTranscriptionError, transcribe


class TestTranscribe:
    @pytest.mark.asyncio
    async def test_successful_transcription(self):
        """transcribe() returns text on a 200 response."""
        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value={"text": "hello world"})

        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        mock_session.post = MagicMock(return_value=AsyncMock(
            __aenter__=AsyncMock(return_value=mock_resp),
            __aexit__=AsyncMock(return_value=False),
        ))

        config = {"groq": {"api_key": "test-key"}}

        with patch("aiohttp.ClientSession", return_value=mock_session):
            result = await transcribe(b"\x00\x01\x02", config)

        assert result == "hello world"

    @pytest.mark.asyncio
    async def test_missing_api_key_raises(self):
        """transcribe() raises VoiceTranscriptionError with no API key."""
        with pytest.raises(VoiceTranscriptionError, match="Groq API key"):
            await transcribe(b"\x00", {})

    @pytest.mark.asyncio
    async def test_empty_api_key_raises(self):
        with pytest.raises(VoiceTranscriptionError, match="Groq API key"):
            await transcribe(b"\x00", {"groq": {"api_key": ""}})

    @pytest.mark.asyncio
    async def test_non_200_raises(self):
        """transcribe() raises on non-200 response."""
        mock_resp = AsyncMock()
        mock_resp.status = 500
        mock_resp.text = AsyncMock(return_value="Internal Server Error")

        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        mock_session.post = MagicMock(return_value=AsyncMock(
            __aenter__=AsyncMock(return_value=mock_resp),
            __aexit__=AsyncMock(return_value=False),
        ))

        config = {"groq": {"api_key": "test-key"}}

        with patch("aiohttp.ClientSession", return_value=mock_session):
            with pytest.raises(VoiceTranscriptionError, match="500"):
                await transcribe(b"\x00", config)

    @pytest.mark.asyncio
    async def test_empty_transcription_raises(self):
        """transcribe() raises when Groq returns empty text."""
        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value={"text": ""})

        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        mock_session.post = MagicMock(return_value=AsyncMock(
            __aenter__=AsyncMock(return_value=mock_resp),
            __aexit__=AsyncMock(return_value=False),
        ))

        config = {"groq": {"api_key": "test-key"}}

        with patch("aiohttp.ClientSession", return_value=mock_session):
            with pytest.raises(VoiceTranscriptionError, match="empty"):
                await transcribe(b"\x00", config)

    @pytest.mark.asyncio
    async def test_accepts_bytes_not_telegram_file(self):
        """transcribe() signature accepts bytes, not TelegramFile."""
        sig = inspect.signature(transcribe)
        params = list(sig.parameters.keys())
        assert params[0] == "voice_bytes"

    def test_no_telegram_import(self):
        """core/voice.py must not import from telegram."""
        import spare_paw.core.voice as mod

        source = inspect.getsource(mod)
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert not alias.name.startswith("telegram"), (
                        f"Found 'import {alias.name}' in core/voice.py"
                    )
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    assert not node.module.startswith("telegram"), (
                        f"Found 'from {node.module} import ...' in core/voice.py"
                    )
