"""Voice message transcription via Groq Whisper API.

Accepts raw bytes — no Telegram dependency. The caller (bot handler or
webhook handler) is responsible for downloading the audio file.
"""

from __future__ import annotations

import logging

import aiohttp

logger = logging.getLogger(__name__)

GROQ_WHISPER_URL = "https://api.groq.com/openai/v1/audio/transcriptions"
WHISPER_MODEL = "whisper-large-v3"


class VoiceTranscriptionError(Exception):
    """Raised when voice transcription fails."""


async def transcribe(voice_bytes: bytes, config: dict) -> str:
    """Transcribe audio bytes via Groq Whisper.

    Args:
        voice_bytes: Raw audio data (OGG, WAV, etc.).
        config: Application config dict. Must contain ``groq.api_key``.

    Returns:
        The transcribed text.

    Raises:
        VoiceTranscriptionError: If Groq is not configured or the API call fails.
    """
    groq_cfg = config.get("groq", {})
    api_key = groq_cfg.get("api_key", "")
    if not api_key:
        raise VoiceTranscriptionError(
            "Voice messages require a Groq API key in config."
        )

    headers = {"Authorization": f"Bearer {api_key}"}

    form = aiohttp.FormData()
    form.add_field(
        "file",
        bytes(voice_bytes),
        filename="voice.ogg",
        content_type="audio/ogg",
    )
    form.add_field("model", WHISPER_MODEL)

    async with aiohttp.ClientSession() as session:
        async with session.post(
            GROQ_WHISPER_URL, headers=headers, data=form
        ) as resp:
            if resp.status != 200:
                body = await resp.text()
                raise VoiceTranscriptionError(
                    f"Groq Whisper API returned {resp.status}: {body}"
                )
            data = await resp.json()

    text = data.get("text", "").strip()
    if not text:
        raise VoiceTranscriptionError("Groq Whisper returned empty transcription.")

    logger.info("Transcribed voice message (%d bytes) -> %d chars", len(voice_bytes), len(text))
    return text
