"""Voice message transcription via Groq Whisper API."""

from __future__ import annotations

import logging

import aiohttp
from telegram import File as TelegramFile

logger = logging.getLogger(__name__)

GROQ_WHISPER_URL = "https://api.groq.com/openai/v1/audio/transcriptions"
WHISPER_MODEL = "whisper-large-v3"


class VoiceTranscriptionError(Exception):
    """Raised when voice transcription fails."""


async def transcribe(voice_file: TelegramFile, config: dict) -> str:
    """Download a Telegram voice file and transcribe it via Groq Whisper.

    Args:
        voice_file: A ``telegram.File`` object obtained from ``voice.get_file()``.
        config: The full application config dict. Must contain ``groq.api_key``.

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

    # Download the voice file bytes from Telegram
    voice_bytes = await voice_file.download_as_bytearray()

    # Send to Groq Whisper for transcription
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
