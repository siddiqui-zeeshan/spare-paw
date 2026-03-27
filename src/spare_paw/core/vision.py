"""Vision preprocessing — describe images/videos via a vision-capable model."""

from __future__ import annotations

import base64
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from spare_paw.router.openrouter import OpenRouterClient

_SYSTEM_PROMPT = (
    "Describe this image/video in the context of the user's message. "
    "Focus on details relevant to what the user is asking or saying."
)

_GENERIC_PROMPT = "Describe this image/video in detail."


async def describe_media(
    *,
    client: OpenRouterClient,
    media_bytes: bytes,
    media_mime: str,
    user_text: str | None,
    model: str,
) -> str:
    """Call a vision model to describe image or video content.

    Returns the model's text description of the media.
    No semaphore needed — client.chat() already acquires app_state.semaphore internally.
    """
    b64 = base64.b64encode(media_bytes).decode("ascii")
    data_url = f"data:{media_mime};base64,{b64}"

    is_video = media_mime.startswith("video/")
    if is_video:
        media_block: dict[str, Any] = {
            "type": "video_url",
            "video_url": {"url": data_url},
        }
    else:
        media_block = {
            "type": "image_url",
            "image_url": {"url": data_url},
        }

    prompt = user_text if user_text else _GENERIC_PROMPT

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                media_block,
            ],
        },
    ]

    response = await client.chat(messages=messages, model=model)
    return response["choices"][0]["message"]["content"]
