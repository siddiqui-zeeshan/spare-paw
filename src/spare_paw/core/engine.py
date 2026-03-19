"""Platform-agnostic message processor.

Handles message processing, agent callbacks, and the message queue.
No Telegram imports — all platform-specific behavior is delegated
to the MessageBackend.
"""

from __future__ import annotations


def split_text(text: str, max_length: int) -> list[str]:
    """Split text into chunks of at most *max_length* characters.

    Prefers splitting at newlines, falling back to hard cuts.
    """
    if not text:
        return [text]
    chunks: list[str] = []
    while text:
        if len(text) <= max_length:
            chunks.append(text)
            break

        cut_at = text.rfind("\n", 0, max_length)
        if cut_at <= 0:
            cut_at = max_length

        chunks.append(text[:cut_at])
        text = text[cut_at:].lstrip("\n")

    return chunks
