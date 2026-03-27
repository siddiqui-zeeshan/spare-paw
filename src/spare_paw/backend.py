"""MessageBackend protocol and IncomingMessage dataclass.

Defines the platform-agnostic interface that every messaging backend
(Telegram, webhook, future) must implement, and the self-contained
message representation that backends produce for the engine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass
class IncomingMessage:
    """Platform-agnostic representation of a user message.

    Backends populate ALL fields before handing off to the engine.
    The engine never needs to inspect platform-specific structures.
    """

    text: str | None = None
    image_bytes: bytes | None = None
    image_mime: str = "image/jpeg"
    video_bytes: bytes | None = None
    video_mime: str = "video/mp4"
    voice_bytes: bytes | None = None
    caption: str | None = None
    cron_context: str | None = None
    command: str | None = None
    command_args: list[str] = field(default_factory=list)
    user_id: int | str | None = None
    plan: bool = False


@runtime_checkable
class MessageBackend(Protocol):
    """Interface every messaging backend must implement.

    The engine outputs markdown. Each backend is responsible for
    converting to its own format (HTML for Telegram, raw for webhook).
    Chunking is also the backend's responsibility.
    """

    async def send_text(self, text: str) -> None: ...

    async def send_file(self, path: str, caption: str = "") -> None: ...

    async def send_typing(self) -> None: ...

    async def send_notification(
        self, text: str, actions: list[dict] | None = None
    ) -> None: ...

    async def start(self) -> None: ...

    async def stop(self) -> None: ...
