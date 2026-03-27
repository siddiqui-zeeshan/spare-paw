"""Tests for MessageBackend protocol and IncomingMessage dataclass."""

from __future__ import annotations

import pytest

from spare_paw.backend import IncomingMessage, MessageBackend


class TestIncomingMessage:
    def test_defaults(self):
        msg = IncomingMessage()
        assert msg.text is None
        assert msg.image_bytes is None
        assert msg.image_mime == "image/jpeg"
        assert msg.voice_bytes is None
        assert msg.caption is None
        assert msg.cron_context is None
        assert msg.command is None
        assert msg.command_args == []
        assert msg.user_id is None

    def test_text_only(self):
        msg = IncomingMessage(text="hello", user_id=123)
        assert msg.text == "hello"
        assert msg.user_id == 123

    def test_voice_bytes(self):
        audio = b"\x00\x01\x02\x03"
        msg = IncomingMessage(voice_bytes=audio)
        assert msg.voice_bytes == audio

    def test_image_bytes(self):
        img = b"\xff\xd8\xff\xe0"
        msg = IncomingMessage(image_bytes=img, image_mime="image/png", caption="a photo")
        assert msg.image_bytes == img
        assert msg.image_mime == "image/png"
        assert msg.caption == "a photo"

    def test_command(self):
        msg = IncomingMessage(command="status", command_args=["verbose"])
        assert msg.command == "status"
        assert msg.command_args == ["verbose"]

    def test_cron_context(self):
        msg = IncomingMessage(text="looks good", cron_context="cron output here")
        assert msg.cron_context == "cron output here"

    def test_command_args_independent_instances(self):
        """Default mutable field should not be shared between instances."""
        msg1 = IncomingMessage()
        msg2 = IncomingMessage()
        msg1.command_args.append("x")
        assert msg2.command_args == []


class TestIncomingMessageVideo:
    def test_video_fields_default(self):
        msg = IncomingMessage()
        assert msg.video_bytes is None
        assert msg.video_mime == "video/mp4"

    def test_video_fields_set(self):
        msg = IncomingMessage(video_bytes=b"\x00\x01", video_mime="video/webm")
        assert msg.video_bytes == b"\x00\x01"
        assert msg.video_mime == "video/webm"


class TestMessageBackendProtocol:
    def test_runtime_checkable(self):
        """A class implementing all methods satisfies isinstance check."""

        class StubBackend:
            async def send_text(self, text: str) -> None:
                pass

            async def send_file(self, path: str, caption: str = "") -> None:
                pass

            async def send_typing(self) -> None:
                pass

            async def send_notification(self, text: str, actions: list[dict] | None = None) -> None:
                pass

            async def start(self) -> None:
                pass

            async def stop(self) -> None:
                pass

        backend = StubBackend()
        assert isinstance(backend, MessageBackend)

    def test_incomplete_class_fails(self):
        """A class missing methods does NOT satisfy the protocol."""

        class Incomplete:
            async def send_text(self, text: str) -> None:
                pass

        assert not isinstance(Incomplete(), MessageBackend)

    @pytest.mark.asyncio
    async def test_stub_backend_callable(self):
        """Stub backend methods can be awaited without error."""

        class StubBackend:
            async def send_text(self, text: str) -> None:
                pass

            async def send_file(self, path: str, caption: str = "") -> None:
                pass

            async def send_typing(self) -> None:
                pass

            async def send_notification(self, text: str, actions: list[dict] | None = None) -> None:
                pass

            async def start(self) -> None:
                pass

            async def stop(self) -> None:
                pass

        backend = StubBackend()
        await backend.send_text("hello")
        await backend.send_file("/tmp/test.txt")
        await backend.send_typing()
        await backend.send_notification("alert", [{"label": "OK", "callback_data": "ok"}])
        await backend.start()
        await backend.stop()
