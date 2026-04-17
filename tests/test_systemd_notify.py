"""Tests for systemd sd_notify() helper — used for Type=notify watchdog."""

from __future__ import annotations

import os
import socket
from unittest.mock import patch

from spare_paw.gateway import _sd_notify


class TestSdNotify:
    def test_no_socket_env_is_noop(self):
        """Without NOTIFY_SOCKET env var, sd_notify is a no-op and does not raise."""
        env = {k: v for k, v in os.environ.items() if k != "NOTIFY_SOCKET"}
        with patch.dict(os.environ, env, clear=True):
            # Should not raise
            _sd_notify("READY=1")
            _sd_notify("WATCHDOG=1")

    def test_sends_message_over_unix_socket(self):
        """With NOTIFY_SOCKET set, sd_notify sends the message to the socket."""
        import tempfile

        # Use a short tmpdir to stay under macOS 104-char sun_path limit
        with tempfile.TemporaryDirectory(dir="/tmp") as d:
            sock_path = os.path.join(d, "n.sock")

            server = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
            server.bind(sock_path)
            server.settimeout(1.0)

            try:
                with patch.dict(os.environ, {"NOTIFY_SOCKET": sock_path}):
                    _sd_notify("WATCHDOG=1")

                data, _ = server.recvfrom(1024)
                assert data == b"WATCHDOG=1"
            finally:
                server.close()

    def test_swallows_socket_errors(self):
        """If the socket is unreachable, sd_notify does not crash the caller."""
        with patch.dict(os.environ, {"NOTIFY_SOCKET": "/nonexistent/path.sock"}):
            # Should not raise
            _sd_notify("WATCHDOG=1")
