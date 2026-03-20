"""Tests for md_to_html — redirects to test_telegram_backend.py.

The canonical tests now live in test_telegram_backend.py.
This file verifies the re-export from bot.handler still works.
"""

from spare_paw.bot.handler import _md_to_html


def test_reexport_works():
    """The _md_to_html re-export from handler.py should still function."""
    assert _md_to_html("**bold**") == "<b>bold</b>"
