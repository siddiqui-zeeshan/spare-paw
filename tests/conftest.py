"""Shared pytest fixtures and configuration."""

from __future__ import annotations

import os


def pytest_sessionfinish(session, exitstatus):
    """Force-kill any lingering APScheduler threads after all tests complete."""
    os._exit(exitstatus)
