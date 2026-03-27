"""Configuration loading from ~/.spare-paw/config.yaml with runtime overrides.

Thread-safe config with dot-notation access and deep merge against defaults.
"""

from __future__ import annotations

import copy
import threading
from pathlib import Path
from typing import Any

import yaml

from spare_paw.platform import default_allowed_paths, platform_label

CONFIG_DIR = Path.home() / ".spare-paw"
CONFIG_PATH = CONFIG_DIR / "config.yaml"

DEFAULT_MODEL = "xiaomi/mimo-v2-omni"
MODEL_ROLES = ("main_agent", "coder", "planner", "cron", "researcher", "analyst", "summary", "vision")


def _build_defaults() -> dict[str, Any]:
    """Build DEFAULTS with platform-aware values resolved at import time."""
    label = platform_label()
    return {
        "models": {
            "main_agent": DEFAULT_MODEL,
            "coder": "xiaomi/mimo-v2-pro",
            "planner": "xiaomi/mimo-v2-omni",
            "cron": "minimax/minimax-m2.7",
            "researcher": "minimax/minimax-m2.7",
            "analyst": "minimax/minimax-m2.7",
            "summary": "google/gemini-3.1-flash-lite-preview",
            "vision": "google/gemini-3.1-flash-lite-preview",
        },
        "context": {
            "max_messages": 64,
            "token_budget": 120000,
            "safety_margin": 0.85,
        },
        "agent": {
            "max_tool_iterations": 20,
            "system_prompt": (
                "You are a personal AI assistant running 24/7.\n"
                "You have access to the local filesystem, shell, web search, and web scraping.\n"
                "You can manage scheduled tasks (crons) for the user.\n"
                "Be concise.\n"
                "Current time: {current_time}\n"
                f"Device: {label}"
            ),
        },
        "tools": {
            "shell": {
                "enabled": True,
                "timeout_seconds": 30,
                "max_output_chars": 10000,
            },
            "files": {
                "enabled": True,
                "allowed_paths": default_allowed_paths(),
            },
            "web_search": {
                "enabled": True,
                "max_results": 5,
            },
            "web_scrape": {
                "enabled": True,
                "timeout_seconds": 15,
                "max_content_chars": 20000,
            },
            "cron": {
                "enabled": True,
            },
        },
        "logging": {
            "level": "INFO",
            "max_bytes": 10485760,
            "backup_count": 3,
        },
        "mcp": {
            "servers": [],
        },
    }


DEFAULTS: dict[str, Any] = _build_defaults()


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Deep merge *override* into a copy of *base*. Override wins for leaf values."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _resolve_dot(data: dict[str, Any], dotpath: str) -> Any:
    """Resolve a dot-separated path like 'models.default' into a nested dict value."""
    keys = dotpath.split(".")
    current: Any = data
    for key in keys:
        if isinstance(current, dict):
            if key not in current:
                return None
            current = current[key]
        else:
            return None
    return current


def _set_dot(data: dict[str, Any], dotpath: str, value: Any) -> None:
    """Set a value at a dot-separated path, creating intermediate dicts as needed."""
    keys = dotpath.split(".")
    current = data
    for key in keys[:-1]:
        if key not in current or not isinstance(current[key], dict):
            current[key] = {}
        current = current[key]
    current[keys[-1]] = value


class Config:
    """Thread-safe configuration container with dot-notation access and runtime overrides."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._data: dict[str, Any] = copy.deepcopy(DEFAULTS)
        self._overrides: dict[str, Any] = {}

    def load(self, path: Path | None = None) -> None:
        """Load config from YAML file, deep-merging with defaults."""
        path = path or CONFIG_PATH
        with self._lock:
            if path.exists():
                with open(path, "r") as f:
                    file_data = yaml.safe_load(f) or {}
                self._data = _deep_merge(DEFAULTS, file_data)
            else:
                self._data = copy.deepcopy(DEFAULTS)
            # Re-apply any active runtime overrides on top
            if self._overrides:
                self._data = _deep_merge(self._data, self._overrides)

    def get(self, dotpath: str, default: Any = None) -> Any:
        """Get a config value using dot notation, e.g. 'models.default'."""
        with self._lock:
            value = _resolve_dot(self._data, dotpath)
            return value if value is not None else default

    def set_override(self, dotpath: str, value: Any) -> None:
        """Set a runtime override that persists until reset.

        Used for /config model <name> etc.
        """
        with self._lock:
            _set_dot(self._overrides, dotpath, value)
            _set_dot(self._data, dotpath, value)

    def reset_overrides(self) -> None:
        """Clear all runtime overrides and reload from file defaults."""
        with self._lock:
            self._overrides.clear()
            self.load()

    @property
    def data(self) -> dict[str, Any]:
        """Return a deep copy of the full config dict."""
        with self._lock:
            return copy.deepcopy(self._data)

    def __getitem__(self, key: str) -> Any:
        with self._lock:
            return self._data[key]

    def __contains__(self, key: str) -> bool:
        with self._lock:
            return key in self._data


def resolve_model(config: "Config", role: str) -> str:
    """Resolve the model for a given role using the fallback chain.

    Chain: models.<role> → models.main_agent → DEFAULT_MODEL
    """
    if role != "main_agent":
        specific = config.get(f"models.{role}")
        if specific:
            return specific
    return config.get("models.main_agent", DEFAULT_MODEL)


# Module-level singleton
config = Config()
