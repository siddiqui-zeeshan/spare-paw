"""Tests for spare_paw.config — YAML loading, deep merge, dot-notation, overrides."""

from __future__ import annotations

import threading
from pathlib import Path

import yaml

from spare_paw.config import Config, DEFAULTS, _deep_merge, _resolve_dot, _set_dot


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_yaml(path: Path, data: dict) -> Path:
    """Write a dict as YAML to *path* and return the path."""
    path.write_text(yaml.dump(data, default_flow_style=False))
    return path


# ---------------------------------------------------------------------------
# _deep_merge
# ---------------------------------------------------------------------------

class TestDeepMerge:
    def test_override_leaf(self):
        base = {"a": 1, "b": {"c": 2, "d": 3}}
        override = {"b": {"c": 99}}
        result = _deep_merge(base, override)
        assert result["b"]["c"] == 99
        assert result["b"]["d"] == 3  # untouched

    def test_add_new_key(self):
        base = {"a": 1}
        override = {"b": 2}
        result = _deep_merge(base, override)
        assert result == {"a": 1, "b": 2}

    def test_does_not_mutate_base(self):
        base = {"a": {"x": 1}}
        override = {"a": {"x": 99}}
        _deep_merge(base, override)
        assert base["a"]["x"] == 1

    def test_override_dict_with_scalar(self):
        base = {"a": {"nested": True}}
        override = {"a": "flat"}
        result = _deep_merge(base, override)
        assert result["a"] == "flat"


# ---------------------------------------------------------------------------
# _resolve_dot / _set_dot
# ---------------------------------------------------------------------------

class TestDotNotation:
    def test_resolve_simple(self):
        data = {"a": {"b": {"c": 42}}}
        assert _resolve_dot(data, "a.b.c") == 42

    def test_resolve_missing_returns_none(self):
        assert _resolve_dot({"a": 1}, "a.b.c") is None
        assert _resolve_dot({}, "x") is None

    def test_resolve_top_level(self):
        assert _resolve_dot({"key": "val"}, "key") == "val"

    def test_set_creates_intermediate(self):
        data: dict = {}
        _set_dot(data, "a.b.c", 7)
        assert data == {"a": {"b": {"c": 7}}}

    def test_set_overwrites_existing(self):
        data = {"a": {"b": 1}}
        _set_dot(data, "a.b", 2)
        assert data["a"]["b"] == 2


# ---------------------------------------------------------------------------
# Config — loading from YAML
# ---------------------------------------------------------------------------

class TestConfigLoad:
    def test_load_from_yaml(self, tmp_path: Path):
        cfg_file = _write_yaml(tmp_path / "config.yaml", {
            "logging": {"level": "DEBUG"},
        })
        cfg = Config()
        cfg.load(cfg_file)

        assert cfg.get("logging.level") == "DEBUG"
        # defaults still present for keys not in file
        assert cfg.get("logging.max_bytes") == DEFAULTS["logging"]["max_bytes"]

    def test_load_missing_file_uses_defaults(self, tmp_path: Path):
        cfg = Config()
        cfg.load(tmp_path / "nonexistent.yaml")
        assert cfg.get("context.max_messages") == DEFAULTS["context"]["max_messages"]

    def test_load_empty_file_uses_defaults(self, tmp_path: Path):
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text("")
        cfg = Config()
        cfg.load(cfg_file)
        assert cfg.get("agent.max_tool_iterations") == DEFAULTS["agent"]["max_tool_iterations"]


# ---------------------------------------------------------------------------
# Config — deep merge with defaults (missing keys get defaults)
# ---------------------------------------------------------------------------

class TestConfigMergeDefaults:
    def test_partial_override_keeps_defaults(self, tmp_path: Path):
        cfg_file = _write_yaml(tmp_path / "config.yaml", {
            "tools": {"shell": {"timeout_seconds": 60}},
        })
        cfg = Config()
        cfg.load(cfg_file)

        # overridden value
        assert cfg.get("tools.shell.timeout_seconds") == 60
        # default value still present
        assert cfg.get("tools.shell.enabled") is True
        assert cfg.get("tools.web_search.max_results") == 5


# ---------------------------------------------------------------------------
# Config — dot-notation get()
# ---------------------------------------------------------------------------

class TestConfigGet:
    def test_get_nested(self):
        cfg = Config()
        assert cfg.get("context.safety_margin") == 0.85

    def test_get_missing_returns_default(self):
        cfg = Config()
        assert cfg.get("nonexistent.path") is None
        assert cfg.get("nonexistent.path", "fallback") == "fallback"

    def test_getitem(self):
        cfg = Config()
        assert isinstance(cfg["context"], dict)
        assert cfg["context"]["max_messages"] == 64

    def test_contains(self):
        cfg = Config()
        assert "context" in cfg
        assert "nonexistent" not in cfg


# ---------------------------------------------------------------------------
# Config — runtime overrides
# ---------------------------------------------------------------------------

class TestConfigOverrides:
    def test_set_override(self):
        cfg = Config()
        cfg.set_override("context.max_messages", 128)
        assert cfg.get("context.max_messages") == 128

    def test_override_persists_after_reload(self, tmp_path: Path):
        cfg_file = _write_yaml(tmp_path / "config.yaml", {})
        cfg = Config()
        cfg.load(cfg_file)
        cfg.set_override("logging.level", "ERROR")
        # reload should re-apply override on top
        cfg.load(cfg_file)
        assert cfg.get("logging.level") == "ERROR"

    def test_reset_overrides(self, tmp_path: Path):
        cfg_file = _write_yaml(tmp_path / "config.yaml", {})
        cfg = Config()
        cfg.load(cfg_file)
        cfg.set_override("context.max_messages", 999)
        assert cfg.get("context.max_messages") == 999

        cfg.reset_overrides()
        assert cfg.get("context.max_messages") == DEFAULTS["context"]["max_messages"]


# ---------------------------------------------------------------------------
# Config — basic thread safety
# ---------------------------------------------------------------------------

class TestConfigThreadSafety:
    def test_concurrent_reads_and_writes(self):
        """Ensure no exceptions when reading/writing from multiple threads."""
        cfg = Config()
        errors: list[Exception] = []

        def writer():
            try:
                for i in range(200):
                    cfg.set_override("context.max_messages", i)
            except Exception as exc:
                errors.append(exc)

        def reader():
            try:
                for _ in range(200):
                    cfg.get("context.max_messages")
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=writer) for _ in range(3)]
        threads += [threading.Thread(target=reader) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Thread safety errors: {errors}"
