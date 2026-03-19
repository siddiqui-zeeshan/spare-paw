"""Tests for core/engine.py — platform-agnostic message processor."""

from __future__ import annotations

import ast
import inspect

from spare_paw.core.engine import split_text


class TestSplitText:
    def test_short_text_single_chunk(self):
        assert split_text("hello", 100) == ["hello"]

    def test_exact_length(self):
        text = "a" * 50
        assert split_text(text, 50) == [text]

    def test_splits_at_newline(self):
        text = "line1\nline2\nline3"
        chunks = split_text(text, 12)
        assert all(len(c) <= 12 for c in chunks)
        assert "line1" in chunks[0]

    def test_hard_cut_when_no_newline(self):
        text = "a" * 100
        chunks = split_text(text, 30)
        assert all(len(c) <= 30 for c in chunks)
        assert "".join(chunks) == text

    def test_preserves_all_content(self):
        text = "first\nsecond\nthird\nfourth"
        chunks = split_text(text, 10)
        joined = "\n".join(chunks)
        assert "first" in joined
        assert "second" in joined
        assert "third" in joined
        assert "fourth" in joined

    def test_empty_string(self):
        assert split_text("", 100) == [""]

    def test_multiple_newlines_stripped(self):
        """Leading newlines on subsequent chunks should be stripped."""
        text = "aaa\n\n\nbbb"
        chunks = split_text(text, 5)
        for chunk in chunks:
            assert not chunk.startswith("\n")


class TestNoTelegramImport:
    def test_no_telegram_import_in_engine(self):
        import spare_paw.core.engine as mod

        source = inspect.getsource(mod)
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert not alias.name.startswith("telegram"), (
                        f"Found 'import {alias.name}' in core/engine.py"
                    )
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    assert not node.module.startswith("telegram"), (
                        f"Found 'from {node.module}' in core/engine.py"
                    )
