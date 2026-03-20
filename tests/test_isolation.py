"""Import isolation check — no telegram imports outside bot/."""

from __future__ import annotations

from pathlib import Path


def _get_python_files():
    """Return all .py files in src/spare_paw/ excluding bot/."""
    src = Path(__file__).parent.parent / "src" / "spare_paw"
    files = []
    for p in src.rglob("*.py"):
        # Skip bot/ package entirely — it's allowed to import telegram
        rel = p.relative_to(src)
        if rel.parts and rel.parts[0] == "bot":
            continue
        files.append(p)
    return files


class TestTelegramIsolation:
    def test_no_telegram_imports_outside_bot(self):
        """No file outside bot/ should import from telegram."""
        violations = []
        for path in _get_python_files():
            try:
                source = path.read_text(encoding="utf-8")
            except OSError:
                continue
            for i, line in enumerate(source.splitlines(), 1):
                stripped = line.strip()
                if stripped.startswith("#"):
                    continue
                if "import telegram" in stripped or "from telegram" in stripped:
                    rel = path.relative_to(path.parent.parent.parent)
                    violations.append(f"{rel}:{i}: {stripped}")

        assert not violations, (
            "Found telegram imports outside bot/:\n" + "\n".join(violations)
        )

    def test_core_has_no_platform_imports(self):
        """core/ must not import from bot/, webhook/, or telegram."""
        src = Path(__file__).parent.parent / "src" / "spare_paw" / "core"
        violations = []
        for path in src.rglob("*.py"):
            try:
                source = path.read_text(encoding="utf-8")
            except OSError:
                continue
            for i, line in enumerate(source.splitlines(), 1):
                stripped = line.strip()
                if stripped.startswith("#"):
                    continue
                for forbidden in ["from spare_paw.bot", "import spare_paw.bot",
                                  "from spare_paw.webhook", "import spare_paw.webhook",
                                  "from telegram", "import telegram"]:
                    if forbidden in stripped:
                        violations.append(f"{path.name}:{i}: {stripped}")

        assert not violations, (
            "Found platform imports in core/:\n" + "\n".join(violations)
        )
