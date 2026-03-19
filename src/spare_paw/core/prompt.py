"""System prompt builder — leaf module.

Builds the full system prompt from config, prompt files (IDENTITY.md,
USER.md, SYSTEM.md), skills, and persistent memories.

INVARIANT: This module must NOT import from core/engine, router/, or
tools/subagent. It is a leaf in the dependency graph — both subagent.py
and cron/executor.py import from here without creating cycles.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_PROMPT_DIR = Path.home() / ".spare-paw"
_PROMPT_FILES = ["IDENTITY.md", "USER.md", "SYSTEM.md"]


async def build_system_prompt(config: Any) -> str:
    """Build the system prompt from config + markdown files + memories.

    Loads IDENTITY.md, USER.md, and SYSTEM.md (if they exist) and appends
    them to the base system prompt from config. Also injects all persistent
    memories. Files are re-read on every call so edits take effect without restart.
    """
    base = config.get("agent.system_prompt", "")
    current_time = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    base = base.replace("{current_time}", current_time)

    sections = [base]
    for filename in _PROMPT_FILES:
        path = _PROMPT_DIR / filename
        if path.is_file():
            try:
                content = path.read_text(encoding="utf-8").strip()
                if content:
                    sections.append(content)
            except OSError:
                logger.warning("Failed to read prompt file: %s", path)

    # Load skills from ~/.spare-paw/skills/
    skills_dir = _PROMPT_DIR / "skills"
    if skills_dir.is_dir():
        for skill_path in sorted(skills_dir.glob("*.md")):
            try:
                content = skill_path.read_text(encoding="utf-8").strip()
                if content:
                    sections.append(content)
            except OSError:
                logger.warning("Failed to read skill file: %s", skill_path)

    # Inject persistent memories
    try:
        from spare_paw.tools.memory import get_all_memories

        memories = await get_all_memories()
        if memories:
            mem_lines = [f"- {m['key']}: {m['value']}" for m in memories]
            sections.append("# Memories\n" + "\n".join(mem_lines))
    except Exception:
        logger.debug("Failed to load memories for system prompt", exc_info=True)

    return "\n\n".join(sections)
