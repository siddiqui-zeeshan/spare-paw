"""Dream consolidation engine — nightly knowledge extraction from conversations.

Runs as a cron job and consolidates knowledge from recent conversations into
topic-based markdown files at ~/.spare-paw/knowledge/. Knowledge files are
injected into the system prompt to give the agent persistent, organized context.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from spare_paw.config import resolve_model
from spare_paw.db import get_db

if TYPE_CHECKING:
    from spare_paw.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

KNOWLEDGE_DIR = Path.home() / ".spare-paw" / "knowledge"
DREAM_CRON_NAME = "dream_consolidation"

# -- Schemas ---------------------------------------------------------------

DREAM_CONSOLIDATE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {},
}

LIST_KNOWLEDGE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {},
}

# -- Consolidation prompt --------------------------------------------------

_CONSOLIDATION_PROMPT = """\
You are organizing knowledge from recent conversations into topic files.

## Existing knowledge files:
{existing_knowledge}

## Recent conversation excerpts (last 24 hours):
{recent_messages}

## Instructions:
Review the recent conversations and update the knowledge files. For each file \
you want to update, output:

### FILE: filename.md
[full updated content]

### FILE: INDEX.md
[updated index]

Rules:
- If new info contradicts old info, update the old (don't duplicate)
- Convert relative dates to absolute (e.g., "yesterday" -> "{yesterday_date}")
- Only store facts that would be useful in future conversations
- Keep each file under 2000 tokens
- Do not store ephemeral task details or debugging steps
- Standard topic files: user-preferences.md, active-projects.md, workflows.md, \
people.md, tech-stack.md (create only if relevant info exists)
- You may create additional topic files if needed
- Today's date is {today_date}
"""


# -- Core functions --------------------------------------------------------


def ensure_knowledge_dir() -> Path:
    """Create knowledge directory if it doesn't exist. Return the path."""
    KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)
    index = KNOWLEDGE_DIR / "INDEX.md"
    if not index.exists():
        index.write_text("# Knowledge Index\n", encoding="utf-8")
    return KNOWLEDGE_DIR


def get_knowledge_for_context(max_tokens: int = 2000) -> str:
    """Read all knowledge files and return as a formatted string for context injection.

    Cap at max_tokens total (estimated as chars / 4).
    """
    kdir = KNOWLEDGE_DIR
    if not kdir.is_dir():
        return ""

    sections: list[str] = []
    char_budget = max_tokens * 4  # rough token-to-char estimate
    chars_used = 0

    for path in sorted(kdir.glob("*.md")):
        if path.name == "INDEX.md":
            continue
        try:
            content = path.read_text(encoding="utf-8").strip()
            if not content:
                continue
            section = f"## {path.stem}\n{content}"
            section_len = len(section)
            if chars_used + section_len > char_budget:
                # Add truncated remainder if there's room for at least something
                remaining = char_budget - chars_used
                if remaining > 100:
                    sections.append(section[:remaining] + "\n[truncated]")
                break
            sections.append(section)
            chars_used += section_len
        except OSError:
            logger.warning("Failed to read knowledge file: %s", path)

    if not sections:
        return ""

    return "# Knowledge\n" + "\n\n".join(sections)


def get_selective_knowledge(query: str, max_files: int = 3) -> str:
    """Read INDEX.md, pick most relevant files based on keyword matching, return their content."""
    kdir = KNOWLEDGE_DIR
    index_path = kdir / "INDEX.md"
    if not index_path.is_file():
        return ""

    try:
        index_content = index_path.read_text(encoding="utf-8")
    except OSError:
        return ""

    # Parse index lines: "- filename.md — description"
    query_lower = query.lower()
    query_words = set(query_lower.split())

    scored_files: list[tuple[int, str]] = []
    for line in index_content.splitlines():
        line = line.strip()
        if not line.startswith("- "):
            continue
        # Extract filename
        parts = line[2:].split(" — ", 1)
        if not parts:
            continue
        filename = parts[0].strip()
        description = parts[1].strip() if len(parts) > 1 else ""

        # Score by keyword overlap
        text = (filename + " " + description).lower()
        score = sum(1 for w in query_words if w in text)
        if score > 0:
            scored_files.append((score, filename))

    # Sort by score descending, take top max_files
    scored_files.sort(key=lambda x: x[0], reverse=True)
    selected = [f for _, f in scored_files[:max_files]]

    if not selected:
        return ""

    sections: list[str] = []
    for filename in selected:
        path = kdir / filename
        if path.is_file():
            try:
                content = path.read_text(encoding="utf-8").strip()
                if content:
                    sections.append(f"## {path.stem}\n{content}")
            except OSError:
                pass

    return "\n\n".join(sections)


async def run_dream(app_state: Any) -> str:
    """Run dream consolidation: extract knowledge from recent conversations.

    Steps:
        1. Orient — read existing knowledge files
        2. Gather — fetch last 24h of messages from DB
        3. Build consolidation prompt
        4. Call LLM (summary model)
        5. Parse response and write files
        6. Update INDEX.md
        7. Return summary
    """
    # 1. Orient — ensure dir exists and read existing files
    kdir = ensure_knowledge_dir()

    existing_knowledge = _read_existing_knowledge(kdir)

    # 2. Gather — last 24 hours of messages
    messages = await _gather_recent_messages()

    if not messages:
        logger.info("Dream: no recent messages to consolidate")
        return "No recent messages to consolidate."

    # 3. Build consolidation prompt
    now = datetime.now(timezone.utc)
    today_date = now.strftime("%Y-%m-%d")
    yesterday_date = (now - timedelta(days=1)).strftime("%Y-%m-%d")

    formatted_messages = "\n".join(
        f"[{m['created_at']}] {m['role']}: {m['content']}" for m in messages
    )

    prompt = _CONSOLIDATION_PROMPT.format(
        existing_knowledge=existing_knowledge or "(no existing knowledge files)",
        recent_messages=formatted_messages,
        today_date=today_date,
        yesterday_date=yesterday_date,
    )

    # 4. Call the LLM (summary model — cheap)
    router_client = getattr(app_state, "router_client", None)
    if router_client is None:
        raise RuntimeError("Router client not available on app_state")

    model = resolve_model(app_state.config, "summary")

    response = await router_client.chat(
        messages=[{"role": "user", "content": prompt}],
        model=model,
    )
    llm_output = response["choices"][0]["message"]["content"]

    # 5. Parse response and write files
    files_written = _parse_and_write_files(kdir, llm_output)

    # 6. Return summary
    if files_written:
        summary = f"Dream consolidation complete. Updated {len(files_written)} file(s): {', '.join(files_written)}"
    else:
        summary = "Dream consolidation complete. No files updated."

    logger.info(summary)
    return summary


# -- Helpers ---------------------------------------------------------------


def _read_existing_knowledge(kdir: Path) -> str:
    """Read all existing knowledge files and return formatted string."""
    sections: list[str] = []
    for path in sorted(kdir.glob("*.md")):
        try:
            content = path.read_text(encoding="utf-8").strip()
            if content:
                sections.append(f"### {path.name}\n{content}")
        except OSError:
            logger.warning("Failed to read knowledge file: %s", path)
    return "\n\n".join(sections)


async def _gather_recent_messages(hours: int = 24) -> list[dict[str, str]]:
    """Fetch messages from the last N hours."""
    db = await get_db()
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()

    cursor = await db.execute(
        """SELECT role, content, created_at FROM messages
           WHERE created_at >= ? AND role IN ('user', 'assistant')
           ORDER BY created_at ASC
           LIMIT 500""",
        (cutoff,),
    )
    rows = await cursor.fetchall()
    return [
        {"role": r["role"], "content": r["content"], "created_at": r["created_at"]}
        for r in rows
    ]


def _parse_and_write_files(kdir: Path, llm_output: str) -> list[str]:
    """Parse LLM output for ### FILE: blocks and write them."""
    files_written: list[str] = []
    current_file: str | None = None
    current_lines: list[str] = []

    for line in llm_output.splitlines():
        if line.startswith("### FILE:"):
            # Write previous file if any
            if current_file and current_lines:
                _write_knowledge_file(kdir, current_file, "\n".join(current_lines))
                files_written.append(current_file)
            # Start new file
            current_file = line.split("### FILE:", 1)[1].strip()
            current_lines = []
        elif current_file is not None:
            current_lines.append(line)

    # Write last file
    if current_file and current_lines:
        _write_knowledge_file(kdir, current_file, "\n".join(current_lines))
        files_written.append(current_file)

    return files_written


def _write_knowledge_file(kdir: Path, filename: str, content: str) -> None:
    """Write a knowledge file, sanitizing the filename."""
    # Sanitize: only allow simple filenames
    safe_name = filename.strip().replace("/", "").replace("\\", "")
    if not safe_name.endswith(".md"):
        safe_name += ".md"
    path = kdir / safe_name
    path.write_text(content.strip() + "\n", encoding="utf-8")
    logger.debug("Wrote knowledge file: %s", path)


# -- Tool handlers ---------------------------------------------------------


async def _handle_dream_consolidate(app_state: Any) -> str:
    """Handler for the dream_consolidate tool."""
    result = await run_dream(app_state)
    return json.dumps({"status": "ok", "summary": result})


async def _handle_list_knowledge() -> str:
    """Handler for the list_knowledge tool."""
    kdir = KNOWLEDGE_DIR
    index_path = kdir / "INDEX.md"
    if not index_path.is_file():
        return json.dumps({"index": "(no knowledge directory)", "files": []})

    index_content = index_path.read_text(encoding="utf-8")
    files = [p.name for p in sorted(kdir.glob("*.md")) if p.name != "INDEX.md"]
    return json.dumps({"index": index_content, "files": files})


# -- Registration ----------------------------------------------------------


def register(registry: ToolRegistry, config: dict[str, Any], state: Any = None) -> None:
    """Register dream consolidation tools."""

    async def _dream_handler() -> str:
        return await _handle_dream_consolidate(state)

    registry.register(
        name="dream_consolidate",
        description=(
            "Run dream consolidation: review recent conversations and extract "
            "knowledge into persistent topic files. Normally runs as a nightly cron."
        ),
        parameters_schema=DREAM_CONSOLIDATE_SCHEMA,
        handler=_dream_handler,
        run_in_executor=False,
    )

    registry.register(
        name="list_knowledge",
        description="List all knowledge files from the dream consolidation system.",
        parameters_schema=LIST_KNOWLEDGE_SCHEMA,
        handler=_handle_list_knowledge,
        run_in_executor=False,
    )
