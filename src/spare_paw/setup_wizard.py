"""Interactive setup wizard for first-time spare-paw configuration.

Prompts for API keys, writes config.yaml, and initializes the database.
No API validation in v1 — just writes the config and trusts the user.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

CLAW_DIR = Path.home() / ".spare-paw"
CONFIG_PATH = CLAW_DIR / "config.yaml"

CONFIG_TEMPLATE = """\
telegram:
  bot_token: "{bot_token}"
  owner_id: {owner_id}

openrouter:
  api_key: "{openrouter_key}"

models:
  default: "google/gemini-2.0-flash"
  smart: "anthropic/claude-sonnet-4"
  cron_default: "google/gemini-2.0-flash"

tavily:
  api_key: "{tavily_key}"

groq:
  api_key: "{groq_key}"

context:
  max_messages: 64
  token_budget: 120000
  safety_margin: 0.85

tools:
  shell:
    enabled: true
    timeout_seconds: 30
    max_output_chars: 10000
  files:
    enabled: true
    allowed_paths:
{allowed_paths}
  web_search:
    enabled: true
    max_results: 5
  web_scrape:
    enabled: true
    timeout_seconds: 15
    max_content_chars: 20000
  cron:
    enabled: true

agent:
  max_tool_iterations: 20
  system_prompt: |
    You are a personal AI assistant running 24/7 on an Android phone.
    You have access to the local filesystem, shell, web search, and web scraping.
    You can manage scheduled tasks (crons) for the user.
    Be concise. The user is on Telegram, likely on mobile.
    Current time: {{current_time}}
    Device: Android (Termux)

logging:
  level: "INFO"
  max_bytes: 10485760
  backup_count: 3
"""


def _detect_platform() -> str:
    """Detect the current platform: 'termux', 'mac', or 'linux'."""
    import os
    # Check for Termux
    if os.path.exists("/data/data/com.termux"):
        return "termux"
    if sys.platform == "darwin":
        return "mac"
    return "linux"


def _copy_defaults() -> None:
    """Copy default prompt files, selecting SYSTEM.md based on platform."""
    defaults_dir = Path(__file__).resolve().parent.parent.parent / "defaults"
    platform = _detect_platform()
    print(f"  Detected platform: {platform}")

    for filename in ("IDENTITY.md", "USER.md"):
        target = CLAW_DIR / filename
        if target.exists():
            continue
        source = defaults_dir / filename
        if source.exists():
            target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
            print(f"  Created {target}")

    # SYSTEM.md — pick platform-specific version
    target = CLAW_DIR / "SYSTEM.md"
    if not target.exists():
        system_file = {
            "termux": "SYSTEM.md",
            "mac": "SYSTEM.mac.md",
            "linux": "SYSTEM.linux.md",
        }.get(platform, "SYSTEM.md")
        source = defaults_dir / system_file
        if source.exists():
            target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
            print(f"  Created {target} (from {system_file})")


def _prompt_required(label: str) -> str:
    """Prompt until the user provides a non-empty value."""
    while True:
        value = input(f"  {label}: ").strip()
        if value:
            return value
        print("    This field is required. Please enter a value.")


def _prompt_int(label: str) -> int:
    """Prompt until the user provides a valid integer."""
    while True:
        value = input(f"  {label}: ").strip()
        try:
            return int(value)
        except ValueError:
            print("    Must be an integer. Please try again.")


def _prompt_optional(label: str) -> str:
    """Prompt for an optional value. Returns empty string if skipped."""
    value = input(f"  {label} (press Enter to skip): ").strip()
    return value


def run() -> None:
    """Run the interactive setup wizard."""
    print()
    print("=" * 52)
    print("    spare-paw setup wizard")
    print("=" * 52)
    print()
    print("This will create your configuration and database.")
    print()

    # 1. Create directories
    CLAW_DIR.mkdir(parents=True, exist_ok=True)
    (CLAW_DIR / "logs").mkdir(parents=True, exist_ok=True)
    (CLAW_DIR / "skills").mkdir(parents=True, exist_ok=True)
    (CLAW_DIR / "custom_tools").mkdir(parents=True, exist_ok=True)
    (CLAW_DIR / "custom_tools" / ".pending").mkdir(parents=True, exist_ok=True)
    print(f"Created {CLAW_DIR}/")

    # Copy default prompt files if they don't exist
    _copy_defaults()
    print()

    # 2. Check for existing config
    if CONFIG_PATH.exists():
        overwrite = input(f"  Config already exists at {CONFIG_PATH}. Overwrite? [y/N]: ").strip().lower()
        if overwrite != "y":
            print("  Keeping existing config.")
            print()
            # Still init DB in case it's missing
            _init_database()
            _print_success()
            return

    # 3. Collect credentials
    print("--- Required ---")
    print()
    bot_token = _prompt_required("Telegram bot token")
    owner_id = _prompt_int("Telegram owner ID (your numeric user ID)")
    openrouter_key = _prompt_required("OpenRouter API key")

    print()
    print("--- Optional ---")
    print()
    tavily_key = _prompt_optional("Tavily Search API key")
    groq_key = _prompt_optional("Groq API key (for voice messages)")

    # 4. Write config
    platform = _detect_platform()
    allowed_paths = {
        "termux": '      - "/sdcard"\n      - "/data/data/com.termux/files/home"',
        "mac": f'      - "{Path.home()}"',
        "linux": f'      - "{Path.home()}"',
    }.get(platform, f'      - "{Path.home()}"')

    config_content = CONFIG_TEMPLATE.format(
        bot_token=bot_token,
        owner_id=owner_id,
        openrouter_key=openrouter_key,
        tavily_key=tavily_key,
        groq_key=groq_key,
        allowed_paths=allowed_paths,
    )
    CONFIG_PATH.write_text(config_content, encoding="utf-8")
    # Restrict permissions — config contains secrets
    CONFIG_PATH.chmod(0o600)
    print()
    print(f"Config written to {CONFIG_PATH}")

    # 5. Initialize database
    _init_database()

    # 6. Success
    _print_success()


def _init_database() -> None:
    """Initialize the SQLite database synchronously."""
    from spare_paw.db import init_db

    print("Initializing database...")
    asyncio.run(init_db())
    print("Database ready.")


def _print_success() -> None:
    """Print success message and next steps."""
    print()
    print("=" * 52)
    print("    Setup complete!")
    print("=" * 52)
    print()
    print("To start spare-paw:")
    print()
    print("  python -m spare_paw gateway")
    print()
    print("Or with the watchdog (recommended for Termux):")
    print()
    print("  bash scripts/watchdog.sh")
    print()
