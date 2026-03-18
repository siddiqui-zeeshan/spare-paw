"""Entry point for claw-phone: python -m claw_phone [setup|gateway]."""

import sys


USAGE = """\
Usage: python -m claw_phone <command>

Commands:
  setup     Run the interactive setup wizard
  gateway   Start the Telegram bot + scheduler (main loop)
"""


def main() -> None:
    if len(sys.argv) < 2:
        print(USAGE.strip())
        sys.exit(1)

    command = sys.argv[1].lower()

    if command == "setup":
        from claw_phone.setup_wizard import run

        run()
    elif command == "gateway":
        from claw_phone.gateway import run

        run()
    else:
        print(f"Unknown command: {command}\n")
        print(USAGE.strip())
        sys.exit(1)


if __name__ == "__main__":
    main()
