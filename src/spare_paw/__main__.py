"""Entry point for spare-paw: python -m spare_paw [setup|gateway]."""

import sys


USAGE = """\
Usage: python -m spare_paw <command>

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
        from spare_paw.setup_wizard import run

        run()
    elif command == "gateway":
        from spare_paw.gateway import run

        run()
    else:
        print(f"Unknown command: {command}\n")
        print(USAGE.strip())
        sys.exit(1)


if __name__ == "__main__":
    main()
