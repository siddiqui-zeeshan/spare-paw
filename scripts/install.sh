#!/usr/bin/env bash
# One-liner installer for spare-paw
#
# Usage:
#   curl -sSL https://raw.githubusercontent.com/siddiqui-zeeshan/spare-paw/main/scripts/install.sh | bash
#
# What it does:
#   1. Checks Python 3.11+ is available
#   2. Creates a venv at ~/.spare-paw/venv/
#   3. Installs spare-paw from GitHub
#   4. Links the CLI to ~/.local/bin/
#   5. Runs the setup wizard
#   6. Optionally sets up a systemd service (Linux only)

set -euo pipefail

REPO="https://github.com/siddiqui-zeeshan/spare-paw.git"
SPARE_DIR="$HOME/.spare-paw"
VENV_DIR="$SPARE_DIR/venv"
BIN_DIR="$HOME/.local/bin"

info()  { printf '\033[1;34m==>\033[0m %s\n' "$1"; }
warn()  { printf '\033[1;33mWARN:\033[0m %s\n' "$1"; }
error() { printf '\033[1;31mERROR:\033[0m %s\n' "$1" >&2; exit 1; }

# --- Platform detection ---

detect_platform() {
    if [ -d "/data/data/com.termux" ]; then
        echo "termux"
    elif [ "$(uname)" = "Darwin" ]; then
        echo "mac"
    else
        echo "linux"
    fi
}

PLATFORM=$(detect_platform)

# --- Check Python ---

find_python() {
    for cmd in python3.13 python3.12 python3.11 python3 python; do
        if command -v "$cmd" > /dev/null 2>&1; then
            local ver
            ver=$("$cmd" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
            local major minor
            major=$(echo "$ver" | cut -d. -f1)
            minor=$(echo "$ver" | cut -d. -f2)
            if [ "$major" -ge 3 ] && [ "$minor" -ge 11 ]; then
                echo "$cmd"
                return 0
            fi
        fi
    done
    return 1
}

info "Checking Python version..."
PYTHON=$(find_python) || error "Python 3.11+ is required. Install it first:
  Ubuntu/Debian: sudo apt install python3.11 python3.11-venv
  macOS:         brew install python@3.13
  Termux:        pkg install python"

PY_VERSION=$("$PYTHON" --version)
info "Found $PY_VERSION ($PYTHON)"

# --- Termux: skip venv, use system pip ---

if [ "$PLATFORM" = "termux" ]; then
    info "Termux detected — installing system-wide (no venv)"
    pip install --break-system-packages --upgrade \
        "spare-paw @ git+${REPO}" 2>&1 | tail -1

    info "Running setup wizard..."
    spare-paw setup

    echo ""
    info "Installation complete!"
    echo ""
    echo "  Start the bot:    spare-paw gateway"
    echo "  With watchdog:    spare-paw gateway & bash ~/.spare-paw/watchdog.sh"
    echo ""
    exit 0
fi

# --- Create venv ---

if [ -d "$VENV_DIR" ]; then
    info "Existing venv found — upgrading spare-paw..."
else
    info "Creating venv at $VENV_DIR..."
    "$PYTHON" -m venv "$VENV_DIR"
fi

# --- Install from GitHub ---

info "Installing spare-paw from GitHub..."
"$VENV_DIR/bin/pip" install --upgrade \
    "spare-paw @ git+${REPO}" 2>&1 | tail -1

# --- Link CLI to PATH ---

mkdir -p "$BIN_DIR"
ln -sf "$VENV_DIR/bin/spare-paw" "$BIN_DIR/spare-paw"

if ! echo "$PATH" | grep -q "$BIN_DIR"; then
    warn "$BIN_DIR is not in your PATH. Add it:"
    echo ""
    echo "  echo 'export PATH=\"\$HOME/.local/bin:\$PATH\"' >> ~/.bashrc && source ~/.bashrc"
    echo ""
fi

# --- Run setup wizard (skip if config exists) ---

if [ -f "$SPARE_DIR/config.yaml" ]; then
    info "Config already exists — skipping setup wizard"
else
    info "Running setup wizard..."
    "$VENV_DIR/bin/spare-paw" setup
fi

# --- Systemd service (Linux only) ---

if [ "$PLATFORM" = "linux" ]; then
    echo ""
    printf "Set up systemd service for auto-start? [y/N]: "
    read -r SETUP_SYSTEMD
    if [ "$SETUP_SYSTEMD" = "y" ] || [ "$SETUP_SYSTEMD" = "Y" ]; then
        SERVICE_FILE="$HOME/.config/systemd/user/spare-paw.service"
        mkdir -p "$(dirname "$SERVICE_FILE")"
        cat > "$SERVICE_FILE" << EOF
[Unit]
Description=spare-paw AI assistant
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
ExecStart=$VENV_DIR/bin/spare-paw gateway
Restart=always
RestartSec=5

[Install]
WantedBy=default.target
EOF
        systemctl --user daemon-reload
        systemctl --user enable spare-paw
        info "Systemd service created at $SERVICE_FILE"
        echo ""
        echo "  Start now:   systemctl --user start spare-paw"
        echo "  View logs:   journalctl --user -u spare-paw -f"
        echo "  Stop:        systemctl --user stop spare-paw"
        echo ""

        # Enable lingering so service runs without login
        if command -v loginctl > /dev/null 2>&1; then
            loginctl enable-linger "$(whoami)" 2>/dev/null || true
        fi
    fi
fi

# --- Done ---

echo ""
info "Installation complete!"
echo ""
echo "  Start the bot:     spare-paw gateway"
if [ "$PLATFORM" = "linux" ]; then
    echo "  Or with systemd:   systemctl --user start spare-paw"
fi
echo "  Upgrade later:     curl -sSL https://raw.githubusercontent.com/siddiqui-zeeshan/spare-paw/main/scripts/install.sh | bash"
echo ""
