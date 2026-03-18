#!/bin/bash
# Install dependencies for spare-paw on Termux
#
# Run this script once after cloning the repo:
#   bash scripts/install-termux.sh

set -euo pipefail

echo "=== spare-paw Termux installer ==="
echo ""

# 1. Update Termux packages
echo "[1/4] Updating Termux packages..."
pkg update -y && pkg upgrade -y

# 2. Install system dependencies
echo "[2/4] Installing Python, pip, and git..."
pkg install -y python python-pip git

# 3. Install spare-paw in editable mode
echo "[3/4] Installing spare-paw and Python dependencies..."
pip install --break-system-packages -e .

# 4. Create runtime directory
echo "[4/4] Creating ~/.spare-paw directory..."
mkdir -p "$HOME/.spare-paw/logs"

echo ""
echo "=== Installation complete ==="
echo ""
echo "Next steps:"
echo "  1. Run the setup wizard:  python -m spare_paw setup"
echo "  2. Start the gateway:     python -m spare_paw gateway"
echo "  3. Or use the watchdog:   bash scripts/watchdog.sh"
echo ""
