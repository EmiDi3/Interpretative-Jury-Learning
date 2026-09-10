#!/bin/bash
set -euo pipefail

echo "=== Claude Code Server Sandbox Setup ==="

# 1. Install sandbox dependencies (if running without Docker)
echo "[1/3] Installing sandbox dependencies..."
sudo apt-get update
sudo apt-get install -y bubblewrap socat

# 2. Deploy managed settings
echo "[2/3] Deploying managed settings..."
sudo mkdir -p /etc/claude-code
sudo cp managed-settings.json /etc/claude-code/managed-settings.json
sudo chmod 444 /etc/claude-code/managed-settings.json
sudo chown root:root /etc/claude-code/managed-settings.json

# 3. Verify
echo "[3/3] Verifying..."
echo "Managed settings installed at:"
ls -la /etc/claude-code/managed-settings.json

echo ""
echo "Done. Run 'claude /sandbox' to verify sandbox is active."
echo ""
echo "To use with Docker instead:"
echo "  export ANTHROPIC_API_KEY=sk-ant-..."
echo "  docker compose up --build"
