#!/usr/bin/env bash
# Animus Deployment Script for Server

echo "🚀 Updating Animus..."

# Pull latest changes
git pull origin master

# Build release binary
echo "🛠️ Building release..."
cargo build --release

# Restart systemd service
echo "🔄 Restarting daemon..."
sudo systemctl restart animus

# Show status
sudo systemctl status animus --no-pager

echo "✨ Done! View logs with: journalctl -u animus -f"
