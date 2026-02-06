#!/usr/bin/env bash
# deployment/deploy.sh
# Run this from your main machine to sync and restart the farm

SERVER_IP=$1
API_KEY=${ANIMUS_API_KEY:-"animus_dev_key"}

if [ -z "$SERVER_IP" ]; then
    echo "Usage: ./deploy.sh <server_ip>"
    exit 1
fi

echo "🚀 Syncing code to $SERVER_IP..."
rsync -avz --exclude 'target' --exclude '.git' --exclude '.env' --exclude 'manual_scripts/*.json' . user@$SERVER_IP:~/Animus/

echo "🔨 Building and restarting on server..."
ssh user@$SERVER_IP "cd ~/Animus && nix develop --command cargo build --release && sudo systemctl restart animus"

echo "✅ Deployment complete. Status:"
curl -s -H "X-Api-Key: $API_KEY" http://$SERVER_IP:8080/status | jq .
