#!/usr/bin/env bash
# scripts/setup_farm.sh
# Run this ON THE LENOVO to set everything up

set -e

echo "🏠 Setting up Animus Farm on $(hostname)..."

# 1. Install Nix if not present
if ! command -v nix &> /dev/null; then
    echo "Installing Nix..."
    curl -L https://nixos.org/nix/install | sh
    source ~/.nix-profile/etc/profile.d/nix.sh
fi

# 2. Set up Directories
mkdir -p ~/Animus/manual_scripts
mkdir -p ~/Animus/models

# 3. Initialize Git Bare Repository for CI/CD
mkdir -p ~/Animus_bare.git
cd ~/Animus_bare.git
git init --bare

# 4. Create post-receive hook
cat <<EOF > hooks/post-receive
#!/usr/bin/env bash
echo "🚀 New code received. Deploying..."
GIT_WORK_TREE=/home/user/Animus git checkout -f
cd /home/user/Animus
nix develop --command cargo build --release
sudo systemctl restart animus
echo "✅ Animus updated and restarted."
EOF
chmod +x hooks/post-receive

# 5. Setup Systemd Service
sudo cp /home/user/Animus/scripts/animus.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable animus

echo "✅ Setup complete. You can now run 'git push farm master' from your dev machine."
EOF
