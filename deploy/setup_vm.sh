#!/bin/bash
#
# GCP VM Setup Script for Twitter Influencer Bot
# ==============================================
#
# This script sets up a GCP VM with:
# - Discord bot for rankings & Mastodon approval
# - Chrome Remote Desktop for headful browser automation
# - LinkedIn Ranking UI web interface
# - Cloudflare Tunnel for secure HTTPS access
#
# Prerequisites:
#   - gcloud CLI installed and authenticated
#   - GCP project 'eagleeye' exists
#
# Usage:
#   ./deploy/setup_vm.sh
#

set -e

# Configuration
PROJECT_ID="eagleeye"
ZONE="us-central1-a"
INSTANCE_NAME="twitter-influencer"
# e2-standard-2: 2 vCPU, 8GB RAM (~$49/month)
# Required for: Chrome Remote Desktop + headful Chrome + ML models
# e2-small (2GB) is NOT enough - will OOM with Chrome
MACHINE_TYPE="e2-standard-2"
BOOT_DISK_SIZE="30GB"

echo "=== GCP VM Setup for Twitter Influencer Bot ==="
echo ""

# Step 1: Set project
echo "[1/8] Setting GCP project..."
gcloud config set project "$PROJECT_ID"

# Step 2: Create VM
echo "[2/8] Creating VM instance..."
if gcloud compute instances describe "$INSTANCE_NAME" --zone="$ZONE" &>/dev/null; then
    echo "  VM '$INSTANCE_NAME' already exists. Skipping creation."
else
    gcloud compute instances create "$INSTANCE_NAME" \
        --zone="$ZONE" \
        --machine-type="$MACHINE_TYPE" \
        --image-family=ubuntu-2204-lts \
        --image-project=ubuntu-os-cloud \
        --boot-disk-size="$BOOT_DISK_SIZE" \
        --boot-disk-type=pd-standard \
        --tags=discord-bot,http-server,https-server
    echo "  VM created successfully."
fi

# Step 3: Wait for VM to be ready
echo "[3/8] Waiting for VM to be ready..."
sleep 30

# Step 4: Install system dependencies + desktop environment
echo "[4/8] Installing dependencies on VM (this takes a few minutes)..."
gcloud compute ssh "$INSTANCE_NAME" --zone="$ZONE" --command="
    sudo apt-get update

    # Install Python, git, Chrome, and desktop environment
    sudo apt-get install -y \
        python3.12 \
        python3.12-venv \
        git \
        chromium-browser \
        chromium-chromedriver \
        sqlite3 \
        curl \
        unzip \
        xfce4 \
        xfce4-goodies \
        fail2ban

    # Install Chrome Remote Desktop
    wget -q https://dl.google.com/linux/direct/chrome-remote-desktop_current_amd64.deb
    sudo dpkg -i chrome-remote-desktop_current_amd64.deb || sudo apt-get -f install -y
    rm chrome-remote-desktop_current_amd64.deb

    # Configure Chrome Remote Desktop session
    cat > ~/.chrome-remote-desktop-session << 'XFCE'
#!/bin/bash
exec /usr/bin/xfce4-session
XFCE
    chmod +x ~/.chrome-remote-desktop-session

    # Install cloudflared for Cloudflare Tunnel
    curl -L https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb -o /tmp/cloudflared.deb
    sudo dpkg -i /tmp/cloudflared.deb
    rm /tmp/cloudflared.deb

    # Install uv if not present
    if ! command -v uv &>/dev/null; then
        curl -LsSf https://astral.sh/uv/install.sh | sh
        echo 'export PATH=\"\$HOME/.local/bin:\$PATH\"' >> ~/.bashrc
    fi

    # Create logs directory
    mkdir -p ~/logs

    # Enable fail2ban for SSH protection
    sudo systemctl enable fail2ban
    sudo systemctl start fail2ban

    echo 'Dependencies installed.'
"

# Step 5: Clone/update repository
echo "[5/8] Setting up repository..."
gcloud compute ssh "$INSTANCE_NAME" --zone="$ZONE" --command="
    cd ~
    if [ -d 'twitter_influencer' ]; then
        cd twitter_influencer
        git pull
    else
        git clone https://github.com/YOUR_USERNAME/twitter_influencer.git
        cd twitter_influencer
    fi

    # Sync dependencies
    ~/.local/bin/uv sync

    echo 'Repository setup complete.'
"

# Step 6: Create wrapper scripts for cron
echo "[6/8] Creating wrapper scripts..."
gcloud compute ssh "$INSTANCE_NAME" --zone="$ZONE" --command="
    USER=\$(whoami)
    HOME_DIR=/home/\$USER

    # Scraper wrapper (needs DISPLAY for headful Chrome)
    cat > ~/run_scraper.sh << 'WRAPPER'
#!/bin/bash
export DISPLAY=:20
export HOME=/home/\$(whoami)
cd ~/twitter_influencer
~/.local/bin/uv run python run_scrapers.py >> ~/logs/scrape.log 2>&1
WRAPPER
    chmod +x ~/run_scraper.sh

    # Ranking wrapper (needs DISPLAY for headful Chrome)
    cat > ~/run_ranking.sh << 'WRAPPER'
#!/bin/bash
export DISPLAY=:20
export HOME=/home/\$(whoami)
cd ~/twitter_influencer
~/.local/bin/uv run python linkedin_autopilot.py --rank >> ~/logs/rank.log 2>&1
WRAPPER
    chmod +x ~/run_ranking.sh

    echo 'Wrapper scripts created.'
"

# Step 7: Setup systemd services
echo "[7/8] Setting up systemd services..."
gcloud compute ssh "$INSTANCE_NAME" --zone="$ZONE" --command="
    USER=\$(whoami)
    HOME_DIR=/home/\$USER

    # Discord Bot Service
    sudo tee /etc/systemd/system/discord-bot.service > /dev/null << EOF
[Unit]
Description=Twitter Influencer Discord Bot
After=network.target

[Service]
Type=simple
User=\$USER
WorkingDirectory=\$HOME_DIR/twitter_influencer
EnvironmentFile=\$HOME_DIR/twitter_influencer/.env
ExecStart=\$HOME_DIR/.local/bin/uv run python discord_bot.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

    # LinkedIn Ranking UI Service
    sudo tee /etc/systemd/system/linkedin-ui.service > /dev/null << EOF
[Unit]
Description=LinkedIn Ranking UI
After=network.target

[Service]
Type=simple
User=\$USER
WorkingDirectory=\$HOME_DIR/twitter_influencer
Environment=PATH=\$HOME_DIR/.local/bin:/usr/bin
Environment=DISPLAY=:20
EnvironmentFile=\$HOME_DIR/twitter_influencer/.env
ExecStart=\$HOME_DIR/.local/bin/uv run python linkedin_ranking_ui.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

    # Enable services (don't start yet - need .env and Chrome Remote Desktop)
    sudo systemctl daemon-reload
    sudo systemctl enable discord-bot
    sudo systemctl enable linkedin-ui

    echo 'Systemd services configured.'
"

# Step 8: Setup GCP Firewall (allow only SSH by default)
echo "[8/8] Configuring firewall..."
# Check if firewall rule exists
if ! gcloud compute firewall-rules describe allow-ssh-ingress --project="$PROJECT_ID" &>/dev/null; then
    gcloud compute firewall-rules create allow-ssh-ingress \
        --project="$PROJECT_ID" \
        --direction=INGRESS \
        --priority=1000 \
        --network=default \
        --action=ALLOW \
        --rules=tcp:22 \
        --source-ranges=0.0.0.0/0 \
        --target-tags=discord-bot
    echo "  Firewall rule created."
else
    echo "  Firewall rule already exists."
fi

echo ""
echo "=============================================="
echo "=== Setup Complete ==="
echo "=============================================="
echo ""
echo "NEXT STEPS:"
echo ""
echo "1. SSH into VM:"
echo "   gcloud compute ssh $INSTANCE_NAME --zone=$ZONE"
echo ""
echo "2. Setup Chrome Remote Desktop:"
echo "   - Go to: https://remotedesktop.google.com/headless"
echo "   - Click 'Set up via SSH' > 'Begin' > 'Next' > 'Authorize'"
echo "   - Copy the Debian Linux command and run it on the VM"
echo "   - Set a PIN when prompted"
echo ""
echo "3. Create .env file:"
echo "   cat > ~/twitter_influencer/.env << 'EOF'"
echo "   # OpenRouter (replaces Claude CLI)"
echo "   OPENROUTER_API_KEY=sk-or-..."
echo "   OPENROUTER_MODEL=anthropic/claude-sonnet-4"
echo ""
echo "   # Discord Bot"
echo "   DISCORD_BOT_TOKEN=..."
echo ""
echo "   # Mastodon"
echo "   MASTODON_ACCESS_TOKEN=..."
echo "   MASTODON_API_BASE=https://mastodon.social"
echo ""
echo "   # Twitter/X (for scraping)"
echo "   TWITTER_USERNAME=..."
echo "   TWITTER_PASSWORD=..."
echo ""
echo "   # LinkedIn (for posting)"
echo "   LINKEDIN_EMAIL=..."
echo "   LINKEDIN_PASSWORD=..."
echo "   EOF"
echo "   chmod 600 ~/twitter_influencer/.env"
echo ""
echo "4. Run database migration:"
echo "   cd ~/twitter_influencer && uv run python db_migrations.py"
echo ""
echo "5. First-time logins (via Chrome Remote Desktop):"
echo "   - Connect via https://remotedesktop.google.com"
echo "   - Open terminal and run:"
echo "     cd ~/twitter_influencer"
echo "     uv run python ai_news_scraper.py  # Login to Twitter"
echo "     uv run python linkedin_autopilot.py --post  # Login to LinkedIn"
echo "   - Cookies are cached for future runs"
echo ""
echo "6. Start services:"
echo "   sudo systemctl start discord-bot"
echo "   sudo systemctl start linkedin-ui"
echo "   sudo systemctl status discord-bot linkedin-ui"
echo ""
echo "7. Setup cron jobs:"
echo "   crontab -e"
echo "   # Add these lines:"
echo "   0 8 * * * ~/run_scraper.sh"
echo "   0 20 * * * ~/run_scraper.sh"
echo "   0 9 * * * ~/run_ranking.sh"
echo ""
echo "8. (Optional) Setup Cloudflare Tunnel for HTTPS access:"
echo "   cloudflared tunnel login"
echo "   cloudflared tunnel create ai-influencer"
echo "   # See DEPLOY_HETZNER.md for full tunnel config"
echo ""
echo "=============================================="
echo "SERVICES:"
echo "  - Discord Bot: sudo systemctl status discord-bot"
echo "  - Ranking UI:  sudo systemctl status linkedin-ui (port 5001)"
echo ""
echo "LOGS:"
echo "  - Discord Bot: sudo journalctl -u discord-bot -f"
echo "  - Ranking UI:  sudo journalctl -u linkedin-ui -f"
echo "  - Scrapers:    tail -f ~/logs/scrape.log"
echo "  - Rankings:    tail -f ~/logs/rank.log"
echo ""
echo "COST: e2-standard-2 = ~\$49/month (2 vCPU, 8GB RAM)"
echo "=============================================="
