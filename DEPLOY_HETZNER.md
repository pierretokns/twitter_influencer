# Hetzner Cloud Deployment Guide

Deploy the LinkedIn AI Influencer system to a Hetzner Cloud VM with desktop GUI for headful browser automation.

## VM Recommendation

**CX32** - €6.80/month (~$7.50 USD) - RECOMMENDED for GUI
- 4 vCPU (Intel shared)
- 8 GB RAM (needed for desktop + Chrome + ML models)
- 80 GB NVMe SSD
- 20 TB traffic

Why CX32 over CX22:
- Desktop environment (XFCE) needs ~1-2GB RAM
- Chrome headful needs more RAM than headless
- Sentence-transformers model needs ~500MB
- More comfortable for interactive use

## Step 1: Create Hetzner VM

1. Go to [Hetzner Cloud Console](https://console.hetzner.cloud/)
2. Create new project → "linkedin-influencer"
3. Add Server:
   - Location: Nuremberg (nbg1) or Falkenstein (fsn1) - cheapest
   - Image: **Ubuntu 24.04**
   - Type: **CX32** (€6.80/mo) - for GUI support
   - SSH Key: Add your public key
4. Create & note the IP address

## Step 2: Initial Server Setup + Desktop Environment

```bash
# SSH into server
ssh root@YOUR_SERVER_IP

# Update system
apt update && apt upgrade -y

# Install XFCE desktop environment (lightweight)
apt install -y xfce4 xfce4-goodies

# Install VNC server for remote desktop access
apt install -y tigervnc-standalone-server tigervnc-common

# Install display manager (optional, for auto-login)
apt install -y lightdm

# Install dependencies
apt install -y \
    python3.12 \
    python3.12-venv \
    python3-pip \
    git \
    chromium-browser \
    chromium-chromedriver \
    sqlite3 \
    curl \
    unzip \
    nodejs \
    npm

# Create app user
useradd -m -s /bin/bash appuser
echo "appuser:YOUR_SECURE_PASSWORD" | chpasswd
usermod -aG sudo appuser
```

## Step 3: Setup VNC for Remote Desktop Access

```bash
# Switch to appuser
su - appuser

# Set VNC password
vncpasswd
# Enter a password (you'll use this to connect)

# Create VNC startup script
mkdir -p ~/.vnc
cat > ~/.vnc/xstartup << 'EOF'
#!/bin/bash
unset SESSION_MANAGER
unset DBUS_SESSION_BUS_ADDRESS
exec startxfce4
EOF
chmod +x ~/.vnc/xstartup

# Start VNC server (run this each time or setup systemd)
vncserver :1 -geometry 1920x1080 -depth 24

# VNC will be available on port 5901
```

### Connect to VNC

**Option A: SSH Tunnel (Secure - Recommended)**
```bash
# On your LOCAL machine, create SSH tunnel:
ssh -L 5901:localhost:5901 appuser@YOUR_SERVER_IP

# Then connect VNC client to: localhost:5901
```

**Option B: Direct Connection (Less secure)**
```bash
# On server, allow VNC port through firewall
sudo ufw allow 5901/tcp

# Connect VNC client to: YOUR_SERVER_IP:5901
```

**VNC Clients:**
- macOS: Built-in Screen Sharing (vnc://localhost:5901)
- Windows: RealVNC, TightVNC
- Linux: Remmina, TigerVNC viewer

## Step 4: Install UV and Clone Repo (In VNC Desktop)

Open a terminal in the VNC desktop session:

```bash
cd ~

# Install uv (fast Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc

# Clone the repository
git clone https://github.com/pierretokns/twitter_influencer.git
cd twitter_influencer
git checkout claude/cloudflare-workers-ai-exploration-015SWzxnn3Nw8ZnbrGus1xMF

# Install dependencies
uv sync
```

## Step 5: Install and Login to Claude CLI

Claude CLI uses `/login` for authentication via browser OAuth.

```bash
# Install Claude CLI globally
sudo npm install -g @anthropic-ai/claude-code

# Login via browser (MUST be done in VNC desktop session!)
claude /login
```

This will:
1. Open Chromium browser
2. Redirect to Anthropic login page
3. You authenticate with your Anthropic account
4. CLI stores credentials locally

**Test it works:**
```bash
claude -p "Say hello"
```

**Re-login if needed:**
```bash
claude /login
```

## Step 5b: Configure Environment Variables

```bash
cd ~/twitter_influencer

# Create .env file
cat > .env << 'EOF'
# Twitter/X Authentication (for scraping bookmarks)
TWITTER_USERNAME=your_twitter_username
TWITTER_PASSWORD=your_twitter_password
# OR use Google auth:
GOOGLE_EMAIL=your_google_email@gmail.com

# LinkedIn Authentication (for posting)
LINKEDIN_EMAIL=your_linkedin_email
LINKEDIN_PASSWORD=your_linkedin_password

# Optional: OpenAI for image generation
OPENAI_API_KEY=sk-...
EOF

chmod 600 .env
```

## Step 6: First-Time Login & Cookie Caching (In VNC Desktop)

All browser automation runs headful in the VNC desktop session. This is more reliable than headless mode.

### Twitter Login (First Time)

Open a terminal in your VNC desktop session:

```bash
cd ~/twitter_influencer

# Run scraper - Chrome will open in VNC desktop
uv run python ai_news_scraper.py --google-auth your_email@gmail.com

# Complete Google login in the browser window
# Cookies are saved to: output_data/chrome_profile/
```

### LinkedIn Login (First Time)

```bash
cd ~/twitter_influencer

# Run LinkedIn poster - Chrome will open in VNC desktop
uv run python linkedin_autopilot.py --post

# Complete login in the browser window
# Cookies saved to: output_data/chrome_profile/
```

After initial login, cookies are cached and you won't need to login again (until cookies expire).

## Step 7: Verify Everything Works (In VNC Desktop)

```bash
cd ~/twitter_influencer

# Test news scraper (Chrome opens in VNC desktop)
uv run python ai_news_scraper.py

# Check database has content
sqlite3 output_data/ai_news.db "SELECT COUNT(*) FROM tweets WHERE is_ai_relevant=1"

# Test ranking UI
uv run python linkedin_ranking_ui.py &
curl http://localhost:5001/api/status

# Test Claude CLI works
claude -p "Say hello"
```

## Step 8: Setup Cron Jobs (Twice Daily Scraping)

For headful browser automation, cron jobs need access to the VNC display.

First, create a wrapper script:
```bash
cat > ~/run_scraper.sh << 'EOF'
#!/bin/bash
export DISPLAY=:1
export HOME=/home/appuser
cd /home/appuser/twitter_influencer
/home/appuser/.local/bin/uv run python ai_news_scraper.py >> /home/appuser/logs/scrape.log 2>&1
EOF
chmod +x ~/run_scraper.sh

cat > ~/run_ranking.sh << 'EOF'
#!/bin/bash
export DISPLAY=:1
export HOME=/home/appuser
cd /home/appuser/twitter_influencer
/home/appuser/.local/bin/uv run python linkedin_autopilot.py --rank >> /home/appuser/logs/rank.log 2>&1
EOF
chmod +x ~/run_ranking.sh
```

Then setup crontab:
```bash
crontab -e

# Add these lines (scrape at 8am and 8pm UTC):
# ┌───────────── minute (0-59)
# │ ┌───────────── hour (0-23)
# │ │ ┌───────────── day of month (1-31)
# │ │ │ ┌───────────── month (1-12)
# │ │ │ │ ┌───────────── day of week (0-6)
# │ │ │ │ │
# * * * * *

# Morning scrape (8am UTC) - uses VNC display
0 8 * * * /home/appuser/run_scraper.sh

# Evening scrape (8pm UTC)
0 20 * * * /home/appuser/run_scraper.sh

# Optional: Generate and rank posts daily at 9am UTC
0 9 * * * /home/appuser/run_ranking.sh
```

Create log directory:
```bash
mkdir -p ~/logs
```

**Important:** VNC server must be running for cron jobs to work. See Step 9 for auto-starting VNC.

## Step 9: Auto-Start VNC Server on Boot

Create a systemd service to auto-start VNC on boot:

```bash
sudo cat > /etc/systemd/system/vncserver@.service << 'EOF'
[Unit]
Description=VNC Server for display %i
After=syslog.target network.target

[Service]
Type=forking
User=appuser
Group=appuser
WorkingDirectory=/home/appuser

ExecStartPre=/bin/sh -c '/usr/bin/vncserver -kill :%i > /dev/null 2>&1 || :'
ExecStart=/usr/bin/vncserver :%i -geometry 1920x1080 -depth 24
ExecStop=/usr/bin/vncserver -kill :%i

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable vncserver@1
sudo systemctl start vncserver@1

# Check status
sudo systemctl status vncserver@1
```

## Step 10: Setup Systemd Service (Optional - for UI)

If you want the ranking UI to run continuously:

```bash
sudo cat > /etc/systemd/system/linkedin-ui.service << 'EOF'
[Unit]
Description=LinkedIn Ranking UI
After=network.target vncserver@1.service

[Service]
Type=simple
User=appuser
WorkingDirectory=/home/appuser/twitter_influencer
Environment=PATH=/home/appuser/.local/bin:/usr/bin
Environment=DISPLAY=:1
ExecStart=/home/appuser/.local/bin/uv run python linkedin_ranking_ui.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable linkedin-ui
sudo systemctl start linkedin-ui

# Check status
sudo systemctl status linkedin-ui
```

## Step 11: Firewall Setup (Optional)

```bash
# Allow SSH and UI port
sudo ufw allow 22/tcp
sudo ufw allow 5001/tcp  # Ranking UI
sudo ufw enable
```

## Maintenance Commands

```bash
# View scraping logs
tail -f ~/logs/scrape.log

# Check cron is running
grep CRON /var/log/syslog | tail -20

# Manual scrape (in VNC desktop)
cd ~/twitter_influencer && uv run python ai_news_scraper.py

# Check database stats
sqlite3 output_data/ai_news.db "
SELECT 'Tweets:' as type, COUNT(*) as count FROM tweets WHERE is_ai_relevant=1
UNION ALL
SELECT 'Articles:', COUNT(*) FROM web_articles WHERE is_ai_relevant=1;
"

# Update code from git
cd ~/twitter_influencer && git pull && uv sync

# Restart services after update
sudo systemctl restart linkedin-ui
sudo systemctl restart vncserver@1

# Check VNC is running
sudo systemctl status vncserver@1
```

## Troubleshooting

### VNC not starting
```bash
# Check VNC service status
sudo systemctl status vncserver@1

# Manual restart
sudo systemctl restart vncserver@1

# Check logs
journalctl -u vncserver@1 -n 50
```

### Chrome not opening in VNC
```bash
# Ensure DISPLAY is set
echo $DISPLAY  # Should show :1

# Set manually if needed
export DISPLAY=:1

# Test with simple app
xterm &
```

### Cookie session expired
```bash
# Connect to VNC desktop and re-run login
cd ~/twitter_influencer
uv run python ai_news_scraper.py --google-auth your_email@gmail.com
```

### Claude CLI not working
```bash
# Test directly
claude -p "Hello"

# Re-authenticate (opens browser in VNC desktop)
claude /login

# Check Claude is installed
which claude
```

### Cron jobs not running
```bash
# Ensure VNC is running (cron jobs need DISPLAY=:1)
sudo systemctl status vncserver@1

# Check cron logs
grep CRON /var/log/syslog | tail -20

# Test wrapper script manually
./run_scraper.sh
```

### Out of disk space
```bash
# Check usage
df -h

# Clean old logs
rm ~/logs/*.log.old

# Vacuum database
sqlite3 output_data/ai_news.db "VACUUM;"
```

## Cost Summary

| Component | Monthly Cost |
|-----------|-------------|
| Hetzner CX32 | €6.80 (~$7.50) |
| Claude CLI | Included with Claude subscription |
| OpenAI API (optional, for images) | ~$2-5 |
| **Total** | **~€7-12/month** |

Note: Claude CLI uses your Claude.ai subscription (Pro or Max), not API credits.

## Step 12: Security Hardening (DISA STIG)

Apply military-grade security hardening using DISA STIG (Security Technical Implementation Guides) - the DoD-approved security standard.

```bash
# Install Ansible
sudo apt install -y ansible git

# Clone the STIG role (Ubuntu 22.04 - compatible with 24.04)
cd /tmp
git clone https://github.com/ansible-lockdown/UBUNTU22-STIG.git
cd UBUNTU22-STIG

# Create inventory file
echo "localhost ansible_connection=local" > inventory

# Create playbook
cat > harden.yml << 'EOF'
---
- name: Apply DISA STIG to Ubuntu
  hosts: localhost
  become: yes
  vars:
    # Customize for our use case (GUI + VNC needed)
    ubtu22stig_gui: true
    ubtu22stig_disruption_high: false
    # Keep SSH access
    ubtu22stig_ssh_required: true
    # Skip controls that break VNC/Chrome
    ubtu22stig_skip_for_travis: false
  roles:
    - UBUNTU22-STIG
EOF

# Run the hardening (takes 10-20 minutes)
sudo ansible-playbook -i inventory harden.yml --check  # Dry run first
sudo ansible-playbook -i inventory harden.yml          # Apply

# Reboot to apply kernel parameters
sudo reboot
```

### Post-Hardening Verification

```bash
# Verify SSH still works (test from another terminal before disconnecting!)
ssh appuser@YOUR_SERVER_IP

# Check VNC still works
sudo systemctl status vncserver@1

# Run a compliance scan (optional)
sudo apt install -y lynis
sudo lynis audit system

# Check security score
sudo lynis show details
```

### Important Notes

1. **Test in dry-run mode first** (`--check` flag)
2. **Keep an SSH session open** while applying - don't lock yourself out
3. **VNC access**: Always use SSH tunnel (port 5901 is blocked by STIG):
   ```bash
   ssh -L 5901:localhost:5901 appuser@YOUR_SERVER_IP
   # Then connect VNC client to localhost:5901
   ```

## Security Notes

1. Never commit `.env` to git
2. Use SSH keys, not passwords
3. Keep system updated: `apt update && apt upgrade`
4. Consider using fail2ban for SSH protection:
   ```bash
   sudo apt install -y fail2ban
   sudo systemctl enable fail2ban
   ```
5. Backup your database: `cp output_data/ai_news.db ~/backups/`
6. Run periodic security audits: `sudo lynis audit system`
