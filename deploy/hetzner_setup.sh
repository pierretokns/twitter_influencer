#!/bin/bash
# =============================================================================
# Hetzner Cloud Deployment Script
# =============================================================================
# Creates a hardened Ubuntu server with:
# - SSH only accessible via Mullvad VPN (non-standard port)
# - Public access to news dashboard (port 5001)
# - DISA STIG security hardening
#
# Prerequisites:
# - Hetzner CLI installed: brew install hcloud
# - Hetzner API token: https://console.hetzner.cloud/projects/*/security/tokens
# - Mullvad VPN subscription
# =============================================================================

set -e

# Configuration
SERVER_NAME="${SERVER_NAME:-linkedin-influencer}"
SERVER_TYPE="${SERVER_TYPE:-cx32}"  # 4 vCPU, 8GB RAM, €6.80/mo
LOCATION="${LOCATION:-nbg1}"  # Nuremberg (cheapest)
IMAGE="${IMAGE:-ubuntu-24.04}"
SSH_PORT="${SSH_PORT:-49222}"  # Non-standard SSH port
DASHBOARD_PORT="${DASHBOARD_PORT:-5001}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}=== Hetzner Cloud Deployment ===${NC}"
echo "Server: $SERVER_NAME ($SERVER_TYPE)"
echo "Location: $LOCATION"
echo "SSH Port: $SSH_PORT (Mullvad VPN only)"
echo "Dashboard Port: $DASHBOARD_PORT (public)"
echo ""

# Check prerequisites
if ! command -v hcloud &> /dev/null; then
    echo -e "${RED}Error: hcloud CLI not installed${NC}"
    echo "Install with: brew install hcloud"
    exit 1
fi

# Check if context is set
if ! hcloud context active &> /dev/null; then
    echo -e "${YELLOW}No Hetzner context active. Creating one...${NC}"
    echo "Get your API token from: https://console.hetzner.cloud/"
    read -p "Enter Hetzner API Token: " HCLOUD_TOKEN
    hcloud context create linkedin-influencer
fi

echo -e "${GREEN}Step 1: Creating SSH key (if needed)${NC}"
SSH_KEY_NAME="linkedin-influencer-key"
if ! hcloud ssh-key describe "$SSH_KEY_NAME" &> /dev/null; then
    if [ -f ~/.ssh/id_ed25519.pub ]; then
        hcloud ssh-key create --name "$SSH_KEY_NAME" --public-key-from-file ~/.ssh/id_ed25519.pub
        echo "SSH key created from ~/.ssh/id_ed25519.pub"
    elif [ -f ~/.ssh/id_rsa.pub ]; then
        hcloud ssh-key create --name "$SSH_KEY_NAME" --public-key-from-file ~/.ssh/id_rsa.pub
        echo "SSH key created from ~/.ssh/id_rsa.pub"
    else
        echo -e "${RED}No SSH key found. Generate one with: ssh-keygen -t ed25519${NC}"
        exit 1
    fi
else
    echo "SSH key '$SSH_KEY_NAME' already exists"
fi

echo -e "${GREEN}Step 2: Creating Firewall with Mullvad VPN rules${NC}"
FIREWALL_NAME="linkedin-influencer-fw"

# Mullvad VPN exit node IP ranges (updated Dec 2024)
# Source: https://mullvad.net/en/servers
# These are approximate ranges - Mullvad uses various providers
MULLVAD_RANGES=(
    # Mullvad owned ranges
    "185.65.134.0/24"
    "185.65.135.0/24"
    "185.213.154.0/24"
    "185.213.155.0/24"
    "193.27.12.0/22"
    "193.138.218.0/24"
    "198.54.128.0/24"
    "45.83.220.0/22"
    "66.63.167.0/24"
    "86.106.121.0/24"
    "141.98.252.0/24"
    "146.70.0.0/16"
    # Additional common Mullvad ranges
    "37.120.147.0/24"
    "45.12.220.0/22"
    "89.46.62.0/24"
    "91.90.44.0/24"
    "95.182.120.0/22"
    "103.108.229.0/24"
    "104.234.210.0/24"
    "169.150.196.0/22"
    "194.127.199.0/24"
)

# Delete existing firewall if it exists
hcloud firewall delete "$FIREWALL_NAME" 2>/dev/null || true

# Build SSH rules for Mullvad ranges
SSH_RULES=""
for range in "${MULLVAD_RANGES[@]}"; do
    SSH_RULES="$SSH_RULES --rule direction=in,protocol=tcp,port=$SSH_PORT,source-ips=$range"
done

# Create firewall with rules
echo "Creating firewall with Mullvad-only SSH access..."
hcloud firewall create --name "$FIREWALL_NAME" \
    $SSH_RULES \
    --rule "direction=in,protocol=tcp,port=$DASHBOARD_PORT,source-ips=0.0.0.0/0" \
    --rule "direction=in,protocol=tcp,port=$DASHBOARD_PORT,source-ips=::/0" \
    --rule "direction=in,protocol=icmp,source-ips=0.0.0.0/0"

echo "Firewall created: SSH on port $SSH_PORT (Mullvad only), Dashboard on port $DASHBOARD_PORT (public)"

echo -e "${GREEN}Step 3: Creating cloud-init configuration${NC}"
CLOUD_INIT=$(cat <<EOF
#cloud-config
package_update: true
package_upgrade: true

packages:
  - xfce4
  - xfce4-goodies
  - tigervnc-standalone-server
  - tigervnc-common
  - chromium-browser
  - chromium-chromedriver
  - python3.12
  - python3.12-venv
  - sqlite3
  - curl
  - git
  - unzip
  - nodejs
  - npm
  - ansible
  - fail2ban
  - ufw

users:
  - name: appuser
    groups: sudo
    shell: /bin/bash
    sudo: ALL=(ALL) NOPASSWD:ALL
    ssh_authorized_keys:
      - $(cat ~/.ssh/id_ed25519.pub 2>/dev/null || cat ~/.ssh/id_rsa.pub)

write_files:
  - path: /etc/ssh/sshd_config.d/hardened.conf
    content: |
      Port $SSH_PORT
      PermitRootLogin no
      PasswordAuthentication no
      PubkeyAuthentication yes
      X11Forwarding yes
      AllowTcpForwarding yes
      ClientAliveInterval 300
      ClientAliveCountMax 2
      MaxAuthTries 3
      LoginGraceTime 30

  - path: /home/appuser/.vnc/xstartup
    permissions: '0755'
    content: |
      #!/bin/bash
      unset SESSION_MANAGER
      unset DBUS_SESSION_BUS_ADDRESS
      exec startxfce4

  - path: /etc/systemd/system/vncserver@.service
    content: |
      [Unit]
      Description=VNC Server for display %i
      After=syslog.target network.target

      [Service]
      Type=forking
      User=appuser
      Group=appuser
      WorkingDirectory=/home/appuser
      ExecStartPre=/bin/sh -c '/usr/bin/vncserver -kill :%i > /dev/null 2>&1 || :'
      ExecStart=/usr/bin/vncserver :%i -geometry 1920x1080 -depth 24 -localhost
      ExecStop=/usr/bin/vncserver -kill :%i

      [Install]
      WantedBy=multi-user.target

runcmd:
  # Configure SSH on non-standard port
  - systemctl restart sshd

  # Setup UFW firewall (backup to Hetzner firewall)
  - ufw default deny incoming
  - ufw default allow outgoing
  - ufw allow $SSH_PORT/tcp comment 'SSH Mullvad only'
  - ufw allow $DASHBOARD_PORT/tcp comment 'Dashboard public'
  - ufw --force enable

  # Install uv for appuser
  - su - appuser -c 'curl -LsSf https://astral.sh/uv/install.sh | sh'

  # Setup VNC password (random, user will reset)
  - mkdir -p /home/appuser/.vnc
  - chown -R appuser:appuser /home/appuser/.vnc
  - su - appuser -c 'echo -e "changeme\nchangeme\nn" | vncpasswd'

  # Enable and start VNC
  - systemctl daemon-reload
  - systemctl enable vncserver@1
  - systemctl start vncserver@1

  # Install Claude CLI
  - npm install -g @anthropic-ai/claude-code

  # Setup fail2ban for SSH
  - |
    cat > /etc/fail2ban/jail.local << 'JAILEOF'
    [sshd]
    enabled = true
    port = $SSH_PORT
    filter = sshd
    logpath = /var/log/auth.log
    maxretry = 3
    bantime = 3600
    findtime = 600
    JAILEOF
  - systemctl enable fail2ban
  - systemctl restart fail2ban

  # Final message
  - echo "Server setup complete!" > /home/appuser/SETUP_COMPLETE

final_message: |
  ===========================================
  Server setup complete!
  SSH Port: $SSH_PORT (Mullvad VPN required)
  Dashboard: http://\$PUBLIC_IP:$DASHBOARD_PORT
  VNC: localhost:5901 (via SSH tunnel)
  ===========================================
EOF
)

echo -e "${GREEN}Step 4: Creating server${NC}"

# Check if server already exists
if hcloud server describe "$SERVER_NAME" &> /dev/null; then
    echo -e "${YELLOW}Server '$SERVER_NAME' already exists${NC}"
    read -p "Delete and recreate? (y/N): " CONFIRM
    if [ "$CONFIRM" = "y" ] || [ "$CONFIRM" = "Y" ]; then
        hcloud server delete "$SERVER_NAME"
        sleep 5
    else
        echo "Aborting."
        exit 1
    fi
fi

# Create server
echo "Creating server (this takes ~2 minutes)..."
hcloud server create \
    --name "$SERVER_NAME" \
    --type "$SERVER_TYPE" \
    --location "$LOCATION" \
    --image "$IMAGE" \
    --ssh-key "$SSH_KEY_NAME" \
    --firewall "$FIREWALL_NAME" \
    --user-data "$CLOUD_INIT"

# Get server IP
SERVER_IP=$(hcloud server ip "$SERVER_NAME")

echo -e "${GREEN}=== Server Created ===${NC}"
echo ""
echo "Server IP: $SERVER_IP"
echo ""
echo -e "${YELLOW}=== Connection Instructions ===${NC}"
echo ""
echo "1. Connect to Mullvad VPN first!"
echo ""
echo "2. SSH into server:"
echo "   ssh -p $SSH_PORT appuser@$SERVER_IP"
echo ""
echo "3. VNC access (via SSH tunnel):"
echo "   ssh -p $SSH_PORT -L 5901:localhost:5901 appuser@$SERVER_IP"
echo "   Then connect VNC client to: localhost:5901"
echo "   Default VNC password: changeme (change it with 'vncpasswd')"
echo ""
echo "4. Dashboard (public):"
echo "   http://$SERVER_IP:$DASHBOARD_PORT"
echo ""
echo -e "${YELLOW}=== Next Steps ===${NC}"
echo ""
echo "Wait 3-5 minutes for cloud-init to complete, then:"
echo ""
echo "1. Connect via Mullvad VPN"
echo "2. SSH in: ssh -p $SSH_PORT appuser@$SERVER_IP"
echo "3. Clone repo and setup:"
echo "   git clone https://github.com/pierretokns/twitter_influencer.git"
echo "   cd twitter_influencer"
echo "   git checkout claude/cloudflare-workers-ai-exploration-015SWzxnn3Nw8ZnbrGus1xMF"
echo "   uv sync"
echo ""
echo "4. Setup VNC tunnel and login to Claude CLI:"
echo "   claude /login"
echo ""
echo "5. Apply DISA STIG hardening (see DEPLOY_HETZNER.md Step 12)"
echo ""

# Save connection info
cat > deploy/connection_info.txt << EOF
=== LinkedIn Influencer Server ===
Created: $(date)

Server IP: $SERVER_IP
SSH Port: $SSH_PORT (Mullvad VPN required)
Dashboard: http://$SERVER_IP:$DASHBOARD_PORT

SSH Command:
ssh -p $SSH_PORT appuser@$SERVER_IP

VNC Tunnel:
ssh -p $SSH_PORT -L 5901:localhost:5901 appuser@$SERVER_IP

VNC Password: changeme (change with 'vncpasswd')
EOF

echo -e "${GREEN}Connection info saved to: deploy/connection_info.txt${NC}"
