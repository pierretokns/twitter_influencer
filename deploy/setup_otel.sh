#!/bin/bash
#
# OTEL Collector Setup Script
# ===========================
#
# Installs and configures the OTEL Collector with SQLite exporter
# for storing Claude CLI traces in the local database.
#
# Prerequisites:
#   - Ubuntu 22.04+
#   - twitter_influencer repo cloned to ~/twitter_influencer
#
# Usage:
#   ./deploy/setup_otel.sh
#

set -e

echo "=== OTEL Collector Setup ==="
echo ""

# Step 1: Install Go 1.21+
echo "[1/5] Installing Go 1.21..."
if ! go version 2>/dev/null | grep -q "go1.2[1-9]"; then
    sudo rm -rf /usr/local/go
    curl -fsSL https://go.dev/dl/go1.21.6.linux-amd64.tar.gz | sudo tar -C /usr/local -xzf -
    echo 'export PATH=$PATH:/usr/local/go/bin' >> ~/.bashrc
    export PATH=$PATH:/usr/local/go/bin
    echo "  Go 1.21.6 installed"
else
    echo "  Go 1.21+ already installed"
fi

# Step 2: Clone and build sqliteexporter
echo "[2/5] Building SQLite exporter..."
cd ~
if [ -d "sqliteexporter" ]; then
    cd sqliteexporter
    git pull
else
    git clone https://github.com/pierretokns/sqliteexporter.git
    cd sqliteexporter
fi

# Fix the replace path in builder-config.yaml
sed -i "s|/home/wperron/github.com/wperron/sqliteexporter|$HOME/sqliteexporter|g" builder-config.yaml

# Build
export PATH=/usr/local/go/bin:$PATH
make setup
./bin/ocb --config builder-config.yaml
echo "  SQLite exporter built"

# Step 3: Install collector config
echo "[3/5] Installing collector config..."
cp ~/twitter_influencer/deploy/otel-collector-config.yaml ~/otel-collector-config.yaml
# Update path for current user
sed -i "s|/home/erki|$HOME|g" ~/otel-collector-config.yaml
echo "  Config installed to ~/otel-collector-config.yaml"

# Step 4: Install systemd service
echo "[4/5] Installing systemd service..."
SERVICE_FILE=/etc/systemd/system/otel-collector.service
sudo cp ~/twitter_influencer/deploy/systemd/otel-collector.service $SERVICE_FILE
# Update paths for current user
sudo sed -i "s|/home/erki|$HOME|g" $SERVICE_FILE
sudo sed -i "s|User=erki|User=$(whoami)|g" $SERVICE_FILE
sudo systemctl daemon-reload
sudo systemctl enable otel-collector
echo "  Systemd service installed"

# Step 5: Start collector
echo "[5/5] Starting OTEL collector..."
sudo systemctl start otel-collector
sleep 2
if sudo systemctl is-active --quiet otel-collector; then
    echo "  OTEL collector is running"
else
    echo "  WARNING: OTEL collector failed to start"
    echo "  Check logs: sudo journalctl -u otel-collector -n 20"
fi

echo ""
echo "=== Setup Complete ==="
echo ""
echo "OTEL Collector listening on:"
echo "  - gRPC: 127.0.0.1:4317"
echo "  - HTTP: 127.0.0.1:4318"
echo ""
echo "Traces will be stored in: ~/twitter_influencer/output_data/ai_news.db"
echo ""
echo "To configure Claude CLI to send traces, set these env vars:"
echo "  export CLAUDE_CODE_ENABLE_TELEMETRY=1"
echo "  export OTEL_EXPORTER_OTLP_PROTOCOL=grpc"
echo "  export OTEL_EXPORTER_OTLP_ENDPOINT=http://127.0.0.1:4317"
