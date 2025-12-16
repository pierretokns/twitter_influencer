# Systemd Services

These are the systemd unit files for running the LinkedIn tools on a server.

## Services

| Service | Port | Description |
|---------|------|-------------|
| `linkedin-ui.service` | 5001 | Tournament dashboard for generating and ranking posts |
| `linkedin-feed.service` | 5002 | LinkedIn feed clone showing daily winning posts |

## Installation

```bash
# Copy unit files
sudo cp systemd/*.service /etc/systemd/system/

# Reload systemd
sudo systemctl daemon-reload

# Enable and start services
sudo systemctl enable linkedin-ui.service linkedin-feed.service
sudo systemctl start linkedin-ui.service linkedin-feed.service

# Check status
sudo systemctl status linkedin-ui.service
sudo systemctl status linkedin-feed.service
```

## Cron Setup

```bash
# Copy the scraper script
cp systemd/run_scrapers.sh ~/run_scrapers.sh
chmod +x ~/run_scrapers.sh

# Create logs directory
mkdir -p ~/logs

# Add to crontab (runs at 8am and 8pm UTC)
crontab -e
# Add: 0 8,20 * * * /home/appuser/run_scrapers.sh
```

## Logs

```bash
# View service logs
journalctl -u linkedin-ui.service -f
journalctl -u linkedin-feed.service -f

# View scraper logs
tail -f ~/logs/scrape.log
```
