#!/bin/bash
# Cron script for running scrapers
# Add to crontab: 0 8,20 * * * /home/appuser/run_scrapers.sh

export HOME=/home/appuser
export PATH=/home/appuser/.local/bin:$PATH

# Set DISPLAY for headful Chrome on Linux
if [[ "$(uname)" == "Linux" ]]; then
    if pgrep -f "chrome-remote-desktop" > /dev/null; then
        # Chrome Remote Desktop uses display :20
        export DISPLAY=:20
    elif [[ -f /tmp/.X1-lock ]]; then
        # VNC typically uses display :1
        export DISPLAY=:1
    fi
fi

cd /home/appuser/twitter_influencer
/home/appuser/.local/bin/uv run python run_scrapers.py --no-alert >> /home/appuser/logs/scrape.log 2>&1

# Trigger tournament generation after scraping (15 variants, 10 rounds)
echo "[$(date)] Starting tournament..." >> /home/appuser/logs/scrape.log
curl -s -X POST "http://localhost:5001/api/start" \
    -H "Content-Type: application/json" \
    -d '{"variants": 15, "rounds": 10}' >> /home/appuser/logs/scrape.log 2>&1
echo "" >> /home/appuser/logs/scrape.log
