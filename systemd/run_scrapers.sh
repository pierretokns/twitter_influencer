#!/bin/bash
# Cron script for running scrapers
# Add to crontab: 0 8,20 * * * /home/appuser/run_scrapers.sh

export HOME=/home/appuser
export PATH=/home/appuser/.local/bin:$PATH
cd /home/appuser/twitter_influencer
/home/appuser/.local/bin/uv run python run_scrapers.py --no-alert >> /home/appuser/logs/scrape.log 2>&1
