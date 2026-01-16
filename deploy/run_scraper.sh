#!/bin/bash
#
# Scraper Wrapper Script
# ======================
#
# Runs the unified scraper with OTEL tracing enabled.
# Place in ~/run_scraper.sh on the VM.
#

export DISPLAY=:20
export HOME=/home/$(whoami)

# Load environment variables from .env
set -a
source ~/twitter_influencer/.env
set +a

# OTEL Configuration for Claude CLI tracing
export CLAUDE_CODE_ENABLE_TELEMETRY=1
export OTEL_EXPORTER_OTLP_PROTOCOL=grpc
export OTEL_EXPORTER_OTLP_ENDPOINT=http://127.0.0.1:4317

cd ~/twitter_influencer
~/.local/bin/uv run python run_scrapers.py >> ~/logs/scrape.log 2>&1
