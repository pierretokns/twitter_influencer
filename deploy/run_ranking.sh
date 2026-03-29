#!/bin/bash
#
# Ranking Wrapper Script
# ======================
#
# Runs the LinkedIn ranking tournament with OTEL tracing enabled.
# Place in ~/run_ranking.sh on the VM.
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
~/.local/bin/uv run python linkedin_autopilot.py --rank >> ~/logs/rank.log 2>&1
