#!/usr/bin/env bash
# deploy.sh — Deploy legacy hosted AgentCore runtimes via ZIP (no Docker)
# Usage: ./deploy.sh [--destroy]
#
# IMPORTANT: this is the legacy hosted Bedrock/Strands stack for the old
# tournament/influencer workflow. The Brandon news-summary migration uses the
# Hetzner-local llama.cpp services instead. Destruction is allowed by default,
# but deployment requires:
#
#   ALLOW_HOSTED_AGENTCORE_DEPLOY=1 ./deploy.sh
#
# Creates (all destroyable via --destroy):
#   S3 bucket:    bedrock-agentcore-code-{ACCOUNT}-{REGION}  (may already exist)
#   IAM role:     AgentCoreRuntimeRole
#   AgentCore:    generator_agent, qe_agent, evolution_agent, debate_agent,
#                 ranking_orchestrator, chat_agent
#
# To tear down:  ./deploy.sh --destroy

set -euo pipefail

REGION="us-east-1"
HETZNER_FEED_URL="${HETZNER_FEED_URL:-http://157.90.125.102:5002}"
DIR="$(cd "$(dirname "$0")" && pwd)"
IAM_ROLE="AgentCoreRuntimeRole"
PYTHON_RUNTIME="PYTHON_3_13"

# ── Destroy ───────────────────────────────────────────────────────────────────
if [[ "${1:-}" == "--destroy" ]]; then
  ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
  S3_BUCKET="bedrock-agentcore-code-${ACCOUNT_ID}-${REGION}"
  echo "==> Deleting AgentCore runtimes..."
  for name in generator_agent qe_agent evolution_agent debate_agent ranking_orchestrator chat_agent; do
    aws bedrock-agentcore-control list-agent-runtimes --region "$REGION" \
      --query "agentRuntimes[?starts_with(agentRuntimeId,'${name}')].agentRuntimeId" \
      --output text | tr '\t' '\n' | while read -r id; do
      [[ -z "$id" ]] && continue
      aws bedrock-agentcore-control delete-agent-runtime \
        --agent-runtime-id "$id" --region "$REGION" 2>/dev/null \
        && echo "  deleted $id" || echo "  $id not found"
    done
  done

  echo "==> Deleting IAM role..."
  aws iam delete-role-policy --role-name "$IAM_ROLE" --policy-name AgentCorePolicy 2>/dev/null || true
  aws iam delete-role --role-name "$IAM_ROLE" 2>/dev/null || true

  echo "==> S3 objects..."
  for name in generator_agent qe_agent evolution_agent debate_agent ranking_orchestrator chat_agent; do
    aws s3 rm "s3://${S3_BUCKET}/${name}/" --recursive 2>/dev/null || true
  done

  echo "==> Done. All AgentCore resources destroyed."
  exit 0
fi

if [[ "${ALLOW_HOSTED_AGENTCORE_DEPLOY:-0}" != "1" ]]; then
  cat >&2 <<'EOF'
Refusing to deploy legacy hosted AgentCore runtimes.

This stack still packages Strands/Bedrock leaf agents and the old
tournament/influencer workflow. The current Brandon news-summary migration
uses Hetzner-local llama.cpp services instead.

To intentionally deploy this legacy hosted stack, rerun with:
  ALLOW_HOSTED_AGENTCORE_DEPLOY=1 ./deploy.sh

Destruction remains available without the override:
  ./deploy.sh --destroy
EOF
  exit 2
fi

ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
S3_BUCKET="bedrock-agentcore-code-${ACCOUNT_ID}-${REGION}"
ROLE_ARN="arn:aws:iam::${ACCOUNT_ID}:role/${IAM_ROLE}"

# ── IAM Role ──────────────────────────────────────────────────────────────────
echo "==> IAM role ${IAM_ROLE}..."
aws iam create-role --role-name "$IAM_ROLE" \
  --assume-role-policy-document '{
    "Version":"2012-10-17",
    "Statement":[{"Effect":"Allow","Principal":{"Service":"bedrock-agentcore.amazonaws.com"},"Action":"sts:AssumeRole"}]
  }' 2>/dev/null || true

aws iam put-role-policy --role-name "$IAM_ROLE" --policy-name AgentCorePolicy \
  --policy-document "{
    \"Version\":\"2012-10-17\",
    \"Statement\":[
      {\"Effect\":\"Allow\",\"Action\":[\"bedrock:InvokeModel\",\"bedrock:InvokeModelWithResponseStream\"],\"Resource\":\"*\"},
      {\"Effect\":\"Allow\",\"Action\":[\"logs:CreateLogGroup\",\"logs:CreateLogStream\",\"logs:PutLogEvents\"],\"Resource\":\"arn:aws:logs:*:${ACCOUNT_ID}:log-group:/aws/bedrock-agentcore/*\"},
      {\"Effect\":\"Allow\",\"Action\":[\"s3:GetObject\"],\"Resource\":\"arn:aws:s3:::${S3_BUCKET}/*\"},
      {\"Effect\":\"Allow\",\"Action\":[\"bedrock-agentcore:InvokeAgentRuntime\"],\"Resource\":\"*\"}
    ]
  }"

# ── S3 Bucket ─────────────────────────────────────────────────────────────────
echo "==> S3 bucket ${S3_BUCKET}..."
aws s3api create-bucket --bucket "$S3_BUCKET" --region "$REGION" 2>/dev/null || true

# ── Package & Deploy ──────────────────────────────────────────────────────────
# deploy_agent <name> [env_vars_json]
# name = directory name under deploy/agentcore/ AND runtime name
# Echoes the created agentRuntimeArn to stdout on success.
deploy_agent() {
  local name="$1"
  local extra_env_vars="${2:-}"
  local agent_dir="${DIR}/${name}"
  local runtime_name="${name}"
  local tmp_dir
  tmp_dir=$(mktemp -d)

  echo "==> Packaging ${name}..." >&2

  # Orchestrator needs no Strands (pure boto3 orchestration) — saves ~30MB
  local strands_dep="strands-agents>=0.1.0"
  if [[ "$name" == "ranking_orchestrator" ]]; then
    strands_dep=""
  fi

  # Install deps into package dir (ARM64 wheels for AgentCore)
  uv pip install \
    --python-platform aarch64-manylinux2014 \
    --python-version 3.13 \
    --target "${tmp_dir}/package" \
    --only-binary=:all: \
    ${strands_dep:+"$strands_dep"} \
    "boto3>=1.34.0" \
    "requests>=2.31.0" \
    "python-dotenv>=0.19.0" \
    "fastapi>=0.110.0" \
    "uvicorn>=0.29.0" \
    "httptools>=0.6.0" \
    "uvloop>=0.19.0" \
    -q

  # Copy agent entrypoint
  cp "${agent_dir}/main.py" "${tmp_dir}/package/"

  # Zip it
  local zip_path="${tmp_dir}/${name}.zip"
  (cd "${tmp_dir}/package" && zip -r "$zip_path" . -x "*.pyc" -x "*__pycache__*" -q)

  local size_mb
  size_mb=$(du -sm "$zip_path" | cut -f1)
  echo "  Package size: ${size_mb}MB" >&2

  if [[ $size_mb -gt 250 ]]; then
    echo "  ERROR: Package exceeds 250MB ZIP limit (${size_mb}MB)" >&2
    rm -rf "$tmp_dir"
    exit 1
  fi

  # Upload to S3
  echo "  Uploading to S3..." >&2
  aws s3 cp "$zip_path" "s3://${S3_BUCKET}/${name}/deployment_package.zip" --region "$REGION" >&2

  rm -rf "$tmp_dir"

  # Build env vars JSON — merge base vars with any extra vars
  local base_env="{\"HETZNER_FEED_URL\":\"${HETZNER_FEED_URL}\",\"AWS_DEFAULT_REGION\":\"${REGION}\"}"
  local env_vars
  if [[ -n "$extra_env_vars" ]]; then
    # Merge: strip trailing } from base, strip leading { from extra, join with comma
    env_vars="${base_env%\}}$(echo "$extra_env_vars" | sed 's/^{/,/')}"
  else
    env_vars="$base_env"
  fi

  # Deploy to AgentCore
  echo "  Deploying to AgentCore..." >&2
  local artifact="{\"codeConfiguration\":{\"code\":{\"s3\":{\"bucket\":\"${S3_BUCKET}\",\"prefix\":\"${name}/deployment_package.zip\"}},\"runtime\":\"${PYTHON_RUNTIME}\",\"entryPoint\":[\"main.py\"]}}"

  # Delete existing runtime if present
  local existing_id
  existing_id=$(aws bedrock-agentcore-control list-agent-runtimes --region "$REGION" \
    --query "agentRuntimes[?agentRuntimeName=='${runtime_name}'].agentRuntimeId" \
    --output text 2>/dev/null)
  if [[ -n "$existing_id" ]]; then
    echo "  Deleting existing: ${existing_id}..." >&2
    aws bedrock-agentcore-control delete-agent-runtime \
      --agent-runtime-id "$existing_id" --region "$REGION" >/dev/null 2>&1 || true
    echo "  Waiting for deletion..." >&2
    for i in $(seq 1 12); do
      sleep 5
      local still_exists
      still_exists=$(aws bedrock-agentcore-control list-agent-runtimes --region "$REGION" \
        --query "agentRuntimes[?agentRuntimeId=='${existing_id}'].agentRuntimeId" \
        --output text 2>/dev/null)
      [[ -z "$still_exists" ]] && break
      echo "    still deleting... (${i})" >&2
    done
  fi

  local CREATE_OUTPUT
  CREATE_OUTPUT=$(aws bedrock-agentcore-control create-agent-runtime \
    --agent-runtime-name "$runtime_name" \
    --agent-runtime-artifact "$artifact" \
    --network-configuration '{"networkMode":"PUBLIC"}' \
    --environment-variables "$env_vars" \
    --role-arn "$ROLE_ARN" \
    --region "$REGION" 2>&1)

  if [[ $? -ne 0 ]] || ! echo "$CREATE_OUTPUT" | python3 -c "import sys,json; json.load(sys.stdin)" 2>/dev/null; then
    echo "  ERROR creating ${runtime_name}" >&2
    exit 1
  fi

  local RUNTIME_ARN
  RUNTIME_ARN=$(echo "$CREATE_OUTPUT" | python3 -c "import sys,json; print(json.load(sys.stdin)['agentRuntimeArn'])")
  echo "  Created: ${runtime_name} (ARN: ${RUNTIME_ARN})" >&2
  echo "  Done: ${name}" >&2

  # Echo the ARN to stdout so caller can capture it
  echo "$RUNTIME_ARN"
}

# ── Deploy leaf agents first, capture ARNs ────────────────────────────────────
GENERATOR_ARN=$(deploy_agent "generator_agent")
QE_ARN=$(deploy_agent "qe_agent")
EVOLUTION_ARN=$(deploy_agent "evolution_agent")
DEBATE_ARN=$(deploy_agent "debate_agent")

echo "==> Captured ARNs:"
echo "    GENERATOR_ARN=${GENERATOR_ARN}"
echo "    QE_ARN=${QE_ARN}"
echo "    EVOLUTION_ARN=${EVOLUTION_ARN}"
echo "    DEBATE_ARN=${DEBATE_ARN}"

# ── Deploy orchestrator with leaf ARNs as env vars ────────────────────────────
ORCHESTRATOR_ENV_VARS="{\"GENERATOR_AGENT_ARN\":\"${GENERATOR_ARN}\",\"QE_AGENT_ARN\":\"${QE_ARN}\",\"EVOLUTION_AGENT_ARN\":\"${EVOLUTION_ARN}\",\"DEBATE_AGENT_ARN\":\"${DEBATE_ARN}\"}"
deploy_agent "ranking_orchestrator" "$ORCHESTRATOR_ENV_VARS"

# ── Deploy chat agent (unchanged) ─────────────────────────────────────────────
deploy_agent "chat_agent"

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo "Deployed. Invoke examples:"
echo ""
echo "aws bedrock-agentcore invoke-agent-runtime \\"
echo "  --agent-runtime-arn arn:aws:bedrock-agentcore:${REGION}:${ACCOUNT_ID}:runtime/ranking_orchestrator \\"
echo "  --payload '{\"num_variants\":5,\"elo_rounds\":3}' --region ${REGION}"
echo ""
echo "aws bedrock-agentcore invoke-agent-runtime \\"
echo "  --agent-runtime-arn arn:aws:bedrock-agentcore:${REGION}:${ACCOUNT_ID}:runtime/chat_agent \\"
echo "  --payload '{\"query\":\"What happened with OpenAI this week?\",\"session_id\":\"demo\"}' --region ${REGION}"
echo ""
echo "Tear down: ./deploy.sh --destroy"
