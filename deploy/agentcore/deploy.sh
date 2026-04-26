#!/usr/bin/env bash
# deploy.sh — Deploy both AgentCore runtimes via ZIP (no Docker)
# Usage: ./deploy.sh [--destroy]
#
# Creates (all destroyable via --destroy):
#   S3 bucket:    bedrock-agentcore-code-{ACCOUNT}-{REGION}  (may already exist)
#   IAM role:     AgentCoreRuntimeRole
#   AgentCore:    ranking-agent, chat-agent
#
# To tear down:  ./deploy.sh --destroy

set -euo pipefail

REGION="us-east-1"
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
HETZNER_FEED_URL="${HETZNER_FEED_URL:-http://157.90.125.102:5002}"
DIR="$(cd "$(dirname "$0")" && pwd)"

S3_BUCKET="bedrock-agentcore-code-${ACCOUNT_ID}-${REGION}"
IAM_ROLE="AgentCoreRuntimeRole"
ROLE_ARN="arn:aws:iam::${ACCOUNT_ID}:role/${IAM_ROLE}"
PYTHON_RUNTIME="PYTHON_3_13"

# ── Destroy ───────────────────────────────────────────────────────────────────
if [[ "${1:-}" == "--destroy" ]]; then
  echo "==> Deleting AgentCore runtimes..."
  for name in ranking_agent chat_agent; do
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
  aws s3 rm "s3://${S3_BUCKET}/ranking-agent/" --recursive 2>/dev/null || true
  aws s3 rm "s3://${S3_BUCKET}/chat-agent/" --recursive 2>/dev/null || true

  echo "==> Done. All AgentCore resources destroyed."
  exit 0
fi

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
      {\"Effect\":\"Allow\",\"Action\":[\"s3:GetObject\"],\"Resource\":\"arn:aws:s3:::${S3_BUCKET}/*\"}
    ]
  }"

# ── S3 Bucket ─────────────────────────────────────────────────────────────────
echo "==> S3 bucket ${S3_BUCKET}..."
aws s3api create-bucket --bucket "$S3_BUCKET" --region "$REGION" 2>/dev/null || true

# ── Package & Deploy ──────────────────────────────────────────────────────────
deploy_agent() {
  local name="$1"
  local agent_dir="${DIR}/${name//-agent/}"  # ranking-agent -> ranking dir
  # AgentCore name constraint: [a-zA-Z][a-zA-Z0-9_]{0,47}
  local runtime_name="${name//-/_}"
  local tmp_dir
  tmp_dir=$(mktemp -d)

  echo "==> Packaging ${name}..."

  # Install deps into package dir (ARM64 wheels for AgentCore)
  uv pip install \
    --python-platform aarch64-manylinux2014 \
    --python-version 3.13 \
    --target "${tmp_dir}/package" \
    --only-binary=:all: \
    "strands-agents>=0.1.0" \
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
  echo "  Package size: ${size_mb}MB"

  if [[ $size_mb -gt 250 ]]; then
    echo "  ERROR: Package exceeds 250MB ZIP limit (${size_mb}MB)"
    rm -rf "$tmp_dir"
    exit 1
  fi

  # Upload to S3
  echo "  Uploading to S3..."
  aws s3 cp "$zip_path" "s3://${S3_BUCKET}/${name}/deployment_package.zip" --region "$REGION"

  rm -rf "$tmp_dir"

  # Deploy to AgentCore
  echo "  Deploying to AgentCore..."
  local artifact="{\"codeConfiguration\":{\"code\":{\"s3\":{\"bucket\":\"${S3_BUCKET}\",\"prefix\":\"${name}/deployment_package.zip\"}},\"runtime\":\"${PYTHON_RUNTIME}\",\"entryPoint\":[\"main.py\"]}}"
  local env_vars="{\"HETZNER_FEED_URL\":\"${HETZNER_FEED_URL}\",\"AWS_DEFAULT_REGION\":\"${REGION}\"}"

  # Delete existing runtime if present (UpdateAgentRuntime may be restricted on SSO roles)
  local existing_id
  existing_id=$(aws bedrock-agentcore-control list-agent-runtimes --region "$REGION" \
    --query "agentRuntimes[?agentRuntimeName=='${runtime_name}'].agentRuntimeId" \
    --output text 2>/dev/null)
  if [[ -n "$existing_id" ]]; then
    echo "  Deleting existing: ${existing_id}..."
    aws bedrock-agentcore-control delete-agent-runtime \
      --agent-runtime-id "$existing_id" --region "$REGION" 2>/dev/null || true
    echo "  Waiting for deletion..."
    for i in $(seq 1 12); do
      sleep 5
      local still_exists
      still_exists=$(aws bedrock-agentcore-control list-agent-runtimes --region "$REGION" \
        --query "agentRuntimes[?agentRuntimeId=='${existing_id}'].agentRuntimeId" \
        --output text 2>/dev/null)
      [[ -z "$still_exists" ]] && break
      echo "    still deleting... (${i})"
    done
  fi

  aws bedrock-agentcore-control create-agent-runtime \
    --agent-runtime-name "$runtime_name" \
    --agent-runtime-artifact "$artifact" \
    --network-configuration '{"networkMode":"PUBLIC"}' \
    --environment-variables "$env_vars" \
    --role-arn "$ROLE_ARN" \
    --region "$REGION" && echo "  Created: ${runtime_name}" || { echo "  ERROR creating ${runtime_name}"; exit 1; }

  echo "  Done: ${name} (runtime: ${runtime_name})"
}

deploy_agent "ranking-agent"
deploy_agent "chat-agent"

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo "Deployed. Invoke examples:"
echo ""
echo "aws bedrock-agentcore invoke-agent-runtime \\"
echo "  --agent-runtime-arn arn:aws:bedrock-agentcore:${REGION}:${ACCOUNT_ID}:runtime/ranking-agent \\"
echo "  --payload '{\"num_variants\":5,\"elo_rounds\":3}' --region ${REGION}"
echo ""
echo "aws bedrock-agentcore invoke-agent-runtime \\"
echo "  --agent-runtime-arn arn:aws:bedrock-agentcore:${REGION}:${ACCOUNT_ID}:runtime/chat-agent \\"
echo "  --payload '{\"query\":\"What happened with OpenAI this week?\",\"session_id\":\"demo\"}' --region ${REGION}"
echo ""
echo "Tear down: ./deploy.sh --destroy"
