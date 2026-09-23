#!/usr/bin/env bash
# Run KSI against a Chat-Completions-only OpenAI-compatible provider using the
# URL/KEY already stored in the parent workspace .env. No provider secret is
# copied into this repository or passed into the agent container.
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
ENV_FILE="$REPO_ROOT/../.env"

if [[ ! -f "$ENV_FILE" ]]; then
  echo "Missing provider environment file: $ENV_FILE" >&2
  exit 1
fi

set -a
source "$ENV_FILE"
set +a

if [[ -z "${URL:-}" || -z "${KEY:-}" ]]; then
  echo "The parent .env must define URL and KEY." >&2
  exit 1
fi

cd "$REPO_ROOT"
proxy_pid=""
if ! lsof -nP -iTCP:4002 -sTCP:LISTEN >/dev/null 2>&1; then
  node scripts/responses_chat_proxy.mjs >"${TMPDIR:-/tmp}/ksi-responses-chat-proxy.log" 2>&1 &
  proxy_pid=$!
  # Give Node a moment to bind; fail early with its log if it did not start.
  sleep 1
  if ! kill -0 "$proxy_pid" 2>/dev/null; then
    cat "${TMPDIR:-/tmp}/ksi-responses-chat-proxy.log" >&2 || true
    exit 1
  fi
fi
cleanup() { [[ -z "$proxy_pid" ]] || kill "$proxy_pid" 2>/dev/null || true; }
trap cleanup EXIT INT TERM

uv run python -m ksi.cli \
  --provider-profile configs/ksi/.env.deepinfra-proxy \
  "$@"
