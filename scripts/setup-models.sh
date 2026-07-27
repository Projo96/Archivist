#!/usr/bin/env bash
# setup-models.sh
# Pulls the Ollama models named in .env. Run once, after installing and
# starting Ollama on the host.
#
# Prerequisite: Ollama running (`ollama serve`, or already active as a
# system service).
#
# Usage:  bash scripts/setup-models.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Load .env when present, otherwise fall back to defaults.
if [ -f "$ROOT/.env" ]; then
  # shellcheck disable=SC1091
  set -a; source "$ROOT/.env"; set +a
fi

OLLAMA_BASE_URL="${OLLAMA_BASE_URL:-http://127.0.0.1:11434}"
EMBEDDING_MODEL="${EMBEDDING_MODEL:-nomic-embed-text}"
LLM_MODEL="${LLM_MODEL:-phi3:mini}"

echo "Waiting for Ollama on $OLLAMA_BASE_URL ..."
for _ in $(seq 1 30); do
  if curl -sf "$OLLAMA_BASE_URL" > /dev/null 2>&1; then
    break
  fi
  echo "  no response yet, retrying in 2s..."
  sleep 2
done

if ! curl -sf "$OLLAMA_BASE_URL" > /dev/null 2>&1; then
  echo "Ollama is not reachable at $OLLAMA_BASE_URL — start it with 'ollama serve'." >&2
  exit 1
fi

echo "Ollama is up."
echo
echo "Pulling embedding model: $EMBEDDING_MODEL"
ollama pull "$EMBEDDING_MODEL"

echo
echo "Pulling generation model: $LLM_MODEL"
ollama pull "$LLM_MODEL"

echo
echo "Installed models:"
ollama list

echo
echo "Next:  make up  &&  make ingest"
