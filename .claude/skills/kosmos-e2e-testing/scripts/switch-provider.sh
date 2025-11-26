#!/bin/bash
# Kosmos E2E Testing - Provider Switcher
# Usage: source ./switch-provider.sh [provider]
# Providers: local-fast, local-reasoning, anthropic, openai

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SKILL_DIR="$(dirname "$SCRIPT_DIR")"

PROVIDER=${1:-local-fast}

echo "Switching to provider: $PROVIDER"

CONFIG_FILE="$SKILL_DIR/configs/${PROVIDER}.env"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "[ERROR] Config file not found: $CONFIG_FILE"
    echo ""
    echo "Available providers:"
    ls -1 "$SKILL_DIR/configs/"*.env 2>/dev/null | xargs -n1 basename | sed 's/.env$//'
    return 1 2>/dev/null || exit 1
fi

# Source the configuration
set -a
source "$CONFIG_FILE"
set +a

echo "[OK] Provider switched to: $PROVIDER"
echo ""
echo "Current configuration:"
echo "  LLM_PROVIDER=$LLM_PROVIDER"
echo "  OPENAI_MODEL=${OPENAI_MODEL:-N/A}"
echo "  OPENAI_BASE_URL=${OPENAI_BASE_URL:-N/A}"

# Verify provider is available
echo ""
echo "Verifying provider..."

case $PROVIDER in
    local-*)
        if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
            echo "[OK] Ollama is running"
            MODELS=$(curl -s http://localhost:11434/api/tags | python3 -c "import sys,json; data=json.load(sys.stdin); print(', '.join(m['name'] for m in data.get('models',[])))" 2>/dev/null || echo "unknown")
            echo "  Available models: $MODELS"
        else
            echo "[WARN] Ollama is not running"
            echo "  Start with: ollama serve"
        fi
        ;;
    anthropic)
        if [ -n "$ANTHROPIC_API_KEY" ]; then
            echo "[OK] Anthropic API key is set"
        else
            echo "[WARN] ANTHROPIC_API_KEY is not set"
        fi
        ;;
    openai)
        if [ -n "$OPENAI_API_KEY" ] && [ "$OPENAI_API_KEY" != "ollama" ]; then
            echo "[OK] OpenAI API key is set"
        else
            echo "[WARN] OPENAI_API_KEY is not set"
        fi
        ;;
esac

echo ""
echo "Done. Run tests with: ./scripts/run-tests.sh [tier]"
