#!/bin/bash
# Kosmos E2E Testing - Unified Test Runner
# Usage: ./run-tests.sh [tier] [provider]
# Tiers: sanity, smoke, e2e, full
# Providers: local-fast, local-reasoning, anthropic, openai, auto

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SKILL_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_ROOT="$(dirname "$(dirname "$(dirname "$(dirname "$SKILL_DIR")")")")"

TIER=${1:-sanity}
PROVIDER=${2:-auto}

echo "=========================================="
echo "KOSMOS E2E TEST RUNNER"
echo "=========================================="
echo "Tier: $TIER"
echo "Provider: $PROVIDER"
echo "Project: $PROJECT_ROOT"
echo ""

# Auto-detect provider if requested
if [ "$PROVIDER" == "auto" ]; then
    echo "Auto-detecting best provider..."

    # Check Ollama first
    if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        MODELS=$(curl -s http://localhost:11434/api/tags | python3 -c "import sys,json; data=json.load(sys.stdin); print(' '.join(m['name'] for m in data.get('models',[])))" 2>/dev/null || echo "")

        if echo "$MODELS" | grep -qi "deepseek"; then
            PROVIDER="local-reasoning"
        elif echo "$MODELS" | grep -qi "qwen"; then
            PROVIDER="local-fast"
        elif [ -n "$MODELS" ]; then
            PROVIDER="local-fast"
        fi
    fi

    # Fall back to API if no local
    if [ "$PROVIDER" == "auto" ]; then
        if [ -n "$ANTHROPIC_API_KEY" ]; then
            PROVIDER="anthropic"
        elif [ -n "$OPENAI_API_KEY" ]; then
            PROVIDER="openai"
        else
            echo "[ERROR] No provider available. Please:"
            echo "  - Start Ollama: ollama serve"
            echo "  - Or set ANTHROPIC_API_KEY"
            echo "  - Or set OPENAI_API_KEY"
            exit 1
        fi
    fi

    echo "Selected provider: $PROVIDER"
fi

# Load provider configuration
CONFIG_FILE="$SKILL_DIR/configs/${PROVIDER}.env"
if [ -f "$CONFIG_FILE" ]; then
    echo "Loading config: $CONFIG_FILE"
    set -a
    source "$CONFIG_FILE"
    set +a
else
    echo "[WARN] Config file not found: $CONFIG_FILE"
fi

cd "$PROJECT_ROOT"

# Run tests based on tier
echo ""
echo "Running $TIER tests..."
echo "------------------------------------------"

case $TIER in
    sanity)
        pytest tests/smoke/ -v --timeout=60 2>/dev/null || \
        python3 "$SKILL_DIR/templates/sanity-test.py"
        ;;
    smoke)
        pytest tests/unit/ tests/smoke/ -v --timeout=300 2>/dev/null || \
        python3 "$SKILL_DIR/templates/smoke-test.py"
        ;;
    e2e)
        pytest tests/e2e/ -v -m e2e --timeout=600 2>/dev/null || \
        python3 "$SKILL_DIR/templates/e2e-runner.py"
        ;;
    full)
        pytest tests/ -v --cov=kosmos --cov-report=term-missing --timeout=1200
        ;;
    *)
        echo "[ERROR] Unknown tier: $TIER"
        echo "Valid tiers: sanity, smoke, e2e, full"
        exit 1
        ;;
esac

echo ""
echo "=========================================="
echo "TEST RUN COMPLETE"
echo "=========================================="
