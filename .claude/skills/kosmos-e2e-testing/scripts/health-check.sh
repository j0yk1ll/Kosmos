#!/bin/bash
# Kosmos E2E Testing - Health Check
# Verify all dependencies and infrastructure are available

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SKILL_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_ROOT="$(dirname "$(dirname "$(dirname "$(dirname "$(dirname "$SCRIPT_DIR")")")")")"

echo "=========================================="
echo "KOSMOS E2E TESTING - HEALTH CHECK"
echo "=========================================="
echo "Project: $PROJECT_ROOT"
echo ""

ISSUES=0
WARNINGS=0

# Check Python
echo "Python Environment"
echo "------------------"
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version 2>&1)
    echo "[OK] $PYTHON_VERSION"
else
    echo "[FAIL] Python3 not found"
    ((ISSUES++))
fi

# Check pytest
if python3 -c "import pytest" 2>/dev/null; then
    PYTEST_VERSION=$(python3 -c "import pytest; print(pytest.__version__)")
    echo "[OK] pytest $PYTEST_VERSION"
else
    echo "[WARN] pytest not installed (pip install pytest)"
    ((WARNINGS++))
fi

# Check Kosmos
echo ""
echo "Kosmos Project"
echo "--------------"
if [ -f "$PROJECT_ROOT/kosmos/__init__.py" ]; then
    echo "[OK] Kosmos package found"
else
    echo "[FAIL] Kosmos package not found at $PROJECT_ROOT/kosmos"
    ((ISSUES++))
fi

if [ -f "$PROJECT_ROOT/kosmos.db" ]; then
    echo "[OK] Database exists (kosmos.db)"
else
    echo "[--] Database not found (kosmos.db)"
fi

# Check Ollama
echo ""
echo "Ollama (Local Models)"
echo "---------------------"
if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "[OK] Ollama is running"

    # List models
    MODELS=$(curl -s http://localhost:11434/api/tags | python3 -c "
import sys, json
data = json.load(sys.stdin)
models = data.get('models', [])
for m in models:
    print(f\"     - {m['name']}\")
" 2>/dev/null)

    if [ -n "$MODELS" ]; then
        echo "[OK] Models installed:"
        echo "$MODELS"

        # Check for recommended models
        if echo "$MODELS" | grep -qi "qwen"; then
            echo "[OK] Fast model (qwen) available"
        else
            echo "[--] Fast model (qwen3:4b) not installed"
            echo "     Install with: ollama pull qwen3:4b"
        fi

        if echo "$MODELS" | grep -qi "deepseek"; then
            echo "[OK] Reasoning model (deepseek) available"
        else
            echo "[--] Reasoning model (deepseek-r1:8b) not installed"
            echo "     Install with: ollama pull deepseek-r1:8b"
        fi
    else
        echo "[WARN] No models installed"
        echo "     Install models: ollama pull qwen3:4b"
        ((WARNINGS++))
    fi
else
    echo "[--] Ollama not running"
    echo "     Start with: ollama serve"
fi

# Check Docker
echo ""
echo "Docker (Sandbox)"
echo "----------------"
if docker info > /dev/null 2>&1; then
    echo "[OK] Docker is running"

    if docker images | grep -q "kosmos-sandbox"; then
        echo "[OK] kosmos-sandbox:latest exists"

        # Verify sandbox
        if docker run --rm kosmos-sandbox:latest python3 -c "import pandas; print('OK')" 2>/dev/null; then
            echo "[OK] Sandbox is functional"
        else
            echo "[WARN] Sandbox verification failed"
            echo "     Rebuild with: ./scripts/setup-docker.sh"
            ((WARNINGS++))
        fi
    else
        echo "[--] kosmos-sandbox:latest not built"
        echo "     Build with: ./scripts/setup-docker.sh"
    fi
else
    echo "[--] Docker not running"
    echo "     Start Docker Desktop or: sudo systemctl start docker"
fi

# Check API Keys
echo ""
echo "External APIs"
echo "-------------"
if [ -n "$ANTHROPIC_API_KEY" ]; then
    KEY_PREVIEW="${ANTHROPIC_API_KEY:0:12}..."
    echo "[OK] ANTHROPIC_API_KEY set ($KEY_PREVIEW)"
else
    echo "[--] ANTHROPIC_API_KEY not set"
fi

if [ -n "$OPENAI_API_KEY" ] && [ "$OPENAI_API_KEY" != "ollama" ]; then
    KEY_PREVIEW="${OPENAI_API_KEY:0:12}..."
    echo "[OK] OPENAI_API_KEY set ($KEY_PREVIEW)"
else
    echo "[--] OPENAI_API_KEY not set"
fi

# Check skill files
echo ""
echo "Skill Installation"
echo "------------------"
if [ -f "$SKILL_DIR/SKILL.md" ]; then
    echo "[OK] Skill installed at $SKILL_DIR"
else
    echo "[FAIL] Skill not properly installed"
    ((ISSUES++))
fi

CONFIG_COUNT=$(ls -1 "$SKILL_DIR/configs/"*.env 2>/dev/null | wc -l)
echo "[OK] $CONFIG_COUNT provider configs available"

TEMPLATE_COUNT=$(ls -1 "$SKILL_DIR/templates/"*.py 2>/dev/null | wc -l)
echo "[OK] $TEMPLATE_COUNT test templates available"

# Summary
echo ""
echo "=========================================="
echo "SUMMARY"
echo "=========================================="

if [ $ISSUES -eq 0 ] && [ $WARNINGS -eq 0 ]; then
    echo "[OK] All checks passed!"
    echo ""
    echo "Recommended next step:"
    echo "  ./scripts/run-tests.sh sanity"
elif [ $ISSUES -eq 0 ]; then
    echo "[OK] Core systems working ($WARNINGS warnings)"
    echo ""
    echo "You can run tests, but some features may be limited."
    echo "  ./scripts/run-tests.sh sanity"
else
    echo "[!!] $ISSUES issues found, $WARNINGS warnings"
    echo ""
    echo "Please resolve issues before running tests."
fi

echo ""
exit $ISSUES
