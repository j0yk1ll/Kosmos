#!/bin/bash
# Kosmos E2E Testing - Docker Sandbox Setup
# Auto-setup Docker sandbox for Gap 4 code execution

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$(dirname "$(dirname "$(dirname "$SCRIPT_DIR")")")")")"
SANDBOX_DIR="$PROJECT_ROOT/docker/sandbox"

echo "=========================================="
echo "KOSMOS DOCKER SANDBOX SETUP"
echo "=========================================="

# Check if Docker is available
echo "Checking Docker..."
if ! docker info > /dev/null 2>&1; then
    echo "[ERROR] Docker is not running"
    echo ""
    echo "Please start Docker:"
    echo "  - Linux: sudo systemctl start docker"
    echo "  - macOS/Windows: Start Docker Desktop"
    exit 1
fi
echo "[OK] Docker is running"

# Check if sandbox image exists
echo ""
echo "Checking for kosmos-sandbox image..."
if docker images | grep -q "kosmos-sandbox"; then
    echo "[OK] kosmos-sandbox:latest exists"

    # Verify it works
    echo ""
    echo "Verifying sandbox..."
    if docker run --rm kosmos-sandbox:latest python3 -c "import pandas; print('OK')" 2>/dev/null; then
        echo "[OK] Sandbox is functional"
        exit 0
    else
        echo "[WARN] Sandbox verification failed, rebuilding..."
        docker rmi kosmos-sandbox:latest 2>/dev/null || true
    fi
fi

# Check if Dockerfile exists
if [ ! -f "$SANDBOX_DIR/Dockerfile" ]; then
    echo "[WARN] Dockerfile not found at $SANDBOX_DIR/Dockerfile"
    echo "Creating default Dockerfile..."

    mkdir -p "$SANDBOX_DIR"
    cat > "$SANDBOX_DIR/Dockerfile" << 'EOF'
# Kosmos Sandbox - Secure Python Environment for Code Execution
FROM python:3.11-slim

# Install scientific packages
RUN pip install --no-cache-dir \
    pandas \
    numpy \
    scipy \
    matplotlib \
    seaborn \
    scikit-learn \
    requests \
    aiohttp

# Create non-root user
RUN useradd -m -s /bin/bash sandbox
USER sandbox
WORKDIR /home/sandbox

# Default command
CMD ["python3"]
EOF
fi

# Build sandbox image
echo ""
echo "Building kosmos-sandbox:latest..."
cd "$SANDBOX_DIR"
docker build -t kosmos-sandbox:latest .

# Verify build
echo ""
echo "Verifying build..."
if docker run --rm kosmos-sandbox:latest python3 -c "import pandas; print('Sandbox OK')"; then
    echo ""
    echo "=========================================="
    echo "[SUCCESS] Docker sandbox ready!"
    echo "=========================================="
    echo ""
    echo "Test with:"
    echo "  docker run --rm kosmos-sandbox:latest python3 -c \"print('Hello from sandbox')\""
else
    echo ""
    echo "[ERROR] Sandbox verification failed"
    exit 1
fi
