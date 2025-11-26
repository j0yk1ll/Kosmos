"""Provider Detection for Kosmos E2E Testing

Auto-detect available testing infrastructure including Ollama, Docker, and API keys.
"""

import os
import subprocess
import urllib.request
import json
from typing import Optional


def check_ollama() -> bool:
    """Check if Ollama is running on localhost:11434"""
    try:
        req = urllib.request.Request(
            "http://localhost:11434/api/tags",
            method="GET"
        )
        with urllib.request.urlopen(req, timeout=5) as response:
            return response.status == 200
    except Exception:
        return False


def list_ollama_models() -> list[str]:
    """Return list of installed Ollama models"""
    try:
        req = urllib.request.Request(
            "http://localhost:11434/api/tags",
            method="GET"
        )
        with urllib.request.urlopen(req, timeout=5) as response:
            data = json.loads(response.read().decode())
            return [model["name"] for model in data.get("models", [])]
    except Exception:
        return []


def check_docker() -> bool:
    """Check if Docker daemon is running"""
    try:
        result = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            timeout=10
        )
        return result.returncode == 0
    except Exception:
        return False


def check_sandbox() -> bool:
    """Check if kosmos-sandbox:latest image exists"""
    try:
        result = subprocess.run(
            ["docker", "images", "-q", "kosmos-sandbox:latest"],
            capture_output=True,
            timeout=10
        )
        return bool(result.stdout.strip())
    except Exception:
        return False


def check_database() -> bool:
    """Check if SQLite database is accessible"""
    db_path = os.path.join(os.getcwd(), "kosmos.db")
    return os.path.exists(db_path)


def detect_all() -> dict:
    """Detect all available testing infrastructure

    Returns:
        Dictionary with detection results for each component
    """
    ollama_running = check_ollama()

    return {
        "ollama": ollama_running,
        "ollama_models": list_ollama_models() if ollama_running else [],
        "docker": check_docker(),
        "docker_sandbox": check_sandbox(),
        "anthropic": bool(os.getenv("ANTHROPIC_API_KEY")),
        "openai": bool(os.getenv("OPENAI_API_KEY")),
        "database": check_database(),
    }


def recommend_provider(detection: Optional[dict] = None) -> str:
    """Recommend best provider based on detection

    Priority: local-reasoning > local-fast > anthropic > openai > mock

    Args:
        detection: Detection results from detect_all(). If None, runs detection.

    Returns:
        Provider name string
    """
    if detection is None:
        detection = detect_all()

    models = detection.get("ollama_models", [])

    # Check for local reasoning model first (best for E2E)
    if detection["ollama"]:
        if any("deepseek" in m.lower() for m in models):
            return "local-reasoning"
        if any("qwen" in m.lower() for m in models):
            return "local-fast"
        if models:  # Any model available
            return "local-fast"

    # Fall back to external APIs
    if detection["anthropic"]:
        return "anthropic"
    if detection["openai"]:
        return "openai"

    return "mock"


def recommend_test_tier(detection: Optional[dict] = None) -> str:
    """Recommend test tier based on available infrastructure

    Args:
        detection: Detection results from detect_all(). If None, runs detection.

    Returns:
        'full_e2e' | 'partial_e2e' | 'api_only' | 'mock_only'
    """
    if detection is None:
        detection = detect_all()

    has_llm = detection["ollama"] or detection["anthropic"] or detection["openai"]
    has_docker = detection["docker_sandbox"]

    if has_llm and has_docker:
        return "full_e2e"
    elif has_llm:
        return "partial_e2e"
    elif detection["anthropic"] or detection["openai"]:
        return "api_only"
    return "mock_only"


def print_status(detection: Optional[dict] = None) -> None:
    """Print formatted status of all components"""
    if detection is None:
        detection = detect_all()

    print("Kosmos E2E Testing - Infrastructure Status")
    print("=" * 50)

    # Ollama
    if detection["ollama"]:
        models = ", ".join(detection["ollama_models"]) or "no models"
        print(f"[OK] Ollama: Running ({models})")
    else:
        print("[--] Ollama: Not running")

    # Docker
    if detection["docker"]:
        if detection["docker_sandbox"]:
            print("[OK] Docker: Running (sandbox ready)")
        else:
            print("[!!] Docker: Running (sandbox missing)")
    else:
        print("[--] Docker: Not available")

    # APIs
    if detection["anthropic"]:
        print("[OK] Anthropic: API key set")
    else:
        print("[--] Anthropic: No API key")

    if detection["openai"]:
        print("[OK] OpenAI: API key set")
    else:
        print("[--] OpenAI: No API key")

    # Database
    if detection["database"]:
        print("[OK] Database: kosmos.db exists")
    else:
        print("[--] Database: kosmos.db not found")

    print("=" * 50)
    print(f"Recommended provider: {recommend_provider(detection)}")
    print(f"Recommended tier: {recommend_test_tier(detection)}")


if __name__ == "__main__":
    print_status()
