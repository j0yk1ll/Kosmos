# Kosmos E2E Testing - Technical Reference

Detailed technical documentation for the Kosmos E2E testing skill.

---

## Provider Detection API

### `lib/provider_detector.py`

```python
def check_ollama() -> bool
    """Check if Ollama is running on localhost:11434"""

def list_ollama_models() -> list[str]
    """Return list of installed Ollama models"""

def check_docker() -> bool
    """Check if Docker daemon is running"""

def check_sandbox() -> bool
    """Check if kosmos-sandbox:latest image exists"""

def check_database() -> bool
    """Check if SQLite database is accessible"""

def detect_all() -> dict
    """Detect all available testing infrastructure

    Returns:
        {
            'ollama': bool,
            'ollama_models': list[str],
            'docker': bool,
            'docker_sandbox': bool,
            'anthropic': bool,
            'openai': bool,
            'database': bool
        }
    """

def recommend_provider(detection: dict) -> str
    """Recommend best provider based on detection

    Priority: local-reasoning > local-fast > anthropic > openai > mock
    """

def recommend_test_tier(detection: dict) -> str
    """Recommend test tier based on infrastructure

    Returns: 'full_e2e' | 'partial_e2e' | 'api_only' | 'mock_only'
    """
```

---

## Test Runner API

### `lib/test_runner.py`

```python
def run_tests(
    tier: str = 'sanity',
    provider: str = 'auto',
    timeout: int = 600,
    verbose: bool = True,
    coverage: bool = False
) -> dict
    """Run tests with specified configuration

    Args:
        tier: 'sanity' | 'smoke' | 'e2e' | 'full'
        provider: 'local-fast' | 'local-reasoning' | 'anthropic' | 'openai' | 'auto'
        timeout: Max seconds per test
        verbose: Show detailed output
        coverage: Generate coverage report

    Returns:
        {
            'passed': int,
            'failed': int,
            'skipped': int,
            'total': int,
            'duration': float,
            'coverage': float | None
        }
    """

def run_single_test(
    test_path: str,
    provider: str = 'auto'
) -> dict
    """Run a single test file or function"""
```

---

## Config Manager API

### `lib/config_manager.py`

```python
def load_config(provider: str) -> dict
    """Load environment configuration for provider

    Reads from configs/{provider}.env
    """

def switch_provider(provider: str) -> None
    """Switch active provider by setting environment variables"""

def get_current_provider() -> str
    """Get currently configured provider"""

def validate_config(provider: str) -> tuple[bool, list[str]]
    """Validate provider configuration

    Returns:
        (is_valid, list_of_errors)
    """
```

---

## Environment Variables

### Core Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `LLM_PROVIDER` | Active provider type | `openai`, `anthropic` |
| `OPENAI_API_KEY` | OpenAI/Ollama API key | `sk-...` or `ollama` |
| `OPENAI_BASE_URL` | API endpoint URL | `http://localhost:11434/v1` |
| `OPENAI_MODEL` | Model identifier | `qwen3:4b`, `gpt-4` |
| `ANTHROPIC_API_KEY` | Anthropic API key | `sk-ant-...` |
| `ANTHROPIC_MODEL` | Claude model | `claude-sonnet-4-20250514` |

### Test Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `KOSMOS_TEST_TIMEOUT` | Per-test timeout (seconds) | `600` |
| `KOSMOS_TEST_TIER` | Default test tier | `sanity` |
| `KOSMOS_SANDBOX_IMAGE` | Docker image name | `kosmos-sandbox:latest` |
| `KOSMOS_ARTIFACTS_DIR` | Test artifact directory | `./test_artifacts` |

---

## Test Markers

### Available Markers

| Marker | Description | Usage |
|--------|-------------|-------|
| `@pytest.mark.e2e` | End-to-end tests | Full workflow tests |
| `@pytest.mark.slow` | Long-running tests | Tests > 60s |
| `@pytest.mark.docker` | Requires Docker | Sandbox tests |
| `@pytest.mark.unit` | Unit tests | Fast, isolated |
| `@pytest.mark.integration` | Integration tests | Component interaction |
| `@pytest.mark.smoke` | Smoke tests | Basic functionality |

### Marker Combinations

```bash
# Run only E2E tests
pytest -m e2e

# Run fast tests (exclude slow)
pytest -m "not slow"

# Run Docker-dependent tests
pytest -m docker

# Run integration but not E2E
pytest -m "integration and not e2e"
```

---

## Model Specifications

### Qwen3 4B (Fast)

| Spec | Value |
|------|-------|
| **Model ID** | `qwen3:4b` |
| **Parameters** | 4B |
| **Context Length** | 32K |
| **Speed** | 30-40 tok/s |
| **VRAM** | 2-3 GB |
| **Best For** | Sanity, smoke, rapid iteration |

### DeepSeek-R1 8B (Reasoning)

| Spec | Value |
|------|-------|
| **Model ID** | `deepseek-r1:8b` |
| **Parameters** | 8B |
| **Context Length** | 64K |
| **Speed** | 6-7 tok/s |
| **VRAM** | 5-6 GB |
| **Best For** | E2E, complex reasoning, validation |

---

## Docker Sandbox Specifications

### Image: `kosmos-sandbox:latest`

**Base**: Python 3.11 slim

**Pre-installed Packages**:
- pandas, numpy, scipy
- matplotlib, seaborn
- scikit-learn
- requests, aiohttp
- pytest

**Security**:
- No network access by default
- Read-only root filesystem
- Non-root user
- Resource limits (CPU, memory)

**Build Command**:
```bash
cd docker/sandbox
docker build -t kosmos-sandbox:latest .
```

---

## Test Tier Specifications

### Sanity (~30s)

```python
# Runs:
- tests/smoke/test_imports.py
- tests/smoke/test_config.py

# Validates:
- All modules importable
- Config loads correctly
- Mock workflow initializes
```

### Smoke (~2min)

```python
# Runs:
- tests/unit/
- tests/smoke/

# Validates:
- Unit test suite passes
- Component initialization
- Provider connectivity
```

### E2E (~10min)

```python
# Runs:
- tests/e2e/ -m e2e

# Validates:
- Full research workflow
- All gaps functional
- Real LLM interaction
- Docker sandbox execution
```

### Full (~20min)

```python
# Runs:
- tests/ --cov=kosmos

# Validates:
- All tests pass
- Coverage threshold met
- No regressions
```

---

## Error Codes

| Code | Meaning | Resolution |
|------|---------|------------|
| `E001` | Ollama not running | Start Ollama: `ollama serve` |
| `E002` | Model not found | Pull model: `ollama pull <model>` |
| `E003` | Docker not available | Start Docker daemon |
| `E004` | Sandbox missing | Run: `./scripts/setup-docker.sh` |
| `E005` | API key invalid | Check environment variables |
| `E006` | Timeout exceeded | Increase timeout or use faster model |
| `E007` | Database locked | Close other Kosmos instances |

---

## Pytest Configuration

### Recommended `pytest.ini` Settings

```ini
[pytest]
testpaths = tests
markers =
    e2e: End-to-end tests
    slow: Long-running tests
    docker: Requires Docker
    unit: Unit tests
    integration: Integration tests
    smoke: Smoke tests
timeout = 600
asyncio_mode = auto
```

### Coverage Configuration

```ini
[coverage:run]
source = kosmos
omit =
    */tests/*
    */migrations/*

[coverage:report]
fail_under = 70
show_missing = true
```

---

## File Locations

| Path | Purpose |
|------|---------|
| `.claude/skills/kosmos-e2e-testing/` | Skill root |
| `tests/` | All test files |
| `tests/e2e/` | E2E test suite |
| `tests/conftest.py` | Shared fixtures |
| `docker/sandbox/` | Sandbox Dockerfile |
| `.env` | Environment configuration |
| `kosmos.db` | SQLite database |
| `test_artifacts/` | Test output directory |

---

## Integration Points

### With local-llm Skill

- Same model naming: `qwen3:4b`, `deepseek-r1:8b`
- Compatible environment variable patterns
- Cross-references via skill triggers

### With Kosmos Core

- `kosmos/workflow/research_loop.py` - Main workflow
- `kosmos/core/providers/` - Provider implementations
- `kosmos/config.py` - Configuration loading
- `kosmos/execution/docker_sandbox.py` - Sandbox execution
