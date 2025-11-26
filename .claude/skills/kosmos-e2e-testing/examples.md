# Kosmos E2E Testing - Examples

Real-world usage examples for the Kosmos E2E testing skill.

---

## Example 1: First-Time Setup

Complete setup from scratch:

```bash
# 1. Verify prerequisites
./scripts/health-check.sh

# 2. Install local models
ollama pull qwen3:4b
ollama pull deepseek-r1:8b

# 3. Set up Docker sandbox
./scripts/setup-docker.sh

# 4. Run sanity check
./scripts/run-tests.sh sanity

# Expected output:
# ✓ Ollama running (qwen3:4b, deepseek-r1:8b)
# ✓ Docker available
# ✓ Sandbox image ready
# ✓ All sanity tests passed
```

---

## Example 2: Quick Validation During Development

Fast iteration while coding:

```bash
# Use fast model for quick checks
source .claude/skills/kosmos-e2e-testing/configs/local-fast.env

# Run only sanity tests (~30s)
./scripts/run-tests.sh sanity local-fast

# Run specific test file
pytest tests/unit/test_config.py -v

# Run tests matching pattern
pytest tests/ -k "test_workflow" -v
```

---

## Example 3: Full E2E Validation

Before merging or releasing:

```bash
# Use reasoning model for thorough testing
./scripts/run-tests.sh e2e local-reasoning

# Or with external API for production validation
export ANTHROPIC_API_KEY=sk-ant-...
./scripts/run-tests.sh e2e anthropic

# Full suite with coverage
./scripts/run-tests.sh full local-reasoning
```

---

## Example 4: Testing Research Workflow

Test the core ResearchWorkflow component:

```python
import asyncio
from kosmos.workflow.research_loop import ResearchWorkflow

async def test_single_cycle():
    """Test a single research cycle"""
    workflow = ResearchWorkflow(
        research_objective="What are the latest developments in quantum computing?",
        artifacts_dir="./test_artifacts"
    )

    result = await workflow.run(
        num_cycles=1,
        tasks_per_cycle=2
    )

    print(f"Cycles completed: {result['cycles_completed']}")
    print(f"Papers analyzed: {result['papers_analyzed']}")
    print(f"Findings: {len(result['findings'])}")

    return result

# Run the test
result = asyncio.run(test_single_cycle())
```

---

## Example 5: Provider Auto-Detection

Let the system choose the best provider:

```python
from lib.provider_detector import detect_all, recommend_provider, recommend_test_tier

# Check what's available
status = detect_all()
print("Detection Results:")
for key, value in status.items():
    print(f"  {key}: {value}")

# Get recommendations
provider = recommend_provider(status)
tier = recommend_test_tier(status)

print(f"\nRecommended provider: {provider}")
print(f"Recommended tier: {tier}")

# Output example:
# Detection Results:
#   ollama: True
#   ollama_models: ['qwen3:4b', 'deepseek-r1:8b']
#   docker: True
#   docker_sandbox: True
#   anthropic: False
#   openai: False
#   database: True
#
# Recommended provider: local-reasoning
# Recommended tier: full_e2e
```

---

## Example 6: Switching Providers

Dynamic provider switching:

```bash
# Switch to fast local model
./scripts/switch-provider.sh local-fast

# Verify switch
echo $OPENAI_MODEL  # Should show: qwen3:4b

# Switch to reasoning model
./scripts/switch-provider.sh local-reasoning

# Switch to Anthropic API
./scripts/switch-provider.sh anthropic
```

Python API:

```python
from lib.config_manager import switch_provider, get_current_provider

# Check current
print(f"Current: {get_current_provider()}")

# Switch
switch_provider('local-reasoning')
print(f"Now: {get_current_provider()}")
```

---

## Example 7: Running Tests Programmatically

Use Python API for test automation:

```python
from lib.test_runner import run_tests

# Run sanity tests
results = run_tests(tier='sanity', provider='local-fast')
print(f"Passed: {results['passed']}/{results['total']}")

# Run E2E with timeout
results = run_tests(
    tier='e2e',
    provider='local-reasoning',
    timeout=900,
    coverage=True
)
print(f"Coverage: {results['coverage']}%")

# Run specific tests
from lib.test_runner import run_single_test
result = run_single_test('tests/e2e/test_autonomous_research.py')
```

---

## Example 8: Docker Sandbox Testing

Test code execution in sandbox:

```bash
# Build sandbox image
./scripts/setup-docker.sh

# Verify sandbox works
docker run --rm kosmos-sandbox:latest python3 -c "
import pandas as pd
import numpy as np
print('Sandbox OK')
print(f'pandas: {pd.__version__}')
print(f'numpy: {np.__version__}')
"

# Run Docker-dependent tests
pytest tests/e2e/ -m docker -v
```

---

## Example 9: Benchmarking Models

Compare local vs API performance:

```python
import time
from lib.test_runner import run_tests

providers = ['local-fast', 'local-reasoning', 'anthropic']
results = {}

for provider in providers:
    start = time.time()
    result = run_tests(tier='smoke', provider=provider)
    elapsed = time.time() - start

    results[provider] = {
        'passed': result['passed'],
        'total': result['total'],
        'duration': elapsed
    }

# Print comparison
print("\nBenchmark Results:")
print("-" * 50)
for provider, data in results.items():
    print(f"{provider}:")
    print(f"  Pass rate: {data['passed']}/{data['total']}")
    print(f"  Duration: {data['duration']:.1f}s")
```

---

## Example 10: CI/CD Integration

GitHub Actions workflow:

```yaml
name: Kosmos E2E Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: pip install -e ".[dev]"

      - name: Run sanity tests
        run: pytest tests/smoke/ -v --timeout=120
        env:
          LLM_PROVIDER: mock

      - name: Run unit tests
        run: pytest tests/unit/ -v --cov=kosmos

      - name: Upload coverage
        uses: codecov/codecov-action@v4
```

---

## Example 11: Debugging Failed Tests

Investigate test failures:

```bash
# Run with maximum verbosity
pytest tests/e2e/test_failing.py -vvv --tb=long

# Show local variables in traceback
pytest tests/e2e/test_failing.py --tb=long --showlocals

# Drop into debugger on failure
pytest tests/e2e/test_failing.py --pdb

# Run only failed tests from last run
pytest --lf -v

# Run last failed first, then rest
pytest --ff -v
```

---

## Example 12: Custom Test Configuration

Create custom test environment:

```bash
# Create custom config
cat > my-test-config.env << 'EOF'
export LLM_PROVIDER=openai
export OPENAI_API_KEY=ollama
export OPENAI_BASE_URL=http://localhost:11434/v1
export OPENAI_MODEL=qwen3:4b
export KOSMOS_TEST_TIMEOUT=300
export KOSMOS_ARTIFACTS_DIR=./custom_artifacts
EOF

# Use custom config
source my-test-config.env
pytest tests/e2e/ -v
```

---

## Example 13: Testing Specific Gaps

Test individual gap implementations:

```python
import pytest

# Test Gap 1: Context Compression
@pytest.mark.e2e
async def test_gap1_context_compression():
    from kosmos.gaps.context_compression import compress_context
    result = await compress_context(large_document)
    assert len(result) < len(large_document)

# Test Gap 4: Code Execution
@pytest.mark.docker
async def test_gap4_code_execution():
    from kosmos.execution.docker_sandbox import execute_code
    result = await execute_code("print('hello')")
    assert result.stdout == "hello\n"

# Run specific gap tests
# pytest tests/e2e/ -k "gap1" -v
# pytest tests/e2e/ -k "gap4" -m docker -v
```

---

## Example 14: Parallel Test Execution

Speed up test suite:

```bash
# Install pytest-xdist
pip install pytest-xdist

# Run tests in parallel (4 workers)
pytest tests/ -n 4 -v

# Auto-detect CPU count
pytest tests/ -n auto -v

# Run E2E tests in parallel
pytest tests/e2e/ -n 2 -m e2e --timeout=900
```

---

## Example 15: Generating Test Reports

Create detailed test reports:

```bash
# HTML report
pytest tests/ --html=report.html --self-contained-html

# JUnit XML (for CI)
pytest tests/ --junitxml=results.xml

# Coverage HTML report
pytest tests/ --cov=kosmos --cov-report=html
open htmlcov/index.html

# Combined
pytest tests/ \
    --html=report.html \
    --junitxml=results.xml \
    --cov=kosmos \
    --cov-report=html \
    --cov-report=term
```

---

## Common Patterns

### Pattern: Fixture for Provider Setup

```python
# tests/conftest.py
import pytest

@pytest.fixture
def local_provider():
    """Set up local Ollama provider"""
    import os
    os.environ['LLM_PROVIDER'] = 'openai'
    os.environ['OPENAI_API_KEY'] = 'ollama'
    os.environ['OPENAI_BASE_URL'] = 'http://localhost:11434/v1'
    os.environ['OPENAI_MODEL'] = 'qwen3:4b'
    yield
    # Cleanup if needed

@pytest.fixture
def sandbox_ready():
    """Ensure Docker sandbox is available"""
    import subprocess
    result = subprocess.run(
        ['docker', 'images', '-q', 'kosmos-sandbox:latest'],
        capture_output=True
    )
    if not result.stdout:
        pytest.skip("Docker sandbox not available")
```

### Pattern: Timeout Decorator

```python
import pytest

@pytest.mark.timeout(120)
async def test_with_custom_timeout():
    """This test has a 2-minute timeout"""
    result = await long_running_operation()
    assert result is not None
```

### Pattern: Conditional Test Skipping

```python
import pytest
import os

@pytest.mark.skipif(
    not os.getenv('ANTHROPIC_API_KEY'),
    reason="Anthropic API key not set"
)
async def test_with_claude():
    """Only runs when Anthropic key is available"""
    pass

@pytest.mark.skipif(
    not check_docker(),
    reason="Docker not available"
)
async def test_docker_execution():
    """Only runs when Docker is available"""
    pass
```
