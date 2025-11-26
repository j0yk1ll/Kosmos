"""Test Runner for Kosmos E2E Testing

Automated test execution with provider configuration.
"""

import os
import subprocess
import time
from pathlib import Path
from typing import Optional

from .config_manager import switch_provider, get_current_provider
from .provider_detector import recommend_provider, detect_all


# Project root (Kosmos directory)
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent


# Test tier configurations
TIER_CONFIG = {
    "sanity": {
        "paths": ["tests/smoke/"],
        "markers": None,
        "timeout": 60,
        "description": "Quick validation (~30s)",
    },
    "smoke": {
        "paths": ["tests/unit/", "tests/smoke/"],
        "markers": None,
        "timeout": 300,
        "description": "Unit + smoke tests (~2min)",
    },
    "e2e": {
        "paths": ["tests/e2e/"],
        "markers": "e2e",
        "timeout": 600,
        "description": "Full E2E workflow (~10min)",
    },
    "full": {
        "paths": ["tests/"],
        "markers": None,
        "timeout": 1200,
        "description": "Complete test suite (~20min)",
        "coverage": True,
    },
}


def run_tests(
    tier: str = "sanity",
    provider: str = "auto",
    timeout: Optional[int] = None,
    verbose: bool = True,
    coverage: bool = False,
) -> dict:
    """Run tests with specified configuration

    Args:
        tier: Test tier (sanity, smoke, e2e, full)
        provider: Provider name or 'auto' for auto-detection
        timeout: Max seconds per test (overrides tier default)
        verbose: Show detailed output
        coverage: Generate coverage report

    Returns:
        Dictionary with test results
    """
    if tier not in TIER_CONFIG:
        raise ValueError(f"Unknown tier: {tier}. Choose from: {list(TIER_CONFIG.keys())}")

    config = TIER_CONFIG[tier]

    # Auto-detect provider if requested
    if provider == "auto":
        provider = recommend_provider()
        print(f"Auto-detected provider: {provider}")

    # Switch to the specified provider
    switch_provider(provider)

    # Build pytest command
    cmd = ["pytest"]

    # Add test paths
    for path in config["paths"]:
        full_path = PROJECT_ROOT / path
        if full_path.exists():
            cmd.append(str(full_path))

    # Add markers
    if config.get("markers"):
        cmd.extend(["-m", config["markers"]])

    # Add timeout
    test_timeout = timeout or config.get("timeout", 600)
    cmd.extend(["--timeout", str(test_timeout)])

    # Add verbosity
    if verbose:
        cmd.append("-v")

    # Add coverage
    if coverage or config.get("coverage"):
        cmd.extend(["--cov=kosmos", "--cov-report=term-missing"])

    print(f"\nRunning {tier} tests ({config['description']})")
    print(f"Provider: {provider}")
    print(f"Command: {' '.join(cmd)}")
    print("-" * 60)

    # Run tests
    start_time = time.time()
    result = subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        capture_output=not verbose,
        text=True,
    )
    duration = time.time() - start_time

    # Parse results
    output = result.stdout if result.stdout else ""
    return_code = result.returncode

    # Extract counts from pytest output
    passed = _extract_count(output, "passed")
    failed = _extract_count(output, "failed")
    skipped = _extract_count(output, "skipped")
    errors = _extract_count(output, "error")

    total = passed + failed + skipped + errors

    results = {
        "tier": tier,
        "provider": provider,
        "passed": passed,
        "failed": failed,
        "skipped": skipped,
        "errors": errors,
        "total": total,
        "duration": duration,
        "return_code": return_code,
        "success": return_code == 0,
        "coverage": _extract_coverage(output) if coverage else None,
    }

    print("-" * 60)
    print(f"Results: {passed} passed, {failed} failed, {skipped} skipped")
    print(f"Duration: {duration:.1f}s")

    return results


def run_single_test(
    test_path: str,
    provider: str = "auto",
    verbose: bool = True,
) -> dict:
    """Run a single test file or function

    Args:
        test_path: Path to test file or test::function
        provider: Provider name or 'auto'
        verbose: Show detailed output

    Returns:
        Dictionary with test results
    """
    if provider == "auto":
        provider = recommend_provider()

    switch_provider(provider)

    cmd = ["pytest", test_path, "--timeout=600"]
    if verbose:
        cmd.append("-v")

    start_time = time.time()
    result = subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        capture_output=not verbose,
        text=True,
    )
    duration = time.time() - start_time

    return {
        "test_path": test_path,
        "provider": provider,
        "return_code": result.returncode,
        "success": result.returncode == 0,
        "duration": duration,
        "output": result.stdout if result.stdout else "",
    }


def _extract_count(output: str, status: str) -> int:
    """Extract count from pytest output like '5 passed'"""
    import re
    pattern = rf"(\d+) {status}"
    match = re.search(pattern, output)
    return int(match.group(1)) if match else 0


def _extract_coverage(output: str) -> Optional[float]:
    """Extract coverage percentage from pytest output"""
    import re
    pattern = r"TOTAL\s+\d+\s+\d+\s+(\d+)%"
    match = re.search(pattern, output)
    return float(match.group(1)) if match else None


def list_tests(tier: str = "all") -> list[str]:
    """List available tests

    Args:
        tier: Test tier to list or 'all'

    Returns:
        List of test file paths
    """
    tests = []

    if tier == "all":
        paths = [PROJECT_ROOT / "tests"]
    elif tier in TIER_CONFIG:
        paths = [PROJECT_ROOT / p for p in TIER_CONFIG[tier]["paths"]]
    else:
        raise ValueError(f"Unknown tier: {tier}")

    for path in paths:
        if path.exists():
            for test_file in path.rglob("test_*.py"):
                tests.append(str(test_file.relative_to(PROJECT_ROOT)))

    return sorted(tests)


if __name__ == "__main__":
    import sys

    tier = sys.argv[1] if len(sys.argv) > 1 else "sanity"
    provider = sys.argv[2] if len(sys.argv) > 2 else "auto"

    results = run_tests(tier=tier, provider=provider)
    sys.exit(0 if results["success"] else 1)
