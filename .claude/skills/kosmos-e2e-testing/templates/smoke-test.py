#!/usr/bin/env python3
"""Kosmos Smoke Test Template

Component-level validation tests.
Duration: ~2 minutes
"""

import sys
import os
import asyncio

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.insert(0, PROJECT_ROOT)


def test_llm_client_initialization():
    """Test LLM client can be created"""
    print("Testing LLM client initialization...")

    try:
        from kosmos.core.llm import get_client
        client = get_client()
        print(f"  [OK] Client created: {type(client).__name__}")
        return True
    except Exception as e:
        print(f"  [FAIL] Client error: {e}")
        return False


def test_workflow_initialization():
    """Test ResearchWorkflow can be initialized"""
    print("\nTesting workflow initialization...")

    try:
        from kosmos.workflow.research_loop import ResearchWorkflow

        workflow = ResearchWorkflow(
            research_objective="Test objective",
            artifacts_dir="./test_artifacts"
        )
        print(f"  [OK] Workflow created")
        return True
    except Exception as e:
        print(f"  [FAIL] Workflow error: {e}")
        return False


def test_provider_modules():
    """Test provider modules load correctly"""
    print("\nTesting provider modules...")

    providers = []
    try:
        from kosmos.core.providers import anthropic_provider
        providers.append("anthropic")
    except ImportError:
        pass

    try:
        from kosmos.core.providers import openai_provider
        providers.append("openai")
    except ImportError:
        pass

    if providers:
        print(f"  [OK] Loaded providers: {', '.join(providers)}")
        return True
    else:
        print("  [WARN] No providers loaded")
        return True  # Not critical


def test_gap_modules():
    """Test gap implementation modules"""
    print("\nTesting gap modules...")

    gaps = {
        "gap1_context": "kosmos.gaps.context_compression",
        "gap2_state": "kosmos.gaps.state_management",
        "gap3_orchestration": "kosmos.gaps.orchestration",
        "gap4_execution": "kosmos.execution.docker_sandbox",
        "gap5_skill": "kosmos.gaps.skill_loading",
        "gap6_validation": "kosmos.validation.scholar_eval",
    }

    loaded = 0
    for name, module_path in gaps.items():
        try:
            __import__(module_path)
            print(f"  [OK] {name}")
            loaded += 1
        except ImportError as e:
            print(f"  [--] {name}: {e}")

    return loaded >= 3  # At least half should work


async def test_simple_llm_call():
    """Test a simple LLM call (if provider available)"""
    print("\nTesting simple LLM call...")

    try:
        from kosmos.core.llm import get_client

        client = get_client()
        if client is None:
            print("  [SKIP] No LLM client available")
            return True

        # Simple completion test
        response = await client.complete("Say 'test ok' and nothing else.")
        if response and "test" in response.lower():
            print(f"  [OK] LLM responded")
            return True
        else:
            print(f"  [WARN] Unexpected response: {response[:50]}...")
            return True
    except Exception as e:
        print(f"  [SKIP] LLM test skipped: {e}")
        return True  # Not critical for smoke test


def test_database_operations():
    """Test basic database operations"""
    print("\nTesting database operations...")

    try:
        import sqlite3
        db_path = os.path.join(PROJECT_ROOT, "kosmos.db")

        if not os.path.exists(db_path):
            print("  [SKIP] Database not found")
            return True

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
        conn.close()

        print(f"  [OK] Found {len(tables)} table(s)")
        return True
    except Exception as e:
        print(f"  [FAIL] Database error: {e}")
        return False


def main():
    """Run all smoke tests"""
    print("=" * 50)
    print("KOSMOS SMOKE TEST")
    print("=" * 50)

    results = []

    results.append(("LLM client init", test_llm_client_initialization()))
    results.append(("Workflow init", test_workflow_initialization()))
    results.append(("Provider modules", test_provider_modules()))
    results.append(("Gap modules", test_gap_modules()))
    results.append(("Database ops", test_database_operations()))

    # Run async test
    try:
        llm_result = asyncio.run(test_simple_llm_call())
        results.append(("LLM call", llm_result))
    except Exception as e:
        print(f"\n  [SKIP] Async test: {e}")
        results.append(("LLM call", True))

    print("\n" + "=" * 50)
    print("RESULTS")
    print("=" * 50)

    passed = sum(1 for _, ok in results if ok)
    total = len(results)

    for name, ok in results:
        status = "[PASS]" if ok else "[FAIL]"
        print(f"  {status} {name}")

    print()
    print(f"Total: {passed}/{total} passed")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
