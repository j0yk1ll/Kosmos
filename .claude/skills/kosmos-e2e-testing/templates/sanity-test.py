#!/usr/bin/env python3
"""Kosmos Sanity Test Template

Quick validation that basic imports and configuration work.
Duration: ~30 seconds
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))))


def test_core_imports():
    """Test that core modules can be imported"""
    print("Testing core imports...")

    try:
        import kosmos
        print("  [OK] kosmos")
    except ImportError as e:
        print(f"  [FAIL] kosmos: {e}")
        return False

    try:
        from kosmos.config import get_settings
        print("  [OK] kosmos.config")
    except ImportError as e:
        print(f"  [FAIL] kosmos.config: {e}")
        return False

    try:
        from kosmos.workflow.research_loop import ResearchWorkflow
        print("  [OK] kosmos.workflow.research_loop")
    except ImportError as e:
        print(f"  [FAIL] kosmos.workflow.research_loop: {e}")
        return False

    try:
        from kosmos.core.llm import get_client
        print("  [OK] kosmos.core.llm")
    except ImportError as e:
        print(f"  [FAIL] kosmos.core.llm: {e}")
        return False

    return True


def test_config_loading():
    """Test that configuration loads correctly"""
    print("\nTesting configuration...")

    try:
        from kosmos.config import get_settings
        settings = get_settings()
        print(f"  [OK] Settings loaded")
        print(f"       LLM Provider: {getattr(settings, 'llm_provider', 'unknown')}")
        return True
    except Exception as e:
        print(f"  [FAIL] Config error: {e}")
        return False


def test_database_connection():
    """Test database connectivity"""
    print("\nTesting database...")

    db_path = "kosmos.db"
    if os.path.exists(db_path):
        print(f"  [OK] Database exists: {db_path}")
        return True
    else:
        print(f"  [WARN] Database not found: {db_path}")
        return True  # Not critical for sanity


def test_provider_detection():
    """Test provider detection library"""
    print("\nTesting provider detection...")

    try:
        skill_lib = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "lib"
        )
        sys.path.insert(0, skill_lib)

        from provider_detector import detect_all, recommend_provider
        status = detect_all()
        provider = recommend_provider(status)

        print(f"  [OK] Detection working")
        print(f"       Ollama: {status['ollama']}")
        print(f"       Docker: {status['docker']}")
        print(f"       Recommended: {provider}")
        return True
    except Exception as e:
        print(f"  [FAIL] Detection error: {e}")
        return False


def main():
    """Run all sanity tests"""
    print("=" * 50)
    print("KOSMOS SANITY TEST")
    print("=" * 50)

    results = []

    results.append(("Core imports", test_core_imports()))
    results.append(("Config loading", test_config_loading()))
    results.append(("Database", test_database_connection()))
    results.append(("Provider detection", test_provider_detection()))

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
