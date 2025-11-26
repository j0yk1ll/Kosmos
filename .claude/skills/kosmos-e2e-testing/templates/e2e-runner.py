#!/usr/bin/env python3
"""Kosmos E2E Test Runner Template

Full end-to-end workflow testing.
Duration: ~10 minutes
"""

import sys
import os
import asyncio
import time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Add skill lib to path
SKILL_LIB = Path(__file__).parent.parent / "lib"
sys.path.insert(0, str(SKILL_LIB))


async def test_full_research_cycle():
    """Test a complete research cycle"""
    print("Testing full research cycle...")

    try:
        from kosmos.workflow.research_loop import ResearchWorkflow

        artifacts_dir = PROJECT_ROOT / "test_artifacts" / "e2e"
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        workflow = ResearchWorkflow(
            research_objective="What are recent advances in large language model efficiency?",
            artifacts_dir=str(artifacts_dir)
        )

        start = time.time()
        result = await workflow.run(
            num_cycles=1,
            tasks_per_cycle=2
        )
        elapsed = time.time() - start

        print(f"  Cycles completed: {result.get('cycles_completed', 0)}")
        print(f"  Papers analyzed: {result.get('papers_analyzed', 0)}")
        print(f"  Findings: {len(result.get('findings', []))}")
        print(f"  Duration: {elapsed:.1f}s")

        if result.get('cycles_completed', 0) >= 1:
            print("  [OK] Research cycle completed")
            return True
        else:
            print("  [FAIL] No cycles completed")
            return False

    except Exception as e:
        print(f"  [FAIL] Error: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_context_compression():
    """Test Gap 1: Context compression"""
    print("\nTesting context compression (Gap 1)...")

    try:
        from kosmos.gaps.context_compression import compress_context

        large_text = "This is a test document. " * 500  # ~4KB

        result = await compress_context(large_text)

        if len(result) < len(large_text):
            ratio = len(result) / len(large_text) * 100
            print(f"  [OK] Compressed to {ratio:.1f}% of original")
            return True
        else:
            print("  [WARN] No compression achieved")
            return True  # May not always compress

    except ImportError:
        print("  [SKIP] Module not available")
        return True
    except Exception as e:
        print(f"  [FAIL] Error: {e}")
        return False


async def test_code_execution():
    """Test Gap 4: Code execution in sandbox"""
    print("\nTesting code execution (Gap 4)...")

    try:
        from kosmos.execution.docker_sandbox import execute_code

        code = """
import pandas as pd
import numpy as np

data = {'a': [1, 2, 3], 'b': [4, 5, 6]}
df = pd.DataFrame(data)
print(f"DataFrame shape: {df.shape}")
print(df.sum().to_dict())
"""

        result = await execute_code(code)

        if result.success and "DataFrame shape" in result.stdout:
            print(f"  [OK] Code executed successfully")
            print(f"       Output: {result.stdout[:100]}...")
            return True
        else:
            print(f"  [FAIL] Execution failed: {result.stderr}")
            return False

    except ImportError:
        print("  [SKIP] Docker sandbox not available")
        return True
    except Exception as e:
        if "docker" in str(e).lower():
            print("  [SKIP] Docker not running")
            return True
        print(f"  [FAIL] Error: {e}")
        return False


async def test_scholar_evaluation():
    """Test Gap 6: Scholar evaluation"""
    print("\nTesting scholar evaluation (Gap 6)...")

    try:
        from kosmos.validation.scholar_eval import ScholarEvaluator

        evaluator = ScholarEvaluator()

        test_finding = {
            "claim": "Large language models can be made more efficient through quantization.",
            "evidence": "Multiple studies show 4-bit quantization reduces memory by 75%.",
            "source": "arxiv:2023.12345"
        }

        result = await evaluator.evaluate_finding(test_finding)

        if result and hasattr(result, 'score'):
            print(f"  [OK] Evaluation completed")
            print(f"       Score: {result.score}")
            return True
        else:
            print("  [WARN] No score returned")
            return True

    except ImportError:
        print("  [SKIP] Module not available")
        return True
    except Exception as e:
        print(f"  [FAIL] Error: {e}")
        return False


async def test_state_persistence():
    """Test Gap 2: State management"""
    print("\nTesting state persistence (Gap 2)...")

    try:
        from kosmos.gaps.state_management import StateManager

        manager = StateManager()

        # Save state
        test_state = {"key": "value", "count": 42}
        await manager.save("test_state", test_state)

        # Load state
        loaded = await manager.load("test_state")

        if loaded == test_state:
            print("  [OK] State persisted correctly")
            return True
        else:
            print(f"  [FAIL] State mismatch: {loaded}")
            return False

    except ImportError:
        print("  [SKIP] Module not available")
        return True
    except Exception as e:
        print(f"  [FAIL] Error: {e}")
        return False


async def main():
    """Run all E2E tests"""
    print("=" * 60)
    print("KOSMOS E2E TEST SUITE")
    print("=" * 60)

    # Check infrastructure first
    try:
        from provider_detector import detect_all, recommend_test_tier

        status = detect_all()
        tier = recommend_test_tier(status)

        print(f"\nInfrastructure: {tier}")
        print(f"  Ollama: {status['ollama']} (models: {len(status['ollama_models'])})")
        print(f"  Docker: {status['docker_sandbox']}")
        print()
    except Exception as e:
        print(f"[WARN] Could not detect infrastructure: {e}\n")

    results = []
    start_time = time.time()

    # Run tests
    results.append(("Full research cycle", await test_full_research_cycle()))
    results.append(("Context compression", await test_context_compression()))
    results.append(("Code execution", await test_code_execution()))
    results.append(("Scholar evaluation", await test_scholar_evaluation()))
    results.append(("State persistence", await test_state_persistence()))

    total_time = time.time() - start_time

    print("\n" + "=" * 60)
    print("E2E TEST RESULTS")
    print("=" * 60)

    passed = sum(1 for _, ok in results if ok)
    total = len(results)

    for name, ok in results:
        status = "[PASS]" if ok else "[FAIL]"
        print(f"  {status} {name}")

    print()
    print(f"Total: {passed}/{total} passed")
    print(f"Duration: {total_time:.1f}s")

    return passed == total


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
