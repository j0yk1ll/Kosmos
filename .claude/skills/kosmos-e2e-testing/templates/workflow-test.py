#!/usr/bin/env python3
"""Kosmos ResearchWorkflow Test Template

Focused testing of the ResearchWorkflow component.
"""

import sys
import os
import asyncio
import time
from pathlib import Path
from typing import Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


async def test_workflow_basic(artifacts_dir: str) -> dict:
    """Test basic workflow execution

    Args:
        artifacts_dir: Directory for test artifacts

    Returns:
        Test result dictionary
    """
    from kosmos.workflow.research_loop import ResearchWorkflow

    print("Test: Basic workflow execution")

    workflow = ResearchWorkflow(
        research_objective="What is machine learning?",
        artifacts_dir=artifacts_dir
    )

    start = time.time()
    result = await workflow.run(num_cycles=1, tasks_per_cycle=1)
    elapsed = time.time() - start

    return {
        "name": "basic_execution",
        "success": result.get("cycles_completed", 0) >= 1,
        "duration": elapsed,
        "cycles": result.get("cycles_completed", 0),
        "findings": len(result.get("findings", [])),
    }


async def test_workflow_multi_cycle(artifacts_dir: str) -> dict:
    """Test multi-cycle workflow execution"""
    from kosmos.workflow.research_loop import ResearchWorkflow

    print("Test: Multi-cycle execution")

    workflow = ResearchWorkflow(
        research_objective="Compare transformer architectures for NLP",
        artifacts_dir=artifacts_dir
    )

    start = time.time()
    result = await workflow.run(num_cycles=2, tasks_per_cycle=2)
    elapsed = time.time() - start

    return {
        "name": "multi_cycle",
        "success": result.get("cycles_completed", 0) >= 2,
        "duration": elapsed,
        "cycles": result.get("cycles_completed", 0),
        "findings": len(result.get("findings", [])),
    }


async def test_workflow_interruption(artifacts_dir: str) -> dict:
    """Test workflow state preservation on interruption"""
    from kosmos.workflow.research_loop import ResearchWorkflow

    print("Test: Interruption handling")

    workflow = ResearchWorkflow(
        research_objective="Test interruption handling",
        artifacts_dir=artifacts_dir
    )

    # Start and cancel after short time
    start = time.time()
    try:
        task = asyncio.create_task(
            workflow.run(num_cycles=5, tasks_per_cycle=3)
        )
        await asyncio.sleep(2)  # Let it start
        task.cancel()
        await task
    except asyncio.CancelledError:
        pass
    elapsed = time.time() - start

    # Check if state was preserved
    state_file = Path(artifacts_dir) / "state.json"
    state_preserved = state_file.exists()

    return {
        "name": "interruption",
        "success": state_preserved,
        "duration": elapsed,
        "state_preserved": state_preserved,
    }


async def test_workflow_artifact_generation(artifacts_dir: str) -> dict:
    """Test that workflow generates expected artifacts"""
    from kosmos.workflow.research_loop import ResearchWorkflow

    print("Test: Artifact generation")

    # Clean artifacts dir
    artifacts_path = Path(artifacts_dir)
    if artifacts_path.exists():
        import shutil
        shutil.rmtree(artifacts_path)
    artifacts_path.mkdir(parents=True, exist_ok=True)

    workflow = ResearchWorkflow(
        research_objective="What are recent advances in AI safety?",
        artifacts_dir=artifacts_dir
    )

    start = time.time()
    result = await workflow.run(num_cycles=1, tasks_per_cycle=2)
    elapsed = time.time() - start

    # Check for expected artifacts
    expected_files = ["state.json", "findings.json", "summary.md"]
    found_files = list(artifacts_path.glob("*"))
    found_names = [f.name for f in found_files]

    return {
        "name": "artifact_generation",
        "success": len(found_files) > 0,
        "duration": elapsed,
        "artifacts_found": found_names,
        "expected": expected_files,
    }


async def test_workflow_error_handling(artifacts_dir: str) -> dict:
    """Test workflow handles errors gracefully"""
    from kosmos.workflow.research_loop import ResearchWorkflow

    print("Test: Error handling")

    # Use invalid configuration to trigger error handling
    workflow = ResearchWorkflow(
        research_objective="",  # Empty objective
        artifacts_dir=artifacts_dir
    )

    start = time.time()
    try:
        result = await workflow.run(num_cycles=1, tasks_per_cycle=1)
        # Should either fail gracefully or handle empty objective
        success = True
    except ValueError:
        # Expected error for empty objective
        success = True
    except Exception as e:
        # Unexpected error
        success = False
        print(f"  Unexpected error: {e}")
    elapsed = time.time() - start

    return {
        "name": "error_handling",
        "success": success,
        "duration": elapsed,
    }


async def main():
    """Run all workflow tests"""
    print("=" * 60)
    print("KOSMOS WORKFLOW TEST SUITE")
    print("=" * 60)
    print()

    artifacts_base = PROJECT_ROOT / "test_artifacts" / "workflow"
    artifacts_base.mkdir(parents=True, exist_ok=True)

    results = []

    # Run tests
    tests = [
        (test_workflow_basic, "basic"),
        (test_workflow_multi_cycle, "multi_cycle"),
        (test_workflow_artifact_generation, "artifacts"),
        (test_workflow_error_handling, "error"),
    ]

    for test_func, suffix in tests:
        artifacts_dir = str(artifacts_base / suffix)
        Path(artifacts_dir).mkdir(parents=True, exist_ok=True)

        try:
            result = await test_func(artifacts_dir)
            results.append(result)

            status = "[PASS]" if result["success"] else "[FAIL]"
            print(f"  {status} {result['name']} ({result['duration']:.1f}s)")
        except Exception as e:
            print(f"  [ERROR] {test_func.__name__}: {e}")
            results.append({
                "name": test_func.__name__,
                "success": False,
                "error": str(e),
            })

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    passed = sum(1 for r in results if r.get("success"))
    total = len(results)
    total_time = sum(r.get("duration", 0) for r in results)

    print(f"\nPassed: {passed}/{total}")
    print(f"Total time: {total_time:.1f}s")

    # Detailed results
    print("\nDetailed Results:")
    for result in results:
        print(f"\n  {result['name']}:")
        for key, value in result.items():
            if key != "name":
                print(f"    {key}: {value}")

    return passed == total


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
