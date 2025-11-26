#!/usr/bin/env python3
"""Kosmos Model Benchmark Template

Compare performance across different providers.
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


BENCHMARK_PROMPTS = [
    {
        "name": "simple",
        "prompt": "What is 2 + 2? Reply with just the number.",
        "expected_tokens": 5,
    },
    {
        "name": "reasoning",
        "prompt": "Explain why the sky is blue in exactly 3 sentences.",
        "expected_tokens": 100,
    },
    {
        "name": "coding",
        "prompt": "Write a Python function that checks if a number is prime. Return only code.",
        "expected_tokens": 200,
    },
    {
        "name": "analysis",
        "prompt": "Analyze the pros and cons of using local LLMs vs API-based LLMs for testing.",
        "expected_tokens": 300,
    },
]


async def benchmark_provider(provider: str, prompts: list) -> dict:
    """Benchmark a single provider

    Args:
        provider: Provider name
        prompts: List of prompt dictionaries

    Returns:
        Benchmark results
    """
    from config_manager import switch_provider
    from kosmos.core.llm import get_client

    switch_provider(provider)
    client = get_client()

    if client is None:
        return {"provider": provider, "error": "Client not available"}

    results = {
        "provider": provider,
        "prompts": [],
        "total_time": 0,
        "total_tokens": 0,
    }

    for prompt_data in prompts:
        start = time.time()

        try:
            response = await client.complete(prompt_data["prompt"])
            elapsed = time.time() - start

            # Estimate tokens (rough)
            tokens = len(response.split()) if response else 0

            results["prompts"].append({
                "name": prompt_data["name"],
                "time": elapsed,
                "tokens": tokens,
                "tokens_per_second": tokens / elapsed if elapsed > 0 else 0,
                "success": True,
            })

            results["total_time"] += elapsed
            results["total_tokens"] += tokens

        except Exception as e:
            results["prompts"].append({
                "name": prompt_data["name"],
                "error": str(e),
                "success": False,
            })

    # Calculate averages
    successful = [p for p in results["prompts"] if p.get("success")]
    if successful:
        results["avg_time"] = sum(p["time"] for p in successful) / len(successful)
        results["avg_tps"] = sum(p["tokens_per_second"] for p in successful) / len(successful)
    else:
        results["avg_time"] = 0
        results["avg_tps"] = 0

    return results


def print_results(all_results: list):
    """Print benchmark comparison table"""
    print("\n" + "=" * 70)
    print("BENCHMARK RESULTS")
    print("=" * 70)

    # Header
    print(f"{'Provider':<20} {'Avg Time':<12} {'Avg TPS':<12} {'Success':<10}")
    print("-" * 70)

    for result in all_results:
        if "error" in result:
            print(f"{result['provider']:<20} {'ERROR':<12} {'-':<12} {result['error']}")
        else:
            success_count = sum(1 for p in result["prompts"] if p.get("success"))
            total = len(result["prompts"])
            print(
                f"{result['provider']:<20} "
                f"{result['avg_time']:.2f}s{'':<6} "
                f"{result['avg_tps']:.1f}{'':<6} "
                f"{success_count}/{total}"
            )

    # Detailed results
    print("\n" + "-" * 70)
    print("DETAILED RESULTS BY PROMPT")
    print("-" * 70)

    for prompt in BENCHMARK_PROMPTS:
        print(f"\n{prompt['name'].upper()} prompt:")
        for result in all_results:
            if "error" in result:
                continue
            prompt_result = next(
                (p for p in result["prompts"] if p["name"] == prompt["name"]),
                None
            )
            if prompt_result and prompt_result.get("success"):
                print(
                    f"  {result['provider']:<18}: "
                    f"{prompt_result['time']:.2f}s, "
                    f"{prompt_result['tokens']} tokens, "
                    f"{prompt_result['tokens_per_second']:.1f} tok/s"
                )
            elif prompt_result:
                print(f"  {result['provider']:<18}: ERROR - {prompt_result.get('error', 'unknown')}")


async def main():
    """Run benchmark across available providers"""
    print("=" * 70)
    print("KOSMOS MODEL BENCHMARK")
    print("=" * 70)

    # Detect available providers
    from provider_detector import detect_all

    status = detect_all()

    providers_to_test = []

    if status["ollama"]:
        models = status["ollama_models"]
        if any("qwen" in m.lower() for m in models):
            providers_to_test.append("local-fast")
        if any("deepseek" in m.lower() for m in models):
            providers_to_test.append("local-reasoning")

    if status["anthropic"]:
        providers_to_test.append("anthropic")

    if status["openai"]:
        providers_to_test.append("openai")

    if not providers_to_test:
        print("\n[ERROR] No providers available for benchmarking")
        print("Please ensure Ollama is running or API keys are set.")
        return False

    print(f"\nProviders to benchmark: {', '.join(providers_to_test)}")
    print(f"Prompts: {len(BENCHMARK_PROMPTS)}")
    print()

    # Run benchmarks
    all_results = []
    for provider in providers_to_test:
        print(f"Benchmarking {provider}...")
        result = await benchmark_provider(provider, BENCHMARK_PROMPTS)
        all_results.append(result)

    # Print results
    print_results(all_results)

    return True


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
