"""Kosmos E2E Testing Library

Automation library for end-to-end testing of the Kosmos project.
"""

from .provider_detector import (
    check_ollama,
    list_ollama_models,
    check_docker,
    check_sandbox,
    detect_all,
    recommend_provider,
    recommend_test_tier,
)
from .config_manager import (
    load_config,
    switch_provider,
    get_current_provider,
    validate_config,
)
from .test_runner import (
    run_tests,
    run_single_test,
)
from .report_generator import (
    generate_report,
    print_summary,
)

__all__ = [
    # Provider detection
    "check_ollama",
    "list_ollama_models",
    "check_docker",
    "check_sandbox",
    "detect_all",
    "recommend_provider",
    "recommend_test_tier",
    # Config management
    "load_config",
    "switch_provider",
    "get_current_provider",
    "validate_config",
    # Test runner
    "run_tests",
    "run_single_test",
    # Reporting
    "generate_report",
    "print_summary",
]
