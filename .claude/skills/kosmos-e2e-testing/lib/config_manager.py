"""Configuration Manager for Kosmos E2E Testing

Handles dynamic configuration loading and provider switching.
"""

import os
from pathlib import Path
from typing import Optional


# Skill directory path
SKILL_DIR = Path(__file__).parent.parent
CONFIGS_DIR = SKILL_DIR / "configs"


def load_config(provider: str) -> dict:
    """Load environment configuration for provider

    Args:
        provider: Provider name (local-fast, local-reasoning, anthropic, openai)

    Returns:
        Dictionary of environment variables
    """
    config_file = CONFIGS_DIR / f"{provider}.env"

    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")

    config = {}
    with open(config_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                # Handle export statements
                if line.startswith("export "):
                    line = line[7:]
                key, value = line.split("=", 1)
                # Remove quotes if present
                value = value.strip("'\"")
                # Expand environment variables
                if value.startswith("${") and ":-" in value:
                    # Handle ${VAR:-default} syntax
                    var_part = value[2:-1]
                    var_name, default = var_part.split(":-", 1)
                    value = os.environ.get(var_name, default)
                config[key] = value

    return config


def switch_provider(provider: str) -> None:
    """Switch active provider by setting environment variables

    Args:
        provider: Provider name to switch to
    """
    config = load_config(provider)

    for key, value in config.items():
        os.environ[key] = value

    print(f"Switched to provider: {provider}")


def get_current_provider() -> str:
    """Get currently configured provider

    Returns:
        Provider name based on environment variables
    """
    llm_provider = os.environ.get("LLM_PROVIDER", "").lower()
    base_url = os.environ.get("OPENAI_BASE_URL", "")
    model = os.environ.get("OPENAI_MODEL", "")

    if llm_provider == "anthropic":
        return "anthropic"

    if "localhost:11434" in base_url:
        if "deepseek" in model.lower():
            return "local-reasoning"
        return "local-fast"

    if llm_provider == "openai":
        return "openai"

    return "unknown"


def validate_config(provider: str) -> tuple[bool, list[str]]:
    """Validate provider configuration

    Args:
        provider: Provider name to validate

    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []

    try:
        config = load_config(provider)
    except FileNotFoundError as e:
        return False, [str(e)]

    # Provider-specific validation
    if provider in ("local-fast", "local-reasoning"):
        if config.get("OPENAI_BASE_URL") != "http://localhost:11434/v1":
            errors.append("OPENAI_BASE_URL should be http://localhost:11434/v1 for local models")

        # Check if Ollama is running
        from .provider_detector import check_ollama, list_ollama_models
        if not check_ollama():
            errors.append("Ollama is not running on localhost:11434")
        else:
            models = list_ollama_models()
            expected_model = config.get("OPENAI_MODEL", "")
            if expected_model and expected_model not in models:
                errors.append(f"Model '{expected_model}' not installed in Ollama")

    elif provider == "anthropic":
        api_key = os.environ.get("ANTHROPIC_API_KEY", config.get("ANTHROPIC_API_KEY", ""))
        if not api_key or api_key == "your-api-key-here":
            errors.append("ANTHROPIC_API_KEY not set")

    elif provider == "openai":
        api_key = os.environ.get("OPENAI_API_KEY", config.get("OPENAI_API_KEY", ""))
        if not api_key or api_key == "your-api-key-here":
            errors.append("OPENAI_API_KEY not set")

    return len(errors) == 0, errors


def list_providers() -> list[str]:
    """List all available provider configurations

    Returns:
        List of provider names
    """
    providers = []
    for config_file in CONFIGS_DIR.glob("*.env"):
        providers.append(config_file.stem)
    return sorted(providers)


def print_config(provider: Optional[str] = None) -> None:
    """Print configuration details

    Args:
        provider: Specific provider to show, or None for current
    """
    if provider is None:
        provider = get_current_provider()

    print(f"Provider: {provider}")
    print("-" * 40)

    try:
        config = load_config(provider)
        for key, value in config.items():
            # Mask sensitive values
            if "KEY" in key and value and value != "ollama":
                display_value = value[:8] + "..." if len(value) > 8 else "***"
            else:
                display_value = value
            print(f"  {key}={display_value}")
    except FileNotFoundError:
        # Show current environment
        relevant_vars = [
            "LLM_PROVIDER", "OPENAI_API_KEY", "OPENAI_BASE_URL",
            "OPENAI_MODEL", "ANTHROPIC_API_KEY", "ANTHROPIC_MODEL"
        ]
        for var in relevant_vars:
            value = os.environ.get(var, "")
            if "KEY" in var and value:
                display_value = value[:8] + "..." if len(value) > 8 else "***"
            else:
                display_value = value or "(not set)"
            print(f"  {var}={display_value}")


if __name__ == "__main__":
    print("Available providers:", list_providers())
    print()
    print("Current configuration:")
    print_config()
