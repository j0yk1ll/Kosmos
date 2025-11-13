# Kosmos Developer Guide

**Version:** 0.10.0
**Last Updated:** 2025-01-15

This guide helps developers extend and customize Kosmos for their specific needs.

## Table of Contents

1. [Development Setup](#development-setup)
2. [Code Structure](#code-structure)
3. [Creating Custom Agents](#creating-custom-agents)
4. [Adding New Domains](#adding-new-domains)
5. [Custom Experiment Types](#custom-experiment-types)
6. [Extending the CLI](#extending-the-cli)
7. [Testing Guidelines](#testing-guidelines)
8. [Contributing](#contributing)

---

## Development Setup

### Clone and Install

```bash
# Clone repository
git clone https://github.com/your-org/kosmos-ai-scientist.git
cd kosmos-ai-scientist

# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install in development mode with dev dependencies
pip install -e ".[dev]"
```

### Dev Dependencies

The `[dev]` extra includes:

- **pytest**: Testing framework
- **pytest-cov**: Coverage reporting
- **black**: Code formatting
- **ruff**: Linting
- **mypy**: Type checking
- **pre-commit**: Git hooks
- **sphinx**: Documentation

### Pre-commit Hooks

Set up automatic formatting and linting:

```bash
# Install pre-commit
pip install pre-commit

# Install hooks
pre-commit install

# Run manually
pre-commit run --all-files
```

### Environment Setup

Create `.env` for development:

```bash
# Copy example
cp .env.example .env

# Edit for development
nano .env
```

**Recommended dev settings:**

```env
# Use Claude CLI for development (no API costs)
ANTHROPIC_API_KEY=99999999999999999999999999999999999999999999999999

# Enable debug logging
LOG_LEVEL=DEBUG

# Use local database
DATABASE_URL=sqlite:///dev_kosmos.db

# Enable all safety checks
ENABLE_SAFETY_CHECKS=true
```

### Running Tests

```bash
# All tests
pytest

# With coverage
pytest --cov=kosmos --cov-report=html

# Specific test file
pytest tests/unit/agents/test_hypothesis_generator.py

# Specific test function
pytest tests/unit/agents/test_hypothesis_generator.py::test_generate_hypothesis

# Fast tests only (skip slow integration tests)
pytest -m "not slow"

# Show print statements
pytest -s

# Verbose
pytest -v
```

### Code Quality

```bash
# Format code
black kosmos/ tests/

# Lint
ruff check kosmos/ tests/

# Fix auto-fixable linting issues
ruff check --fix kosmos/ tests/

# Type check
mypy kosmos/
```

---

## Code Structure

### Package Organization

```
kosmos/
├── __init__.py              # Package exports
├── core/                    # Core infrastructure
│   ├── llm.py               # LLM client (Claude integration)
│   ├── cache.py             # Base caching functionality
│   ├── cache_manager.py     # Cache orchestration
│   ├── claude_cache.py      # Claude-specific caching
│   ├── experiment_cache.py  # Experiment result caching
│   ├── logging.py           # Structured logging
│   ├── metrics.py           # API usage tracking
│   ├── workflow.py          # State machine
│   ├── feedback.py          # Feedback loops
│   ├── memory.py            # Working memory
│   ├── convergence.py       # Convergence detection
│   └── domain_router.py     # Domain routing
├── agents/                  # Agent implementations
│   ├── base.py              # Abstract base agent
│   ├── registry.py          # Agent discovery
│   ├── research_director.py # Main orchestrator
│   ├── hypothesis_generator.py
│   ├── experiment_designer.py
│   ├── data_analyst.py
│   └── literature_analyzer.py
├── domains/                 # Domain-specific code
│   ├── biology/
│   │   ├── apis.py          # External API clients
│   │   ├── tools.py         # Analysis tools
│   │   └── prompts.py       # Prompt templates
│   ├── neuroscience/
│   ├── materials/
│   ├── physics/
│   ├── chemistry/
│   └── general/
├── execution/               # Experiment execution
│   ├── code_generator.py    # Code generation
│   ├── executor.py          # Sandboxed execution
│   ├── analyzer.py          # Result analysis
│   └── statistics.py        # Statistical tests
├── hypothesis/              # Hypothesis management
│   ├── generator.py         # Generation logic
│   ├── novelty_checker.py   # Novelty assessment
│   ├── prioritizer.py       # Prioritization
│   ├── testability.py       # Testability evaluation
│   └── models.py            # Data models
├── knowledge/               # Knowledge integration
│   ├── graph.py             # Neo4j integration
│   ├── embeddings.py        # Vector embeddings
│   └── domain_kb.py         # Domain knowledge bases
├── safety/                  # Safety and validation
│   ├── validator.py         # Code validation
│   ├── rules.py             # Safety rules
│   └── limits.py            # Resource limits
├── cli/                     # Command-line interface
│   ├── main.py              # Main CLI app
│   ├── interactive.py       # Interactive mode
│   ├── utils.py             # Shared utilities
│   ├── themes.py            # Rich theme config
│   ├── commands/            # CLI commands
│   │   ├── run.py
│   │   ├── status.py
│   │   ├── history.py
│   │   ├── cache.py
│   │   └── config.py
│   └── views/               # Result viewers
│       └── results_viewer.py
├── db/                      # Database layer
│   ├── models.py            # SQLAlchemy models
│   ├── operations.py        # Database operations
│   └── session.py           # Session management
└── config.py                # Configuration management

tests/
├── unit/                    # Unit tests
│   ├── agents/
│   ├── core/
│   ├── domains/
│   └── ...
├── integration/             # Integration tests
│   ├── test_cli.py
│   ├── test_research_flow.py
│   └── ...
└── fixtures/                # Test fixtures
    └── ...
```

### Naming Conventions

- **Files**: `snake_case.py`
- **Classes**: `PascalCase`
- **Functions/Methods**: `snake_case()`
- **Constants**: `UPPER_SNAKE_CASE`
- **Private**: `_leading_underscore()`
- **Tests**: `test_*` files, `test_*` functions

### Import Organization

```python
# Standard library
import sys
from pathlib import Path
from typing import Dict, Any, Optional

# Third-party
import numpy as np
from pydantic import BaseModel

# Local application
from kosmos.core.llm import get_claude_client
from kosmos.agents.base import BaseAgent
```

---

## Creating Custom Agents

### Agent Base Class

All agents inherit from `BaseAgent`:

```python
# kosmos/agents/base.py
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

class BaseAgent(ABC):
    """Abstract base class for all agents."""

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.name = self.__class__.__name__
        self.llm_client = get_claude_client()

    @abstractmethod
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the agent's main logic."""
        raise NotImplementedError

    def save_state(self, run_id: str) -> None:
        """Save agent state."""
        pass

    @classmethod
    def load_state(cls, run_id: str) -> 'BaseAgent':
        """Load agent from saved state."""
        pass
```

### Example Custom Agent

Create a new agent in `kosmos/agents/my_custom_agent.py`:

```python
"""
Custom agent for specialized research task.
"""
from typing import Dict, Any
from kosmos.agents.base import BaseAgent
from kosmos.core.llm import get_claude_client

class MyCustomAgent(BaseAgent):
    """
    Custom agent for [specific purpose].

    This agent performs [description of what it does].
    """

    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        self.name = "MyCustomAgent"
        # Initialize any custom attributes
        self.custom_param = config.get("custom_param", "default")

    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute custom agent logic.

        Args:
            input_data: Input data dictionary with keys:
                - "query": Research query
                - "context": Additional context
                - "domain": Scientific domain

        Returns:
            Dictionary with results:
                - "output": Main output
                - "metadata": Additional metadata
        """
        query = input_data.get("query", "")
        context = input_data.get("context", {})
        domain = input_data.get("domain", "general")

        # Your custom logic here
        result = self._process_query(query, context, domain)

        return {
            "output": result,
            "metadata": {
                "agent": self.name,
                "domain": domain
            }
        }

    def _process_query(
        self, query: str, context: Dict, domain: str
    ) -> Any:
        """
        Internal processing method.

        Args:
            query: Research query
            context: Context dictionary
            domain: Scientific domain

        Returns:
            Processed result
        """
        # Example: Use Claude for processing
        prompt = f"""
        Domain: {domain}
        Query: {query}
        Context: {context}

        [Your prompt here]
        """

        response = self.llm_client.chat(
            message=prompt,
            max_tokens=2048,
            temperature=0.7
        )

        # Parse and return
        return self._parse_response(response)

    def _parse_response(self, response: str) -> Any:
        """Parse Claude's response."""
        # Your parsing logic
        return response

    def save_state(self, run_id: str) -> None:
        """Save agent state to file."""
        state = {
            "name": self.name,
            "config": self.config,
            "custom_param": self.custom_param
        }
        # Save to file or database
        # ...

    @classmethod
    def load_state(cls, run_id: str) -> 'MyCustomAgent':
        """Load agent from saved state."""
        # Load from file or database
        # ...
        return cls(config=loaded_config)
```

### Register Custom Agent

Add to agent registry:

```python
# kosmos/agents/__init__.py
from kosmos.agents.my_custom_agent import MyCustomAgent

# Register agent
from kosmos.agents.registry import AgentRegistry
registry = AgentRegistry()
registry.register("my_custom", MyCustomAgent)
```

### Use Custom Agent

```python
from kosmos.agents.registry import AgentRegistry

# Get agent
registry = AgentRegistry()
agent = registry.get("my_custom", config={"custom_param": "value"})

# Execute
result = agent.execute({
    "query": "research question",
    "context": {},
    "domain": "biology"
})
```

---

## Adding New Domains

### Domain Structure

Create a new domain in `kosmos/domains/<domain_name>/`:

```
kosmos/domains/my_domain/
├── __init__.py       # Exports
├── apis.py           # API clients
├── tools.py          # Analysis tools
├── prompts.py        # Prompt templates
└── ontology.py       # Knowledge structures (optional)
```

### Example Domain: Astronomy

```python
# kosmos/domains/astronomy/__init__.py
"""
Astronomy domain for Kosmos.

Provides access to astronomical databases and analysis tools.
"""
from kosmos.domains.astronomy.apis import (
    SIMBADClient,
    NEDClient,
)
from kosmos.domains.astronomy.tools import (
    calculate_redshift,
    analyze_spectrum,
)

__all__ = [
    "SIMBADClient",
    "NEDClient",
    "calculate_redshift",
    "analyze_spectrum",
]
```

### API Client Example

```python
# kosmos/domains/astronomy/apis.py
"""
API clients for astronomy databases.
"""
import httpx
from typing import Dict, Any, Optional
from tenacity import retry, stop_after_attempt, wait_exponential

class SIMBADClient:
    """
    Client for SIMBAD astronomical database.

    SIMBAD provides data on astronomical objects outside the solar system.
    """

    def __init__(self, base_url: str = "http://simbad.u-strasbg.fr/simbad/sim-script"):
        self.base_url = base_url
        self.client = httpx.Client(timeout=30.0)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    def query_object(self, object_name: str) -> Dict[str, Any]:
        """
        Query SIMBAD for an astronomical object.

        Args:
            object_name: Name of the object (e.g., "M31", "NGC 1234")

        Returns:
            Dictionary with object data

        Raises:
            httpx.HTTPError: If request fails
        """
        # Format SIMBAD script
        script = f"""
        output console=off script=off
        format object "%IDLIST(1) | %COO(A D) | %OTYPE"
        query id {object_name}
        """

        response = self.client.post(
            self.base_url,
            data={"script": script}
        )
        response.raise_for_status()

        # Parse response
        data = self._parse_response(response.text)
        return data

    def _parse_response(self, text: str) -> Dict[str, Any]:
        """Parse SIMBAD response."""
        lines = text.strip().split("|")
        return {
            "name": lines[0].strip() if len(lines) > 0 else None,
            "coordinates": lines[1].strip() if len(lines) > 1 else None,
            "object_type": lines[2].strip() if len(lines) > 2 else None,
        }

    def close(self):
        """Close client."""
        self.client.close()
```

### Tools Example

```python
# kosmos/domains/astronomy/tools.py
"""
Analysis tools for astronomy research.
"""
import numpy as np
from typing import Dict, Any

def calculate_redshift(
    observed_wavelength: float,
    rest_wavelength: float
) -> float:
    """
    Calculate cosmological redshift.

    Args:
        observed_wavelength: Observed wavelength (Angstroms)
        rest_wavelength: Rest wavelength (Angstroms)

    Returns:
        Redshift value (z)
    """
    z = (observed_wavelength - rest_wavelength) / rest_wavelength
    return z

def analyze_spectrum(
    wavelengths: np.ndarray,
    flux: np.ndarray
) -> Dict[str, Any]:
    """
    Analyze astronomical spectrum.

    Args:
        wavelengths: Wavelength array (Angstroms)
        flux: Flux array

    Returns:
        Dictionary with spectral features
    """
    # Find emission/absorption lines
    # ... analysis logic ...

    return {
        "emission_lines": [],
        "absorption_lines": [],
        "continuum_level": np.median(flux),
    }
```

### Prompts Example

```python
# kosmos/domains/astronomy/prompts.py
"""
Prompt templates for astronomy research.
"""

HYPOTHESIS_GENERATION_TEMPLATE = """
You are an expert astronomer generating research hypotheses.

Research Question: {question}
Domain: Astronomy
Context: {context}

Available Data:
- SIMBAD database (object catalogs)
- NED database (extragalactic objects)
- Spectroscopic data
- Photometric surveys

Generate {num_hypotheses} testable hypotheses that:
1. Are scientifically plausible
2. Can be tested with available data
3. Address the research question
4. Are sufficiently novel

For each hypothesis provide:
- Clear claim statement
- Scientific rationale
- Predicted outcomes
- Testability assessment

Format as JSON array.
"""

EXPERIMENT_DESIGN_TEMPLATE = """
Design a computational experiment to test this hypothesis:

Hypothesis: {hypothesis}
Domain: Astronomy

Available Tools:
{tools}

Design an experiment that:
1. Tests the hypothesis
2. Uses appropriate statistical methods
3. Handles observational uncertainties
4. Produces interpretable results

Provide:
- Experiment type
- Data requirements
- Analysis procedure
- Expected outputs
"""
```

### Register Domain

```python
# kosmos/core/domain_router.py
SUPPORTED_DOMAINS = {
    "biology": "Biology and life sciences",
    "neuroscience": "Neuroscience and brain research",
    "materials": "Materials science",
    "physics": "Physics",
    "chemistry": "Chemistry",
    "astronomy": "Astronomy and astrophysics",  # Add new domain
    "general": "General or cross-domain"
}
```

---

## Custom Experiment Types

### Experiment Template

Create experiment template in `kosmos/execution/templates/`:

```python
# kosmos/execution/templates/my_analysis.py
"""
Template for custom analysis experiment.
"""

TEMPLATE = """
#!/usr/bin/env python3
# Generated by Kosmos - Custom Analysis Template

import numpy as np
import pandas as pd
from scipy import stats
import json

# Parameters (injected by Kosmos)
{parameters}

def main():
    # Load data
    data = load_data(data_path)

    # Perform analysis
    result = analyze_data(data, **params)

    # Save results
    save_results(result, output_path)

def load_data(path):
    # Load logic
    pass

def analyze_data(data, **kwargs):
    # Your analysis logic
    pass

def save_results(result, path):
    with open(path, 'w') as f:
        json.dump(result, f, indent=2)

if __name__ == "__main__":
    main()
"""

def generate_code(parameters: Dict[str, Any]) -> str:
    """Generate code from template."""
    params_str = "\\n".join(
        f"{k} = {repr(v)}" for k, v in parameters.items()
    )
    return TEMPLATE.format(parameters=params_str)
```

---

## Extending the CLI

### Add New Command

Create new command in `kosmos/cli/commands/mycommand.py`:

```python
"""
Custom CLI command.
"""
import typer
from kosmos.cli.utils import console, print_success

def my_command(
    arg: str = typer.Argument(..., help="Required argument"),
    option: bool = typer.Option(False, "--flag", "-f", help="Optional flag")
):
    """
    Description of custom command.
    """
    console.print(f"Processing {arg}...")

    # Command logic
    result = process(arg, option)

    print_success(f"Completed: {result}")
```

### Register Command

Add to main CLI app:

```python
# kosmos/cli/main.py
from kosmos.cli.commands import mycommand

app.command(name="mycommand")(mycommand.my_command)
```

---

## Testing Guidelines

### Test Structure

```python
# tests/unit/agents/test_my_custom_agent.py
"""
Tests for MyCustomAgent.
"""
import pytest
from unittest.mock import Mock, patch

from kosmos.agents.my_custom_agent import MyCustomAgent

@pytest.fixture
def agent():
    """Create agent instance."""
    return MyCustomAgent(config={"custom_param": "test"})

class TestMyCustomAgent:
    """Test suite for MyCustomAgent."""

    def test_init(self, agent):
        """Test initialization."""
        assert agent.name == "MyCustomAgent"
        assert agent.custom_param == "test"

    def test_execute(self, agent):
        """Test execute method."""
        input_data = {
            "query": "test query",
            "context": {},
            "domain": "biology"
        }

        result = agent.execute(input_data)

        assert "output" in result
        assert "metadata" in result
        assert result["metadata"]["agent"] == "MyCustomAgent"

    @patch("kosmos.agents.my_custom_agent.get_claude_client")
    def test_execute_with_mock_llm(self, mock_get_client, agent):
        """Test execute with mocked LLM."""
        # Mock LLM response
        mock_client = Mock()
        mock_client.chat.return_value = "mocked response"
        mock_get_client.return_value = mock_client

        result = agent.execute({"query": "test"})

        # Verify LLM was called
        mock_client.chat.assert_called_once()
```

### Integration Tests

```python
# tests/integration/test_custom_domain.py
"""
Integration tests for custom domain.
"""
import pytest
from kosmos.domains.astronomy import SIMBADClient

@pytest.mark.integration
class TestSIMBADIntegration:
    """Integration tests for SIMBAD API."""

    def test_query_real_object(self):
        """Test querying real astronomical object."""
        client = SIMBADClient()

        result = client.query_object("M31")

        assert result is not None
        assert "name" in result
        assert "Andromeda" in result["name"] or "M31" in result["name"]

        client.close()
```

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed contribution guidelines.

**Quick Checklist:**

- [ ] Code follows style guide (black + ruff)
- [ ] Type hints added
- [ ] Docstrings complete
- [ ] Tests written and passing
- [ ] Documentation updated
- [ ] CHANGELOG.md updated
- [ ] Pre-commit hooks pass

---

*Happy developing! 🚀*
