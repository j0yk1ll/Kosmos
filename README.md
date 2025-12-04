# Kosmos

## Getting Started

### Requirements

- Python 3.11+
- Docker (for sandboxed code execution)

### Installation

```bash
git clone https://github.com/jimmc414/Kosmos.git
cd Kosmos
make setup_env
cp .env.example .env
```

### Verify Installation



### Run Research Workflow

```python
import asyncio
from kosmos.workflow.research_loop import ResearchWorkflow

async def run():
    workflow = ResearchWorkflow(
        research_objective="Your research question here",
        artifacts_dir="./artifacts"
    )
    result = await workflow.run(num_cycles=5, tasks_per_cycle=10)
    report = await workflow.generate_report()
    print(report)

asyncio.run(run())
```

## Configuration

All configuration via environment variables. See `.env.example` for full list.

### LLM Providers

```bash
# Anthropic (default)
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-...

# OpenAI and OpenAI-compatible providers
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-5-nano

# LiteLLM (supports 100+ providers including local models)
LLM_PROVIDER=litellm
LITELLM_MODEL=ollama/llama3.1:8b
LITELLM_API_BASE=http://localhost:11434
LITELLM_TIMEOUT=300

# DeepSeek via LiteLLM
LLM_PROVIDER=litellm
LITELLM_MODEL=deepseek/deepseek-chat
LITELLM_API_KEY=sk-...
```

### Debug Mode

```bash
# Enable debug mode
DEBUG_MODE=true

# Debug verbosity level (0-3)
# 0=off, 1=critical path, 2=full trace, 3=data dumps
DEBUG_LEVEL=2

# Log LLM request/response summaries
LOG_LLM_CALLS=true

# Log inter-agent message routing
LOG_AGENT_MESSAGES=true

# Log workflow state transitions with timing
LOG_WORKFLOW_TRANSITIONS=true

# Enable real-time stage tracking (outputs to logs/stages.jsonl)
STAGE_TRACKING_ENABLED=true
STAGE_TRACKING_FILE=logs/stages.jsonl
```

### Optional Services

```bash
# Neo4j (optional, for knowledge graph features)
NEO4J_URI=bolt://localhost:7687
NEO4J_PASSWORD=your-password

# Redis (optional, for distributed caching)
REDIS_URL=redis://localhost:6379
```

## Architecture

```
kosmos/
├── compression/      # Context compression (20:1 ratio)
├── world_model/      # State manager (JSON artifacts + optional graph)
├── orchestration/    # Task generation (plan creator/reviewer)
├── agents/           # Agent integration (skill loader)
├── execution/        # Sandboxed execution (Docker + Jupyter)
├── validation/       # Discovery validation (ScholarEval)
├── workflow/         # Integration layer combining all components
├── core/             # LLM clients, configuration, stage_tracker
│   ├── providers/    # Anthropic, OpenAI, LiteLLM providers with instrumentation
│   └── stage_tracker.py  # Real-time observability for multi-step processes
├── literature/       # Literature search (arXiv, PubMed, Semantic Scholar)
├── knowledge/        # Vector store, embeddings
├── monitoring/       # Metrics, alerts, cost tracking
└── cli/              # Command-line interface with debug options
```

### CLI Usage

```bash
# Run research with default settings
kosmos run --objective "Your research question"

# Enable trace logging (maximum verbosity)
kosmos run --trace --objective "Your research question"

# Set specific debug level (0-3)
kosmos run --debug-level 2 --objective "Your research question"

# Debug specific modules only
kosmos run --debug --debug-modules "research_director,workflow" --objective "Your research question"

# Show system information
kosmos info

# Run diagnostics
kosmos doctor

# Show version
kosmos version
```

## Documentation

- [OPEN_QUESTIONS.md](OPEN_QUESTIONS.md) - Original gap analysis
- [OPENQUESTIONS_SOLUTION.md](OPENQUESTIONS_SOLUTION.md) - How gaps were addressed
- [IMPLEMENTATION_REPORT.md](IMPLEMENTATION_REPORT.md) - Architecture decisions
- [PRODUCTION_READINESS_REPORT.md](PRODUCTION_READINESS_REPORT.md) - Current status
- [TESTS_STATUS.md](TESTS_STATUS.md) - Test coverage
- [MODEL_COMPARISON_REPORT.md](MODEL_COMPARISON_REPORT.md) - Multi-model performance comparison
- [GETTING_STARTED.md](GETTING_STARTED.md) - Usage examples

## Based On

- **Paper**: [Kosmos: An AI Scientist for Autonomous Discovery](https://arxiv.org/abs/2511.02824) (Lu et al., 2024)
- **K-Dense ecosystem**: Pattern repositories for AI agent systems
-- Analysis examples and figure-generation patterns (external repository no longer included)

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

Areas where contributions would be useful:
- Docker sandbox testing and hardening
- Integration test updates
- R language support via rpy2
- Additional scientific domain skills
- Performance benchmarking

## License

MIT License - see [LICENSE](LICENSE).

---

Version: 0.2.0-alpha
Gap Implementation: 6/6 complete
Test Coverage: 339 unit tests + 43 integration tests passing
Features: Debug mode, stage tracking, multi-provider support, model comparison
Last Updated: 2025-11-29
