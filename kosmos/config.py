"""
Configuration management using Pydantic for validation.

Loads configuration from environment variables and provides validated settings
for all Kosmos components.
"""

import logging
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

from kosmos.utils.compat import model_to_dict


# Claude 4.5 models (November 2025)
_DEFAULT_CLAUDE_SONNET_MODEL = "claude-sonnet-4-5"
_DEFAULT_CLAUDE_HAIKU_MODEL = "claude-haiku-4-5"


def parse_comma_separated(v):
    """Parse comma-separated string into list for Pydantic V2 compatibility."""
    if v is None or v == "":
        return None  # Let field default handle it
    if isinstance(v, str):
        return [x.strip() for x in v.split(",") if x.strip()]
    return v


class LLMConfig(BaseSettings):
    """
    Unified LLM configuration for DSPy.

    DSPy supports all major LLM providers with a unified interface:
    - Anthropic: model="anthropic/claude-sonnet-4-5", api_key=ANTHROPIC_API_KEY
    - OpenAI: model="openai/gpt-4", api_key=OPENAI_API_KEY
    - Ollama (local): model="ollama_chat/llama3.1:8b", api_base="http://localhost:11434"
    - DeepSeek: model="deepseek/deepseek-chat", api_key=DEEPSEEK_API_KEY
    - And 100+ more providers via LiteLLM

    Example .env configurations:
    ```
    # Anthropic Claude
    LLM_MODEL=anthropic/claude-sonnet-4-5
    LLM_API_KEY=sk-ant-...

    # OpenAI
    LLM_MODEL=openai/gpt-4
    LLM_API_KEY=sk-...

    # Local Ollama
    LLM_MODEL=ollama_chat/llama3.1:8b
    LLM_API_BASE=http://localhost:11434
    ```
    """

    model: str = Field(
        default="anthropic/claude-sonnet-4-5",
        description="Model identifier in DSPy/LiteLLM format (provider/model-name)",
        alias="LLM_MODEL",
    )
    api_key: str | None = Field(
        default=None,
        description="API key for the provider (not required for local models)",
        alias="LLM_API_KEY",
    )
    api_base: str | None = Field(
        default=None,
        description="Custom base URL for API (e.g., http://localhost:11434 for Ollama)",
        alias="LLM_API_BASE",
    )
    max_tokens: int = Field(
        default=4096,
        ge=1,
        le=200000,
        description="Maximum tokens per request",
        alias="LLM_MAX_TOKENS",
    )
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Sampling temperature",
        alias="LLM_TEMPERATURE",
    )
    timeout: int = Field(
        default=120,
        ge=1,
        le=600,
        description="Request timeout in seconds",
        alias="LLM_TIMEOUT",
    )
    cache_seed: int | None = Field(
        default=None,
        description="Cache seed for deterministic caching (DSPy feature)",
        alias="LLM_CACHE_SEED",
    )

    def to_dspy_config(self) -> dict[str, Any]:
        """
        Convert to DSPy LM configuration dictionary.

        Returns:
            dict: Configuration dict for dspy.LM(**config)
        """
        config = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }

        if self.api_key:
            config["api_key"] = self.api_key
        if self.api_base:
            config["api_base"] = self.api_base
        if self.cache_seed is not None:
            config["cache_seed"] = self.cache_seed

        return config

    @property
    def provider(self) -> str:
        """Extract provider from model string (e.g., 'anthropic' from 'anthropic/claude-sonnet-4-5')."""
        if "/" in self.model:
            return self.model.split("/")[0]
        return "unknown"

    model_config = SettingsConfigDict(populate_by_name=True)


# Backward compatibility aliases (deprecated)
ClaudeConfig = LLMConfig
AnthropicConfig = LLMConfig
OpenAIConfig = LLMConfig
LiteLLMConfig = LLMConfig


class ResearchConfig(BaseSettings):
    """Research workflow configuration."""

    max_iterations: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum research iterations",
        alias="MAX_RESEARCH_ITERATIONS",
    )
    enabled_domains: Annotated[list[str], NoDecode] = Field(
        default=["biology", "physics", "chemistry", "neuroscience"],
        description="Enabled scientific domains",
        alias="ENABLED_DOMAINS",
    )
    enabled_experiment_types: Annotated[list[str], NoDecode] = Field(
        default=["computational", "data_analysis", "literature_synthesis"],
        description="Enabled experiment types",
        alias="ENABLED_EXPERIMENT_TYPES",
    )

    @field_validator("enabled_domains", "enabled_experiment_types", mode="before")
    @classmethod
    def parse_comma_separated_lists(cls, v):
        """Parse comma-separated strings into lists."""
        return parse_comma_separated(v)

    min_novelty_score: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Minimum novelty score for hypotheses",
        alias="MIN_NOVELTY_SCORE",
    )
    enable_autonomous_iteration: bool = Field(
        default=True,
        description="Enable autonomous research iteration",
        alias="ENABLE_AUTONOMOUS_ITERATION",
    )
    budget_usd: float = Field(
        default=10.0,
        ge=0.0,
        description="Research budget in USD for API costs",
        alias="RESEARCH_BUDGET_USD",
    )

    model_config = SettingsConfigDict(populate_by_name=True)


class DatabaseConfig(BaseSettings):
    """Database configuration."""

    url: str = Field(
        default="sqlite:///kosmos.db", description="Database URL", alias="DATABASE_URL"
    )
    echo: bool = Field(default=False, description="Enable SQL echo logging", alias="DATABASE_ECHO")

    @property
    def is_sqlite(self) -> bool:
        """Check if using SQLite."""
        return self.url.startswith("sqlite")

    @property
    def normalized_url(self) -> str:
        """
        Get database URL with absolute path for SQLite.

        For SQLite databases with relative paths, converts to absolute path
        based on project root to ensure consistent database location regardless
        of current working directory.

        Returns:
            str: Normalized database URL
        """
        if not self.is_sqlite:
            return self.url

        # Extract path from SQLite URL
        # Format: sqlite:///path/to/db.db or sqlite:////absolute/path
        if self.url.startswith("sqlite:///"):
            db_path_str = self.url[10:]  # Remove "sqlite:///"

            # Check if already absolute (starts with / on Unix or drive letter on Windows)
            if db_path_str.startswith("/") or (len(db_path_str) > 1 and db_path_str[1] == ":"):
                return self.url

            # Convert relative path to absolute based on project root
            project_root = Path(__file__).parent.parent
            db_path = project_root / db_path_str
            abs_path = db_path.resolve()

            return f"sqlite:///{abs_path}"

        return self.url

    model_config = SettingsConfigDict(populate_by_name=True)


class RedisConfig(BaseSettings):
    """Redis cache configuration."""

    url: str = Field(
        default="redis://localhost:6379/0", description="Redis connection URL", alias="REDIS_URL"
    )
    enabled: bool = Field(default=False, description="Enable Redis caching", alias="REDIS_ENABLED")
    max_connections: int = Field(
        default=50,
        ge=1,
        le=1000,
        description="Maximum connection pool size",
        alias="REDIS_MAX_CONNECTIONS",
    )
    socket_timeout: int = Field(
        default=5,
        ge=1,
        le=60,
        description="Socket timeout in seconds",
        alias="REDIS_SOCKET_TIMEOUT",
    )
    socket_connect_timeout: int = Field(
        default=5,
        ge=1,
        le=60,
        description="Socket connect timeout in seconds",
        alias="REDIS_SOCKET_CONNECT_TIMEOUT",
    )
    retry_on_timeout: bool = Field(
        default=True, description="Retry on timeout", alias="REDIS_RETRY_ON_TIMEOUT"
    )
    decode_responses: bool = Field(
        default=True,
        description="Decode responses as UTF-8 strings",
        alias="REDIS_DECODE_RESPONSES",
    )
    default_ttl_seconds: int = Field(
        default=3600,
        ge=60,
        le=86400,
        description="Default cache TTL in seconds (1 minute to 24 hours)",
        alias="REDIS_DEFAULT_TTL_SECONDS",
    )

    @property
    def is_available(self) -> bool:
        """Check if Redis is enabled and configured."""
        return self.enabled and self.url is not None

    model_config = SettingsConfigDict(populate_by_name=True)


class LoggingConfig(BaseSettings):
    """Logging configuration."""

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO", description="Log level", alias="LOG_LEVEL"
    )
    format: Literal["json", "text"] = Field(
        default="json", description="Log format", alias="LOG_FORMAT"
    )
    file: str | None = Field(
        default="logs/kosmos.log",
        description="Log file path (None for stdout only)",
        alias="LOG_FILE",
    )
    debug_mode: bool = Field(
        default=False, description="Enable debug mode with verbose output", alias="DEBUG_MODE"
    )

    # Enhanced debug configuration
    debug_level: Literal[0, 1, 2, 3] = Field(
        default=0,
        description="Debug verbosity: 0=off, 1=critical path, 2=full trace, 3=data dumps",
        alias="DEBUG_LEVEL",
    )

    debug_modules: Annotated[list[str] | None, NoDecode] = Field(
        default=None,
        description="Modules to debug (None=all when debug_mode=True)",
        alias="DEBUG_MODULES",
    )

    @field_validator("debug_level", mode="before")
    @classmethod
    def parse_debug_level(cls, v):
        """Convert string debug level to int for Literal compatibility."""
        if isinstance(v, str):
            return int(v)
        return v

    @field_validator("debug_modules", mode="before")
    @classmethod
    def parse_debug_modules(cls, v):
        """Parse comma-separated strings into list."""
        return parse_comma_separated(v)

    log_llm_calls: bool = Field(
        default=False, description="Log LLM request/response summaries", alias="LOG_LLM_CALLS"
    )

    log_agent_messages: bool = Field(
        default=False, description="Log inter-agent message routing", alias="LOG_AGENT_MESSAGES"
    )

    log_workflow_transitions: bool = Field(
        default=False,
        description="Log state machine transitions with timing",
        alias="LOG_WORKFLOW_TRANSITIONS",
    )

    stage_tracking_enabled: bool = Field(
        default=False,
        description="Enable real-time stage tracking output",
        alias="STAGE_TRACKING_ENABLED",
    )

    stage_tracking_file: str = Field(
        default="logs/stages.jsonl",
        description="Stage tracking output file",
        alias="STAGE_TRACKING_FILE",
    )

    model_config = SettingsConfigDict(populate_by_name=True)


class LiteratureConfig(BaseSettings):
    """Literature API configuration."""

    semantic_scholar_api_key: str | None = Field(
        default=None,
        description="Semantic Scholar API key (optional, increases rate limits)",
        alias="SEMANTIC_SCHOLAR_API_KEY",
    )
    pubmed_api_key: str | None = Field(
        default=None,
        description="PubMed API key (optional, increases rate limits)",
        alias="PUBMED_API_KEY",
    )
    pubmed_email: str | None = Field(
        default=None, description="Email for PubMed E-utilities (recommended)", alias="PUBMED_EMAIL"
    )
    cache_ttl_hours: int = Field(
        default=48,
        ge=1,
        le=168,
        description="Literature API cache TTL in hours (24-168)",
        alias="LITERATURE_CACHE_TTL_HOURS",
    )
    max_results_per_query: int = Field(
        default=100,
        ge=1,
        le=1000,
        description="Maximum results per literature search query",
        alias="MAX_RESULTS_PER_QUERY",
    )
    pdf_download_timeout: int = Field(
        default=30,
        ge=5,
        le=120,
        description="PDF download timeout in seconds",
        alias="PDF_DOWNLOAD_TIMEOUT",
    )
    search_timeout: int = Field(
        default=60,
        ge=10,
        le=300,
        description="Literature search timeout in seconds (all sources combined)",
        alias="LITERATURE_SEARCH_TIMEOUT",
    )
    api_timeout: int = Field(
        default=30,
        ge=5,
        le=120,
        description="Individual API call timeout in seconds (PubMed, etc.)",
        alias="LITERATURE_API_TIMEOUT",
    )

    model_config = SettingsConfigDict(populate_by_name=True)


class VectorDBConfig(BaseSettings):
    """Vector database configuration."""

    type: Literal["chromadb", "pinecone", "weaviate"] = Field(
        default="chromadb", description="Vector database type", alias="VECTOR_DB_TYPE"
    )
    chroma_persist_directory: str = Field(
        default=".chroma_db",
        description="ChromaDB persistence directory",
        alias="CHROMA_PERSIST_DIRECTORY",
    )
    pinecone_api_key: str | None = Field(
        default=None, description="Pinecone API key", alias="PINECONE_API_KEY"
    )
    pinecone_environment: str | None = Field(
        default=None, description="Pinecone environment", alias="PINECONE_ENVIRONMENT"
    )
    pinecone_index_name: str | None = Field(
        default="kosmos", description="Pinecone index name", alias="PINECONE_INDEX_NAME"
    )

    @model_validator(mode="after")
    def validate_pinecone_config(self):
        """Validate Pinecone configuration if selected."""
        if self.type == "pinecone":
            if not self.pinecone_api_key:
                raise ValueError("PINECONE_API_KEY required when using Pinecone")
            if not self.pinecone_environment:
                raise ValueError("PINECONE_ENVIRONMENT required when using Pinecone")
        return self

    model_config = SettingsConfigDict(populate_by_name=True)


class Neo4jConfig(BaseSettings):
    """Neo4j knowledge graph configuration."""

    uri: str = Field(
        default="bolt://localhost:7687", description="Neo4j connection URI", alias="NEO4J_URI"
    )
    user: str = Field(default="neo4j", description="Neo4j username", alias="NEO4J_USER")
    password: str = Field(
        default="kosmos-password", description="Neo4j password", alias="NEO4J_PASSWORD"
    )
    database: str = Field(
        default="neo4j", description="Neo4j database name", alias="NEO4J_DATABASE"
    )
    max_connection_lifetime: int = Field(
        default=3600,
        ge=60,
        description="Max connection lifetime in seconds",
        alias="NEO4J_MAX_CONNECTION_LIFETIME",
    )
    max_connection_pool_size: int = Field(
        default=50,
        ge=1,
        description="Max connection pool size",
        alias="NEO4J_MAX_CONNECTION_POOL_SIZE",
    )

    model_config = SettingsConfigDict(populate_by_name=True)


class SafetyConfig(BaseSettings):
    """Safety and security configuration."""

    enable_safety_checks: bool = Field(
        default=True, description="Enable code safety checks", alias="ENABLE_SAFETY_CHECKS"
    )
    max_experiment_execution_time: int = Field(
        default=300,
        ge=1,
        description="Max execution time for experiments (seconds)",
        alias="MAX_EXPERIMENT_EXECUTION_TIME",
    )
    max_memory_mb: int = Field(
        default=2048, ge=128, description="Maximum memory usage (MB)", alias="MAX_MEMORY_MB"
    )
    max_cpu_cores: float | None = Field(
        default=None,
        ge=0.1,
        description="Maximum CPU cores to use (None = unlimited)",
        alias="MAX_CPU_CORES",
    )
    enable_sandboxing: bool = Field(
        default=True, description="Enable sandboxed code execution", alias="ENABLE_SANDBOXING"
    )
    require_human_approval: bool = Field(
        default=False,
        description="Require human approval for high-risk operations",
        alias="REQUIRE_HUMAN_APPROVAL",
    )

    # Ethical guidelines
    ethical_guidelines_path: str | None = Field(
        default=None,
        description="Path to ethical guidelines JSON file",
        alias="ETHICAL_GUIDELINES_PATH",
    )

    # Result verification
    enable_result_verification: bool = Field(
        default=True, description="Enable result verification", alias="ENABLE_RESULT_VERIFICATION"
    )
    outlier_threshold: float = Field(
        default=3.0,
        ge=1.0,
        description="Z-score threshold for outlier detection",
        alias="OUTLIER_THRESHOLD",
    )

    # Reproducibility
    default_random_seed: int = Field(
        default=42,
        description="Default random seed for reproducibility",
        alias="DEFAULT_RANDOM_SEED",
    )
    capture_environment: bool = Field(
        default=True, description="Capture environment snapshots", alias="CAPTURE_ENVIRONMENT"
    )

    # Human oversight
    approval_mode: str = Field(
        default="blocking",
        description="Approval workflow mode (blocking/queue/automatic/disabled)",
        alias="APPROVAL_MODE",
    )
    auto_approve_low_risk: bool = Field(
        default=True,
        description="Automatically approve low-risk operations",
        alias="AUTO_APPROVE_LOW_RISK",
    )

    # Notifications
    notification_channel: str = Field(
        default="both",
        description="Notification channel (console/log/both)",
        alias="NOTIFICATION_CHANNEL",
    )
    notification_min_level: str = Field(
        default="info",
        description="Minimum notification level (debug/info/warning/error/critical)",
        alias="NOTIFICATION_MIN_LEVEL",
    )
    use_rich_formatting: bool = Field(
        default=True,
        description="Use rich formatting for console notifications",
        alias="USE_RICH_FORMATTING",
    )

    # Incident logging
    incident_log_path: str = Field(
        default="safety_incidents.jsonl",
        description="Path to safety incident log file",
        alias="INCIDENT_LOG_PATH",
    )
    audit_log_path: str = Field(
        default="human_review_audit.jsonl",
        description="Path to human review audit log",
        alias="AUDIT_LOG_PATH",
    )

    model_config = SettingsConfigDict(populate_by_name=True)


class PerformanceConfig(BaseSettings):
    """Performance and caching configuration."""

    enable_result_caching: bool = Field(
        default=True, description="Enable result caching", alias="ENABLE_RESULT_CACHING"
    )
    cache_ttl: int = Field(
        default=3600, ge=0, description="Cache TTL in seconds", alias="CACHE_TTL"
    )
    parallel_experiments: int = Field(
        default=0,
        ge=0,
        description="Number of parallel experiments (0 = sequential)",
        alias="PARALLEL_EXPERIMENTS",
    )

    # Concurrent operations configuration
    enable_concurrent_operations: bool = Field(
        default=False,
        description="Enable concurrent research operations",
        alias="ENABLE_CONCURRENT_OPERATIONS",
    )
    max_parallel_hypotheses: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum concurrent hypothesis evaluations",
        alias="MAX_PARALLEL_HYPOTHESES",
    )
    max_concurrent_experiments: int = Field(
        default=4,
        ge=1,
        le=16,
        description="Maximum concurrent experiment executions",
        alias="MAX_CONCURRENT_EXPERIMENTS",
    )
    max_concurrent_llm_calls: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum concurrent LLM API calls",
        alias="MAX_CONCURRENT_LLM_CALLS",
    )
    llm_rate_limit_per_minute: int = Field(
        default=50,
        ge=1,
        le=200,
        description="LLM API rate limit per minute",
        alias="LLM_RATE_LIMIT_PER_MINUTE",
    )
    async_batch_timeout: int = Field(
        default=300,
        ge=10,
        le=3600,
        description="Timeout for async batch operations (seconds)",
        alias="ASYNC_BATCH_TIMEOUT",
    )

    model_config = SettingsConfigDict(populate_by_name=True)


class LocalModelConfig(BaseSettings):
    """Configuration for local models (Ollama, LM Studio, etc.).

    These settings optimize behavior when using local LLM providers
    that may have different characteristics than cloud APIs.
    """

    # Retry configuration
    max_retries: int = Field(
        default=1,
        ge=0,
        le=5,
        description="Maximum retry attempts for local models (lower than cloud)",
        alias="LOCAL_MODEL_MAX_RETRIES",
    )

    # JSON parsing
    strict_json: bool = Field(
        default=False,
        description="Require strict JSON compliance (False allows lenient parsing)",
        alias="LOCAL_MODEL_STRICT_JSON",
    )

    json_retry_with_hint: bool = Field(
        default=True,
        description="On JSON parse failure, retry with explicit formatting hint",
        alias="LOCAL_MODEL_JSON_RETRY_HINT",
    )

    # Timeouts and resource management
    request_timeout: int = Field(
        default=120,
        ge=30,
        le=600,
        description="Timeout for local model requests in seconds",
        alias="LOCAL_MODEL_REQUEST_TIMEOUT",
    )

    concurrent_requests: int = Field(
        default=1,
        ge=1,
        le=4,
        description="Max concurrent requests to local model (limited by VRAM)",
        alias="LOCAL_MODEL_CONCURRENT_REQUESTS",
    )

    # Graceful degradation
    fallback_to_unstructured: bool = Field(
        default=True,
        description="On structured output failure, try unstructured extraction",
        alias="LOCAL_MODEL_FALLBACK_UNSTRUCTURED",
    )

    # Circuit breaker settings
    circuit_breaker_threshold: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Consecutive failures before circuit breaker opens",
        alias="LOCAL_MODEL_CB_THRESHOLD",
    )

    circuit_breaker_reset_timeout: int = Field(
        default=60,
        ge=10,
        le=300,
        description="Seconds before circuit breaker allows retry",
        alias="LOCAL_MODEL_CB_RESET_TIMEOUT",
    )

    model_config = SettingsConfigDict(populate_by_name=True)


class MonitoringConfig(BaseSettings):
    """Monitoring and metrics configuration."""

    enable_usage_stats: bool = Field(
        default=True, description="Enable usage statistics tracking", alias="ENABLE_USAGE_STATS"
    )
    metrics_export_interval: int = Field(
        default=60,
        ge=0,
        description="Metrics export interval in seconds (0 = disabled)",
        alias="METRICS_EXPORT_INTERVAL",
    )

    model_config = SettingsConfigDict(populate_by_name=True)


class DevelopmentConfig(BaseSettings):
    """Development settings."""

    hot_reload: bool = Field(
        default=False, description="Enable hot reload (development only)", alias="HOT_RELOAD"
    )
    log_api_requests: bool = Field(
        default=False, description="Log all API requests", alias="LOG_API_REQUESTS"
    )
    test_mode: bool = Field(default=False, description="Test mode (uses mocks)", alias="TEST_MODE")

    model_config = SettingsConfigDict(populate_by_name=True)


class WorldModelConfig(BaseSettings):
    """World model configuration for persistent knowledge graphs."""

    enabled: bool = Field(
        default=True, description="Enable persistent knowledge graphs", alias="WORLD_MODEL_ENABLED"
    )

    mode: Literal["simple", "production"] = Field(
        default="simple",
        description="Storage mode: simple (Neo4j) or production (polyglot)",
        alias="WORLD_MODEL_MODE",
    )

    project: str | None = Field(
        default=None,
        description="Default project namespace for multi-project support",
        alias="WORLD_MODEL_PROJECT",
    )

    auto_save_interval: int = Field(
        default=300,
        ge=0,
        description="Auto-export interval in seconds (0 = disabled)",
        alias="WORLD_MODEL_AUTO_SAVE_INTERVAL",
    )

    model_config = SettingsConfigDict(populate_by_name=True)


def _create_llm_config() -> LLMConfig:
    """Create LLMConfig from environment variables."""
    return LLMConfig()


class KosmosConfig(BaseSettings):
    """
    Master configuration for Kosmos AI Scientist.

    Loads all configuration from environment variables with validation.

    Example:
        ```python
        from kosmos.config import get_config

        config = get_config()

        # Access configuration
        print(config.claude.model)  # Backward compatible
        print(config.research.max_iterations)
        print(config.database.url)

        # Multi-provider support
        print(config.llm_provider)  # "anthropic" or "openai"
        if config.llm_provider == "openai":
            print(config.openai.model)

        # Check Claude mode (backward compatible)
        if config.claude.is_cli_mode:
            print("Using Claude Code CLI")
        else:
            print("Using Anthropic API")
        ```
    """

    # LLM Configuration (unified for all providers via DSPy)
    llm: LLMConfig = Field(
        default_factory=_create_llm_config,
        description="Unified LLM configuration for DSPy",
    )

    # Backward compatibility aliases (deprecated - use llm instead)
    @property
    def claude(self) -> LLMConfig:
        """Backward compatibility: use llm instead."""
        return self.llm

    @property
    def anthropic(self) -> LLMConfig:
        """Backward compatibility: use llm instead."""
        return self.llm

    @property
    def openai(self) -> LLMConfig:
        """Backward compatibility: use llm instead."""
        return self.llm

    @property
    def litellm(self) -> LLMConfig:
        """Backward compatibility: use llm instead."""
        return self.llm

    @property
    def llm_provider(self) -> str:
        """Backward compatibility: extract provider from model string."""
        return self.llm.provider

    local_model: LocalModelConfig = Field(
        default_factory=LocalModelConfig
    )  # Local model settings (Ollama, etc.)
    research: ResearchConfig = Field(default_factory=ResearchConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    redis: RedisConfig = Field(default_factory=RedisConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    literature: LiteratureConfig = Field(default_factory=LiteratureConfig)
    vector_db: VectorDBConfig = Field(default_factory=VectorDBConfig)
    neo4j: Neo4jConfig = Field(default_factory=Neo4jConfig)
    safety: SafetyConfig = Field(default_factory=SafetyConfig)
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    development: DevelopmentConfig = Field(default_factory=DevelopmentConfig)
    world_model: WorldModelConfig = Field(default_factory=WorldModelConfig)

    model_config = SettingsConfigDict(
        env_file=str(Path(__file__).parent.parent / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    @model_validator(mode="after")
    def validate_provider_config(self):
        """Validate LLM configuration."""
        # Check if model is specified
        if not self.llm.model:
            raise ValueError(
                "LLM_MODEL is required. " "Example: LLM_MODEL=anthropic/claude-sonnet-4-5"
            )

        # Warn if API key might be needed but not set
        provider = self.llm.provider
        if provider in ["anthropic", "openai", "deepseek"] and not self.llm.api_key:
            logger = logging.getLogger(__name__)
            logger.warning(
                f"LLM_API_KEY not set for {provider} provider. "
                f"This is required for cloud providers. Set LLM_API_KEY in your environment."
            )

        return self

    def create_directories(self):
        """Create necessary directories if they don't exist."""
        # Create log directory
        if self.logging.file:
            log_dir = Path(self.logging.file).parent
            log_dir.mkdir(parents=True, exist_ok=True)

        # Create ChromaDB directory if using ChromaDB
        if self.vector_db.type == "chromadb":
            Path(self.vector_db.chroma_persist_directory).mkdir(parents=True, exist_ok=True)

    def validate_dependencies(self) -> list[str]:
        """
        Check if all required dependencies are available.

        Returns:
            List[str]: List of missing dependencies (empty if all present)
        """
        missing = []

        # Check LLM configuration
        if not self.llm.model:
            missing.append("LLM_MODEL not set (required for LLM operations)")

        # Check API key for cloud providers
        provider = self.llm.provider
        if provider in ["anthropic", "openai", "deepseek"] and not self.llm.api_key:
            missing.append(f"LLM_API_KEY not set (required for {provider} provider)")

        # Check Pinecone if selected
        if self.vector_db.type == "pinecone" and not self.vector_db.pinecone_api_key:
            missing.append("PINECONE_API_KEY not set")

        return missing

    def to_dict(self) -> dict:
        """
        Convert configuration to dictionary.

        Returns:
            dict: Configuration as dictionary
        """
        config_dict = {
            "llm": model_to_dict(self.llm),
            "llm_provider": self.llm_provider,  # Backward compatibility
            "research": model_to_dict(self.research),
            "database": model_to_dict(self.database),
            "redis": model_to_dict(self.redis),
            "logging": model_to_dict(self.logging),
            "literature": model_to_dict(self.literature),
            "vector_db": model_to_dict(self.vector_db),
            "neo4j": model_to_dict(self.neo4j),
            "safety": model_to_dict(self.safety),
            "performance": model_to_dict(self.performance),
            "monitoring": model_to_dict(self.monitoring),
            "development": model_to_dict(self.development),
            "world_model": model_to_dict(self.world_model),
            "local_model": model_to_dict(self.local_model),
        }

        return config_dict


# Singleton configuration instance
_config: KosmosConfig | None = None


def get_config(reload: bool = False) -> KosmosConfig:
    """
    Get or create configuration singleton.

    Args:
        reload: If True, reload configuration from environment

    Returns:
        KosmosConfig: Configuration instance
    """
    global _config
    if _config is None or reload:
        _config = KosmosConfig()
        _config.create_directories()
    return _config


def reset_config():
    """Reset configuration singleton (useful for testing)."""
    global _config
    _config = None
