"""
Experiment result data models.

Defines Pydantic models for experiment results, extending the database Result model
with validation and structured data handling.
"""

import logging
import platform as platform_module
import sys
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

from kosmos.utils.compat import model_to_dict


class ResultStatus(str, Enum):
    """Status of experiment result."""

    SUCCESS = "success"
    FAILED = "failed"
    PARTIAL = "partial"
    TIMEOUT = "timeout"
    ERROR = "error"


class ExecutionMetadata(BaseModel):
    """Metadata about experiment execution."""

    start_time: datetime = Field(..., description="Execution start timestamp")
    end_time: datetime = Field(..., description="Execution end timestamp")
    duration_seconds: float = Field(..., ge=0, description="Execution duration in seconds")

    # System information
    # Provide sensible defaults so metadata can be created without explicit system info
    python_version: str = Field(
        default_factory=lambda: sys.version, description="Python version used"
    )
    platform: str = Field(
        default_factory=lambda: platform_module.system(), description="Operating system platform"
    )
    hostname: str | None = Field(None, description="Machine hostname")

    # Resource usage
    cpu_time_seconds: float | None = Field(None, ge=0, description="CPU time consumed")
    memory_peak_mb: float | None = Field(None, ge=0, description="Peak memory usage in MB")

    # Execution environment
    random_seed: int | None = Field(None, description="Random seed used")
    library_versions: dict[str, str] = Field(
        default_factory=dict, description="Versions of key libraries used"
    )

    # Parameters
    experiment_id: str = Field(..., description="Experiment ID")
    # protocol_id may be omitted in some contexts (e.g. lightweight test metadata)
    protocol_id: str | None = Field(None, description="Protocol ID")
    hypothesis_id: str | None = Field(None, description="Hypothesis ID")

    # Execution details
    sandbox_used: bool = Field(default=False, description="Whether sandbox was used")
    timeout_occurred: bool = Field(default=False, description="Whether timeout occurred")
    errors: list[str] = Field(default_factory=list, description="Errors encountered")
    warnings: list[str] = Field(default_factory=list, description="Warnings generated")


class StatisticalTestResult(BaseModel):
    """Result of a statistical test."""

    test_type: str = Field(..., description="Type of test (t-test, ANOVA, etc.)")
    test_name: str = Field(..., description="Human-readable test name")

    # Test statistics
    statistic: float = Field(..., description="Test statistic value")
    p_value: float = Field(..., ge=0, le=1, description="P-value")

    # Effect size (optional)
    effect_size: float | None = Field(None, description="Effect size measure")
    effect_size_type: str | None = Field(None, description="Type of effect size (Cohen's d, etc.)")

    # Confidence intervals (optional)
    confidence_interval: dict[str, float] | None = Field(
        None, description="Confidence interval (lower, upper)"
    )
    confidence_level: float | None = Field(None, ge=0, le=1, description="CI confidence level")

    # Significance
    significant_0_05: bool = Field(False, description="Significant at α=0.05")
    significant_0_01: bool = Field(False, description="Significant at α=0.01")
    significant_0_001: bool = Field(False, description="Significant at α=0.001")
    significance_label: str = Field("ns", description="Significance label (**, *, ns)")

    # Primary test designation
    is_primary: bool = Field(default=False, description="Whether this is the primary test")

    # Sample information
    sample_size: int | None = Field(None, ge=0, description="Total sample size")
    degrees_of_freedom: int | None = Field(None, description="Degrees of freedom")

    # Additional test-specific data
    additional_stats: dict[str, Any] = Field(
        default_factory=dict, description="Additional test-specific statistics"
    )

    # Interpretation
    interpretation: str | None = Field(None, description="Human-readable interpretation")

    @model_validator(mode="after")
    def _infer_significance(self):
        """Infer significance flags and label from p_value when not explicitly provided."""
        # Fields explicitly provided during model creation are tracked in
        # __pydantic_fields_set__ (Pydantic v2). Only infer when flags/label
        # were not explicitly set.
        fields_set = getattr(self, "__pydantic_fields_set__", set())

        if self.p_value is not None:
            inferred_005 = self.p_value < 0.05
            inferred_001 = self.p_value < 0.01
            inferred_0001 = self.p_value < 0.001

            if "significant_0_05" not in fields_set:
                self.significant_0_05 = inferred_005
            if "significant_0_01" not in fields_set:
                self.significant_0_01 = inferred_001
            if "significant_0_001" not in fields_set:
                self.significant_0_001 = inferred_0001

            if "significance_label" not in fields_set:
                if inferred_0001:
                    self.significance_label = "***"
                elif inferred_001:
                    self.significance_label = "**"
                elif inferred_005:
                    self.significance_label = "*"
                else:
                    self.significance_label = "ns"

            logging.getLogger(__name__).debug(
                "Inferred significance flags for p_value=%s: %s/%s/%s",
                self.p_value,
                self.significant_0_05,
                self.significant_0_01,
                self.significant_0_001,
            )

        return self


class VariableResult(BaseModel):
    """Result data for a specific variable."""

    variable_name: str = Field(..., description="Variable name")
    variable_type: str = Field(..., description="Variable type (independent, dependent, etc.)")

    # Summary statistics
    mean: float | None = Field(None, description="Mean value")
    median: float | None = Field(None, description="Median value")
    std: float | None = Field(None, description="Standard deviation")
    min: float | None = Field(None, description="Minimum value")
    max: float | None = Field(None, description="Maximum value")

    # Data points
    values: list[float | int | str] | None = Field(
        None, description="Raw data values (if not too large)"
    )
    n_samples: int | None = Field(None, ge=0, description="Number of samples")
    n_missing: int | None = Field(None, ge=0, description="Number of missing values")


class ExperimentResult(BaseModel):
    """
    Complete experiment result model.

    Extends database Result model with structured Pydantic validation.
    """

    # Identifiers
    id: str | None = Field(None, description="Result ID")
    experiment_id: str = Field(..., description="Experiment ID")
    protocol_id: str = Field(..., description="Protocol ID used")
    hypothesis_id: str | None = Field(None, description="Hypothesis ID tested")

    # Status
    status: ResultStatus = Field(..., description="Result status")

    # Data
    raw_data: dict[str, Any] = Field(
        default_factory=dict, description="Raw output data from execution"
    )
    processed_data: dict[str, Any] = Field(
        default_factory=dict, description="Processed/cleaned data"
    )

    # Variables
    variable_results: list[VariableResult] = Field(
        default_factory=list, description="Results for each variable"
    )

    # Statistical tests
    statistical_tests: list[StatisticalTestResult] = Field(
        default_factory=list, description="Statistical test results"
    )
    primary_test: str | None = Field(None, description="Name of primary statistical test")

    # Key results
    primary_p_value: float | None = Field(None, ge=0, le=1, description="Primary p-value")
    primary_effect_size: float | None = Field(None, description="Primary effect size")
    primary_ci_lower: float | None = Field(
        None, description="Primary confidence interval lower bound"
    )
    primary_ci_upper: float | None = Field(
        None, description="Primary confidence interval upper bound"
    )
    supports_hypothesis: bool | None = Field(
        None, description="Whether results support the hypothesis"
    )

    # Metadata
    metadata: ExecutionMetadata = Field(..., description="Execution metadata")

    # Versioning
    version: int = Field(default=1, ge=1, description="Result version number")
    parent_result_id: str | None = Field(None, description="ID of parent result if re-run")

    # Outputs
    stdout: str | None = Field(None, description="Standard output from execution")
    stderr: str | None = Field(None, description="Standard error from execution")
    generated_files: list[str] = Field(
        default_factory=list, description="Paths to generated files (plots, data, etc.)"
    )

    # Interpretation
    summary: str | None = Field(None, description="Summary of results")
    interpretation: str | None = Field(None, description="Interpretation of findings")
    recommendations: list[str] = Field(
        default_factory=list, description="Recommendations for follow-up"
    )

    # Timestamps
    created_at: datetime = Field(
        default_factory=datetime.utcnow, description="Result creation time"
    )
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update time")

    @field_validator("statistical_tests")
    @classmethod
    def validate_statistical_tests(
        cls, v: list[StatisticalTestResult]
    ) -> list[StatisticalTestResult]:
        """Validate statistical tests list."""
        # Check for duplicate test names
        test_names = [test.test_name for test in v]
        if len(test_names) != len(set(test_names)):
            raise ValueError("Duplicate test names in statistical_tests")
        return v

    @field_validator("primary_test")
    @classmethod
    def validate_primary_test(cls, v: str | None, info) -> str | None:
        """Validate primary test exists in statistical_tests."""
        # Only validate when a primary_test is provided and there are tests to match against
        if v is not None and "statistical_tests" in info.data and info.data["statistical_tests"]:
            # In Pydantic V2, info.data contains raw dicts, not model instances
            test_names = [
                test["test_name"] if isinstance(test, dict) else test.test_name
                for test in info.data["statistical_tests"]
            ]
            if v not in test_names:
                raise ValueError(f"Primary test '{v}' not found in statistical_tests")
        return v

    def get_primary_test_result(self) -> StatisticalTestResult | None:
        """Get the primary statistical test result."""
        if self.primary_test is None:
            return None

        for test in self.statistical_tests:
            if test.test_name == self.primary_test:
                return test

        return None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for database storage."""
        return model_to_dict(self, mode="json", exclude_none=True)

    def to_json(self) -> str:
        """Convert to JSON string."""
        return self.model_dump_json(indent=2, exclude_none=True)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExperimentResult":
        """Create from dictionary."""
        return cls.model_validate(data)

    @classmethod
    def from_json(cls, json_str: str) -> "ExperimentResult":
        """Create from JSON string."""
        return cls.model_validate_json(json_str)

    def is_significant(self, alpha: float = 0.05) -> bool:
        """Check if primary result is statistically significant."""
        if self.primary_p_value is None:
            return False
        return self.primary_p_value < alpha

    def get_summary_stats(self) -> dict[str, Any]:
        """Get summary statistics for all variables."""
        stats = {}
        for var_result in self.variable_results:
            stats[var_result.variable_name] = {
                "mean": var_result.mean,
                "median": var_result.median,
                "std": var_result.std,
                "min": var_result.min,
                "max": var_result.max,
                "n_samples": var_result.n_samples,
            }
        return stats


class ResultExport(BaseModel):
    """Model for exporting results in various formats."""

    result: ExperimentResult
    format: str = Field(..., description="Export format (json, csv, markdown)")

    def export_json(self) -> str:
        """Export as JSON."""
        return self.result.to_json()

    def export_csv(self) -> str:
        """Export variable results as CSV."""
        import pandas as pd

        # Create DataFrame from variable results
        data = []
        for var in self.result.variable_results:
            row = {
                "variable": var.variable_name,
                "type": var.variable_type,
                "mean": var.mean,
                "median": var.median,
                "std": var.std,
                "min": var.min,
                "max": var.max,
                "n_samples": var.n_samples,
            }
            data.append(row)

        df = pd.DataFrame(data)
        return df.to_csv(index=False)

    def export_markdown(self) -> str:
        """Export as Markdown report."""
        lines = []

        # Header
        lines.append("# Experiment Result Report")
        lines.append(f"**Experiment ID:** {self.result.experiment_id}")
        lines.append(f"**Status:** {self.result.status.value}")
        lines.append(f"**Date:** {self.result.created_at.isoformat()}")
        lines.append("")

        # Summary
        if self.result.summary:
            lines.append("## Summary")
            lines.append(self.result.summary)
            lines.append("")

        # Statistical Tests
        if self.result.statistical_tests:
            lines.append("## Statistical Tests")
            lines.append("")
            for test in self.result.statistical_tests:
                lines.append(f"### {test.test_name}")
                lines.append(f"- **Statistic:** {test.statistic:.4f}")
                lines.append(f"- **P-value:** {test.p_value:.6f}")
                lines.append(f"- **Significance:** {test.significance_label}")
                if test.effect_size:
                    lines.append(
                        f"- **Effect Size:** {test.effect_size:.4f} ({test.effect_size_type})"
                    )
                lines.append("")

        # Variables
        if self.result.variable_results:
            lines.append("## Variable Statistics")
            lines.append("")
            lines.append("| Variable | Mean | Median | Std | Min | Max | N |")
            lines.append("|----------|------|--------|-----|-----|-----|---|")
            for var in self.result.variable_results:
                lines.append(
                    f"| {var.variable_name} | "
                    f"{var.mean:.2f} | {var.median:.2f} | {var.std:.2f} | "
                    f"{var.min:.2f} | {var.max:.2f} | {var.n_samples} |"
                )
            lines.append("")

        # Interpretation
        if self.result.interpretation:
            lines.append("## Interpretation")
            lines.append(self.result.interpretation)
            lines.append("")

        # Recommendations
        if self.result.recommendations:
            lines.append("## Recommendations")
            for rec in self.result.recommendations:
                lines.append(f"- {rec}")
            lines.append("")

        # Metadata
        lines.append("## Execution Metadata")
        lines.append(f"- **Duration:** {self.result.metadata.duration_seconds:.2f}s")
        lines.append(f"- **Python Version:** {self.result.metadata.python_version}")
        if self.result.metadata.memory_peak_mb:
            lines.append(f"- **Peak Memory:** {self.result.metadata.memory_peak_mb:.1f} MB")
        lines.append("")

        return "\n".join(lines)
