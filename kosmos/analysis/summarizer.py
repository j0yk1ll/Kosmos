"""
Result Summarization.

Natural language summaries of experiment results using DSPy.
"""

import logging
from datetime import datetime
from typing import Any

import dspy

from kosmos.config import get_config
from kosmos.models.hypothesis import Hypothesis
from kosmos.models.result import ExperimentResult


logger = logging.getLogger(__name__)


class ResultSummarySignature(dspy.Signature):
    """Generate comprehensive scientific summary of experiment results."""

    hypothesis: str = dspy.InputField(desc="Original hypothesis statement")
    experimental_results: str = dspy.InputField(desc="Experimental results summary")
    interpretation: str = dspy.InputField(desc="Data analysis interpretation")
    literature_context: str = dspy.InputField(desc="Relevant literature context")

    summary: str = dspy.OutputField(desc="2-3 paragraph natural language summary")
    key_findings: str = dspy.OutputField(desc="Numbered list of 3-5 key findings")
    hypothesis_assessment: str = dspy.OutputField(desc="How results relate to hypothesis")
    limitations: str = dspy.OutputField(desc="Bullet list of limitations")
    future_work: str = dspy.OutputField(desc="Numbered list of suggested follow-up experiments")


class ResultSummary:
    """Structured natural language summary of results."""

    def __init__(
        self,
        experiment_id: str,
        summary: str,
        key_findings: list[str],
        hypothesis_comparison: str,
        limitations: list[str],
        future_work: list[str],
        created_at: datetime | None = None,
    ):
        """
        Initialize result summary.

        Args:
            experiment_id: ID of experiment
            summary: 2-3 paragraph natural language summary
            key_findings: List of 3-5 key findings
            hypothesis_comparison: Comparison to original hypothesis
            limitations: List of limitations and caveats
            future_work: List of suggested follow-up experiments
            created_at: Timestamp
        """
        self.experiment_id = experiment_id
        self.summary = summary
        self.key_findings = key_findings
        self.hypothesis_comparison = hypothesis_comparison
        self.limitations = limitations
        self.future_work = future_work
        self.created_at = created_at or datetime.utcnow()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "experiment_id": self.experiment_id,
            "summary": self.summary,
            "key_findings": self.key_findings,
            "hypothesis_comparison": self.hypothesis_comparison,
            "limitations": self.limitations,
            "future_work": self.future_work,
            "created_at": self.created_at.isoformat(),
        }

    def to_markdown(self) -> str:
        """Convert to markdown format."""
        md = f"# Experiment Result Summary: {self.experiment_id}\n\n"
        md += f"## Summary\n\n{self.summary}\n\n"
        md += "## Key Findings\n\n"
        for i, finding in enumerate(self.key_findings, 1):
            md += f"{i}. {finding}\n"
        md += f"\n## Hypothesis Assessment\n\n{self.hypothesis_comparison}\n\n"
        md += "## Limitations\n\n"
        for limitation in self.limitations:
            md += f"- {limitation}\n"
        md += "\n## Recommended Future Work\n\n"
        for i, work in enumerate(self.future_work, 1):
            md += f"{i}. {work}\n"
        return md


class ResultSummarizer:
    """
    Natural language result summarization using Claude.

    Capabilities:
    - Generate plain-language summaries of results
    - Extract key findings
    - Compare results to hypothesis
    - Identify limitations
    - Suggest follow-up experiments

    Example:
        ```python
        summarizer = ResultSummarizer()

        summary = summarizer.generate_summary(
            result=experiment_result,
            hypothesis=original_hypothesis,
            interpretation=data_analyst_interpretation
        )

        print(summary.summary)
        print("\\n".join(summary.key_findings))
        ```
    """

    def __init__(self, llm_config: dict[str, Any] | None = None):
        """Initialize result summarizer.

        Args:
            llm_config: Optional LLM configuration dict for DSPy (uses default config if not provided)
        """
        # Initialize DSPy LM
        if llm_config is None:
            config = get_config()
            llm_config = config.llm.to_dspy_config()
        self.llm = dspy.LM(**llm_config)
        logger.info("ResultSummarizer initialized with DSPy")

    def generate_summary(
        self,
        result: ExperimentResult,
        hypothesis: Hypothesis | None = None,
        interpretation: dict[str, Any] | None = None,
        literature_context: str | None = None,
    ) -> ResultSummary:
        """
        Generate comprehensive natural language summary.

        Args:
            result: ExperimentResult object
            hypothesis: Optional original hypothesis
            interpretation: Optional interpretation from DataAnalystAgent
            literature_context: Optional literature context

        Returns:
            ResultSummary object
        """
        logger.info(f"Generating summary for experiment {result.experiment_id}")

        # Prepare inputs for DSPy
        hypothesis_str = hypothesis.statement if hypothesis else "No hypothesis provided"

        experimental_results = f"""Primary Test: {result.primary_test}
P-value: {result.primary_p_value}
Effect Size: {result.primary_effect_size}
Hypothesis Supported: {result.supports_hypothesis}
Status: {result.status}"""

        interpretation_str = (
            str(interpretation.get("summary", "No interpretation provided"))
            if interpretation
            else "No interpretation provided"
        )

        literature_context_str = (
            literature_context[:500] if literature_context else "No literature context provided"
        )

        # Get DSPy summary
        try:
            with dspy.context(lm=self.llm):
                predictor = dspy.Predict(ResultSummarySignature)
                response = predictor(
                    hypothesis=hypothesis_str,
                    experimental_results=experimental_results,
                    interpretation=interpretation_str,
                    literature_context=literature_context_str,
                )

            # Parse response
            summary = self._parse_dspy_response(response, result.experiment_id)

            logger.info(f"Completed summary for {result.experiment_id}")
            return summary

        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return self._create_fallback_summary(result, hypothesis)

    def extract_key_findings(self, result: ExperimentResult, max_findings: int = 5) -> list[str]:
        """
        Extract key findings from result.

        Args:
            result: ExperimentResult object
            max_findings: Maximum number of findings

        Returns:
            List of key finding strings
        """
        findings = []

        # Primary result
        if result.primary_test and result.primary_p_value is not None:
            finding = f"{result.primary_test}: "
            if result.primary_p_value < 0.001:
                finding += f"highly significant (p={result.primary_p_value:.4e})"
            elif result.primary_p_value < 0.05:
                finding += f"significant (p={result.primary_p_value:.4f})"
            else:
                finding += f"not significant (p={result.primary_p_value:.4f})"

            if result.primary_effect_size is not None:
                finding += f", effect size = {result.primary_effect_size:.3f}"

            findings.append(finding)

        # Additional statistical tests
        for test in result.statistical_tests[: max_findings - 1]:
            if not test.is_primary:
                test_finding = f"{test.test_name}: p={test.p_value:.4f}"
                if test.effect_size is not None:
                    test_finding += f", effect size = {test.effect_size:.3f}"
                findings.append(test_finding)

        return findings[:max_findings]

    def compare_to_hypothesis(self, result: ExperimentResult, hypothesis: Hypothesis) -> str:
        """
        Compare result to original hypothesis.

        Args:
            result: ExperimentResult object
            hypothesis: Original hypothesis

        Returns:
            Comparison string
        """
        comparison_parts = []

        # Overall support
        if result.supports_hypothesis is True:
            comparison_parts.append(
                f'The experimental results SUPPORT the hypothesis: "{hypothesis.statement}"'
            )
        elif result.supports_hypothesis is False:
            comparison_parts.append(
                f'The experimental results DO NOT SUPPORT the hypothesis: "{hypothesis.statement}"'
            )
        else:
            comparison_parts.append(
                f'The experimental results are INCONCLUSIVE regarding the hypothesis: "{hypothesis.statement}"'
            )

        # Evidence strength
        if result.primary_p_value is not None:
            if result.primary_p_value < 0.01:
                comparison_parts.append("The statistical evidence is strong (p < 0.01).")
            elif result.primary_p_value < 0.05:
                comparison_parts.append("The statistical evidence is moderate (p < 0.05).")
            else:
                comparison_parts.append("The statistical evidence is weak (p > 0.05).")

        # Effect size consideration
        if result.primary_effect_size is not None:
            if abs(result.primary_effect_size) >= 0.8:
                comparison_parts.append(
                    "The effect size is large, suggesting practical significance."
                )
            elif abs(result.primary_effect_size) >= 0.5:
                comparison_parts.append("The effect size is medium.")
            elif abs(result.primary_effect_size) >= 0.2:
                comparison_parts.append("The effect size is small.")
            else:
                comparison_parts.append("The effect size is negligible.")

        return " ".join(comparison_parts)

    def identify_limitations(
        self, result: ExperimentResult, hypothesis: Hypothesis | None = None
    ) -> list[str]:
        """
        Identify limitations of experiment.

        Args:
            result: ExperimentResult object
            hypothesis: Optional hypothesis

        Returns:
            List of limitation strings
        """
        limitations = []

        # Check for execution issues
        if result.status != "success":
            limitations.append(
                f"Experiment did not complete successfully (status: {result.status})"
            )

        # Check sample size (if available from metadata)
        for test in result.statistical_tests:
            if test.sample_size is not None and test.sample_size < 30:
                limitations.append(
                    f"Small sample size (n={test.sample_size}) may limit statistical power"
                )
                break

        # Check for missing confidence intervals
        if result.primary_ci_lower is None or result.primary_ci_upper is None:
            limitations.append("Confidence intervals not reported, limiting precision of estimate")

        # Generic limitations
        limitations.append("Replication in independent dataset recommended to confirm findings")
        limitations.append("Potential confounding variables should be considered in interpretation")

        return limitations

    def suggest_future_work(
        self,
        result: ExperimentResult,
        hypothesis: Hypothesis | None = None,
        max_suggestions: int = 5,
    ) -> list[str]:
        """
        Suggest follow-up experiments.

        Args:
            result: ExperimentResult object
            hypothesis: Optional hypothesis
            max_suggestions: Maximum suggestions

        Returns:
            List of suggestion strings
        """
        suggestions = []

        # Based on result support
        if result.supports_hypothesis is True:
            suggestions.append("Replicate findings in independent cohort to validate results")
            suggestions.append("Investigate mechanism underlying observed effect")
            suggestions.append("Test dose-response relationship if applicable")
        elif result.supports_hypothesis is False:
            suggestions.append("Investigate why hypothesis was not supported")
            suggestions.append("Test alternative hypotheses based on unexpected findings")
        else:
            suggestions.append("Increase sample size to improve statistical power")
            suggestions.append("Refine experimental design to reduce variability")

        # Generic suggestions
        suggestions.append("Conduct sensitivity analysis to test robustness of findings")
        suggestions.append("Explore potential moderating variables")

        return suggestions[:max_suggestions]

    # ========================================================================
    # INTERNAL METHODS
    # ========================================================================

    def _parse_dspy_response(self, response: Any, experiment_id: str) -> ResultSummary:
        """Parse DSPy response into ResultSummary."""
        # Extract summary text
        summary = response.summary if hasattr(response, "summary") else "No summary generated"

        # Parse key findings (numbered list)
        key_findings_text = response.key_findings if hasattr(response, "key_findings") else ""
        key_findings = []
        for line in key_findings_text.split("\n"):
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith("-")):
                key_findings.append(line.lstrip("0123456789.-) "))

        # Extract hypothesis assessment
        hypothesis_comparison = (
            response.hypothesis_assessment
            if hasattr(response, "hypothesis_assessment")
            else "No hypothesis assessment"
        )

        # Parse limitations (bullet list)
        limitations_text = response.limitations if hasattr(response, "limitations") else ""
        limitations = []
        for line in limitations_text.split("\n"):
            line = line.strip()
            if line and line.startswith("-"):
                limitations.append(line.lstrip("- "))

        # Parse future work (numbered list)
        future_work_text = response.future_work if hasattr(response, "future_work") else ""
        future_work = []
        for line in future_work_text.split("\n"):
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith("-")):
                future_work.append(line.lstrip("0123456789.-) "))

        return ResultSummary(
            experiment_id=experiment_id,
            summary=summary.strip(),
            key_findings=key_findings,
            hypothesis_comparison=hypothesis_comparison.strip(),
            limitations=limitations,
            future_work=future_work,
        )

    def _create_fallback_summary(
        self, result: ExperimentResult, hypothesis: Hypothesis | None
    ) -> ResultSummary:
        """Create fallback summary if DSPy fails."""
        summary = f"Experiment {result.experiment_id} completed with status {result.status}. "
        summary += (
            f"Primary test ({result.primary_test}) yielded p-value of {result.primary_p_value}. "
        )

        if result.supports_hypothesis is not None:
            summary += f"Results {'support' if result.supports_hypothesis else 'do not support'} the hypothesis."

        return ResultSummary(
            experiment_id=result.experiment_id,
            summary=summary,
            key_findings=self.extract_key_findings(result),
            hypothesis_comparison=(
                self.compare_to_hypothesis(result, hypothesis)
                if hypothesis
                else "No hypothesis provided for comparison."
            ),
            limitations=self.identify_limitations(result, hypothesis),
            future_work=self.suggest_future_work(result, hypothesis),
        )
