"""
Hypothesis Refiner - Refines, retires, and spawns hypotheses based on experimental results (Phase 7).

Implements hybrid retirement logic:
- Rule-based: Consecutive failures
- Confidence-based: Bayesian updating
- LLM-powered: Ambiguous cases
"""

import json
import logging
import uuid
from datetime import datetime
from enum import Enum
from typing import Any

import dspy
from pydantic import BaseModel, Field

from kosmos.config import get_config
from kosmos.knowledge.vector_db import (
    HAS_CHROMADB,
    PaperVectorDB as VectorDB,
)
from kosmos.models.hypothesis import Hypothesis, HypothesisStatus
from kosmos.models.result import ExperimentResult, ResultStatus


logger = logging.getLogger(__name__)


class RetirementEvaluationSignature(dspy.Signature):
    """Evaluate whether a hypothesis should be retired based on experimental evidence."""

    hypothesis_statement: str = dspy.InputField(desc="The hypothesis statement")
    hypothesis_rationale: str = dspy.InputField(desc="The hypothesis rationale")
    domain: str = dspy.InputField(desc="Scientific domain")
    results_summary: str = dspy.InputField(desc="Summary of experimental results")

    decision: str = dspy.OutputField(desc="Decision: retire, refine, or continue")
    confidence: str = dspy.OutputField(desc="Confidence level 0.0-1.0")
    rationale: str = dspy.OutputField(desc="2-3 sentence explanation")
    suggested_action: str = dspy.OutputField(desc="What to do next if not retire")


class HypothesisRefinementSignature(dspy.Signature):
    """Refine a hypothesis based on experimental results."""

    original_statement: str = dspy.InputField(desc="Original hypothesis statement")
    original_rationale: str = dspy.InputField(desc="Original hypothesis rationale")
    domain: str = dspy.InputField(desc="Scientific domain")
    result_summary: str = dspy.InputField(desc="Experimental result details")

    refined_statement: str = dspy.OutputField(desc="Refined hypothesis statement")
    refined_rationale: str = dspy.OutputField(desc="Updated rationale incorporating evidence")
    changes_made: str = dspy.OutputField(desc="What was changed and why")
    confidence: str = dspy.OutputField(desc="Confidence level 0.0-1.0")


class VariantSpawningSignature(dspy.Signature):
    """Generate variant hypotheses based on experimental findings."""

    original_statement: str = dspy.InputField(desc="Original hypothesis statement")
    original_rationale: str = dspy.InputField(desc="Original hypothesis rationale")
    result_summary: str = dspy.InputField(desc="Experimental result details")
    num_variants: str = dspy.InputField(desc="Number of variants to generate")

    variants_json: str = dspy.OutputField(
        desc="JSON array of variant hypotheses with statement, rationale, and relationship"
    )


class HypothesisMergingSignature(dspy.Signature):
    """Merge similar hypotheses into a unified hypothesis."""

    statements: str = dspy.InputField(desc="Hypothesis statements to merge")
    rationales: str = dspy.InputField(desc="Hypothesis rationales")

    merged_statement: str = dspy.OutputField(desc="Unified hypothesis statement")
    merged_rationale: str = dspy.OutputField(desc="Integrated rationale")
    synthesis_explanation: str = dspy.OutputField(desc="How the hypotheses were combined")


class RetirementDecision(str, Enum):
    """Decision on hypothesis status."""

    CONTINUE_TESTING = "continue_testing"
    RETIRE = "retire"
    REFINE = "refine"
    SPAWN_VARIANT = "spawn_variant"


class RefinerAction(str, Enum):
    """Actions taken by the refiner."""

    NO_ACTION = "no_action"
    RETIRED = "retired"
    REFINED = "refined"
    SPAWNED = "spawned"
    MERGED = "merged"


class HypothesisLineage(BaseModel):
    """
    Hypothesis lineage tracking.

    Tracks parent-child relationships for hypothesis evolution.
    """

    hypothesis_id: str
    parent_id: str | None = None
    children_ids: list[str] = Field(default_factory=list)
    generation: int = 1
    refinement_reason: str | None = None
    evidence_basis: list[str] = Field(default_factory=list)  # Result IDs
    created_at: datetime = Field(default_factory=datetime.utcnow)


class HypothesisRefiner:
    """
    Refines hypotheses based on experimental results.

    Capabilities:
    - Evaluate hypothesis status (supported/rejected/inconclusive)
    - Retire hypotheses (rule-based, confidence-based, or Claude decision)
    - Refine hypotheses (improve based on results)
    - Spawn variants (generate new hypotheses from findings)
    - Merge similar supported hypotheses
    - Detect contradictions between hypotheses
    """

    def __init__(
        self,
        llm_config: dict[str, Any] | None = None,
        vector_db: VectorDB | None = None,
        config: dict[str, Any] | None = None,
    ):
        """
        Initialize hypothesis refiner.

        Args:
            llm_config: Optional LLM configuration dict for DSPy
            vector_db: Vector database for semantic similarity
            config: Configuration dict
        """
        # Initialize DSPy LM
        if llm_config is None:
            cfg = get_config()
            llm_config = cfg.llm.to_dspy_config()
        self.llm = dspy.LM(**llm_config)
        self.vector_db = vector_db

        # Configuration
        self.config = config or {}
        self.failure_threshold = self.config.get(
            "failure_threshold", 3
        )  # Consecutive failures before retirement
        self.confidence_retirement_threshold = self.config.get(
            "confidence_retirement_threshold", 0.1
        )  # Bayesian confidence
        self.similarity_threshold = self.config.get(
            "similarity_threshold", 0.8
        )  # For contradiction/merging

        # Tracking
        self.lineage_tracking: dict[str, HypothesisLineage] = {}  # hypothesis_id -> lineage

        logger.info("HypothesisRefiner initialized")

    # ========================================================================
    # HYPOTHESIS EVALUATION
    # ========================================================================

    def evaluate_hypothesis_status(
        self,
        hypothesis: Hypothesis,
        result: ExperimentResult,
        results_history: list[ExperimentResult] | None = None,
    ) -> RetirementDecision:
        """
        Evaluate hypothesis status and decide on next action.

        Uses hybrid approach:
        1. Rule-based: Check consecutive failures
        2. Confidence-based: Bayesian probability update
        3. Claude-powered: For ambiguous cases

        Args:
            hypothesis: Hypothesis to evaluate
            result: Latest experiment result
            results_history: Historical results for this hypothesis

        Returns:
            RetirementDecision: What to do with this hypothesis
        """
        logger.info(f"Evaluating hypothesis {hypothesis.id}")

        results_history = results_history or []
        all_results = results_history + [result]

        # 1. Rule-based check: Consecutive failures
        consecutive_failures = self._count_consecutive_failures(all_results)

        if consecutive_failures >= self.failure_threshold:
            logger.info(
                f"Hypothesis {hypothesis.id} has {consecutive_failures} consecutive failures "
                f"(threshold: {self.failure_threshold})"
            )
            return RetirementDecision.RETIRE

        # 2. Confidence-based check: Bayesian updating
        posterior_confidence = self._bayesian_confidence_update(hypothesis, all_results)

        if posterior_confidence < self.confidence_retirement_threshold:
            logger.info(
                f"Hypothesis {hypothesis.id} has low confidence {posterior_confidence:.3f} "
                f"(threshold: {self.confidence_retirement_threshold})"
            )
            return RetirementDecision.RETIRE

        # 3. Check if should refine vs continue
        if result.supports_hypothesis is False:
            # Rejected but not enough to retire - should refine
            return RetirementDecision.REFINE
        elif result.supports_hypothesis is None:
            # Inconclusive - spawn variant to explore
            return RetirementDecision.SPAWN_VARIANT
        else:
            # Supported - continue testing or spawn variants to explore related ideas
            if len(all_results) >= 2:
                # After 2+ successes, might want to explore variants
                return RetirementDecision.SPAWN_VARIANT
            else:
                return RetirementDecision.CONTINUE_TESTING

    def should_retire_hypothesis_llm(
        self, hypothesis: Hypothesis, results: list[ExperimentResult]
    ) -> tuple[bool, str]:
        """
        Use LLM to decide if hypothesis should be retired (for ambiguous cases).

        Args:
            hypothesis: Hypothesis to evaluate
            results: All experimental results for this hypothesis

        Returns:
            Tuple of (should_retire: bool, rationale: str)
        """
        # Build results summary
        results_summary = self._format_results_for_llm(results)

        try:
            with dspy.context(lm=self.llm):
                predictor = dspy.Predict(RetirementEvaluationSignature)
                response = predictor(
                    hypothesis_statement=hypothesis.statement,
                    hypothesis_rationale=hypothesis.rationale or "No rationale provided",
                    domain=hypothesis.domain or "General",
                    results_summary=results_summary,
                )

            # Parse response
            decision = response.decision if hasattr(response, "decision") else "continue"
            confidence_str = response.confidence if hasattr(response, "confidence") else "0.5"
            rationale = response.rationale if hasattr(response, "rationale") else ""

            try:
                confidence = float(confidence_str)
            except (ValueError, TypeError):
                confidence = 0.5

            should_retire = "retire" in decision.lower()

            logger.info(f"LLM decision for {hypothesis.id}: {decision} (confidence: {confidence})")

            return should_retire, rationale

        except Exception as e:
            logger.error(f"Error getting LLM decision: {e}")
            return False, f"Error: {str(e)}"

    def _count_consecutive_failures(self, results: list[ExperimentResult]) -> int:
        """Count consecutive failures (rejected or error) from most recent results."""
        count = 0
        for result in reversed(results):
            if result.supports_hypothesis is False or result.status == ResultStatus.FAILED:
                count += 1
            else:
                break
        return count

    def _bayesian_confidence_update(
        self, hypothesis: Hypothesis, results: list[ExperimentResult]
    ) -> float:
        """
        Update hypothesis confidence using Bayesian inference.

        Simple Bayesian model:
        - Prior: hypothesis.confidence_score (or 0.5 if None)
        - Likelihood: Based on p-value and effect size
        - Posterior: Updated confidence

        Args:
            hypothesis: Hypothesis
            results: All experimental results

        Returns:
            float: Posterior confidence (0.0 - 1.0)
        """
        # Start with prior
        prior = hypothesis.confidence_score if hypothesis.confidence_score is not None else 0.5

        # Update with each result
        confidence = prior

        for result in results:
            # Likelihood based on support and strength of evidence
            if result.supports_hypothesis is True:
                # Supported - increase confidence
                # Stronger evidence (lower p-value, larger effect) = larger update
                p_value = result.primary_p_value if result.primary_p_value is not None else 0.5
                effect_size = (
                    abs(result.primary_effect_size)
                    if result.primary_effect_size is not None
                    else 0.0
                )

                # Evidence strength (0-1)
                evidence_strength = (1 - p_value) * min(effect_size, 1.0)

                # Bayesian update (simple version)
                confidence = confidence + (1 - confidence) * evidence_strength * 0.3

            elif result.supports_hypothesis is False:
                # Rejected - decrease confidence
                p_value = result.primary_p_value if result.primary_p_value is not None else 0.5
                effect_size = (
                    abs(result.primary_effect_size)
                    if result.primary_effect_size is not None
                    else 0.0
                )

                # Evidence strength against hypothesis
                evidence_strength = (1 - p_value) * min(effect_size, 1.0)

                # Bayesian update
                confidence = confidence * (1 - evidence_strength * 0.3)

            # Inconclusive results don't change confidence much

        return max(0.0, min(1.0, confidence))

    # ========================================================================
    # HYPOTHESIS REFINEMENT
    # ========================================================================

    def refine_hypothesis(self, hypothesis: Hypothesis, result: ExperimentResult) -> Hypothesis:
        """
        Refine hypothesis based on experimental result.

        Creates a new refined version of the hypothesis.

        Args:
            hypothesis: Original hypothesis
            result: Experiment result that suggests refinement

        Returns:
            Hypothesis: Refined hypothesis
        """
        logger.info(f"Refining hypothesis {hypothesis.id}")

        # Build result summary
        result_summary = f"""Supported: {result.supports_hypothesis}
P-value: {result.primary_p_value}
Effect size: {result.primary_effect_size}
Primary test: {result.primary_test}"""

        try:
            with dspy.context(lm=self.llm):
                predictor = dspy.Predict(HypothesisRefinementSignature)
                response = predictor(
                    original_statement=hypothesis.statement,
                    original_rationale=hypothesis.rationale or "No rationale provided",
                    domain=hypothesis.domain or "General",
                    result_summary=result_summary,
                )

            # Parse response
            refined_statement = (
                response.refined_statement
                if hasattr(response, "refined_statement")
                else hypothesis.statement
            )
            refined_rationale = (
                response.refined_rationale
                if hasattr(response, "refined_rationale")
                else hypothesis.rationale
            )
            changes_made = response.changes_made if hasattr(response, "changes_made") else ""
            confidence_str = response.confidence if hasattr(response, "confidence") else "0.5"

            try:
                confidence = float(confidence_str)
            except (ValueError, TypeError):
                confidence = 0.5

            # Create refined hypothesis with new ID
            refined = Hypothesis(
                id=f"hyp_{uuid.uuid4().hex[:12]}",
                research_question=hypothesis.research_question,
                statement=refined_statement,
                rationale=refined_rationale,
                domain=hypothesis.domain,
                status=HypothesisStatus.GENERATED,
                testability_score=hypothesis.testability_score,
                novelty_score=hypothesis.novelty_score,  # May decrease slightly
                confidence_score=confidence,
                priority_score=hypothesis.priority_score,
                parent_hypothesis_id=hypothesis.id,
                generation=hypothesis.generation + 1,
                refinement_count=0,
                evolution_history=[
                    {
                        "action": "refined",
                        "based_on_result": result.id,
                        "changes": changes_made,
                        "timestamp": datetime.utcnow().isoformat(),
                    }
                ],
            )

            # Track lineage
            self._track_lineage(refined, hypothesis, "refined", [result.id])

            logger.info(f"Created refined hypothesis (generation {refined.generation})")
            return refined

        except Exception as e:
            logger.error(f"Error refining hypothesis: {e}")
            return hypothesis

    # ========================================================================
    # HYPOTHESIS SPAWNING
    # ========================================================================

    def spawn_variant(
        self, hypothesis: Hypothesis, result: ExperimentResult, num_variants: int = 2
    ) -> list[Hypothesis]:
        """
        Spawn variant hypotheses based on findings.

        When a hypothesis is supported or shows interesting patterns,
        spawn related hypotheses to explore.

        Args:
            hypothesis: Original hypothesis
            result: Result that suggests variants
            num_variants: Number of variants to generate

        Returns:
            List[Hypothesis]: Spawned variant hypotheses
        """
        logger.info(f"Spawning {num_variants} variants from hypothesis {hypothesis.id}")

        result_summary = f"""Supported: {result.supports_hypothesis}
P-value: {result.primary_p_value}
Effect size: {result.primary_effect_size}"""

        try:
            with dspy.context(lm=self.llm):
                predictor = dspy.Predict(VariantSpawningSignature)
                response = predictor(
                    original_statement=hypothesis.statement,
                    original_rationale=hypothesis.rationale or "No rationale provided",
                    result_summary=result_summary,
                    num_variants=str(num_variants),
                )

            # Parse JSON from response
            variants_json = response.variants_json if hasattr(response, "variants_json") else "[]"

            # Extract JSON array
            json_start = variants_json.find("[")
            json_end = variants_json.rfind("]") + 1
            if json_start >= 0 and json_end > json_start:
                json_str = variants_json[json_start:json_end]
                variants_data = json.loads(json_str)

                variants = []
                for _i, variant_data in enumerate(variants_data[:num_variants]):
                    variant = Hypothesis(
                        research_question=hypothesis.research_question,
                        statement=variant_data.get("statement", ""),
                        rationale=variant_data.get("rationale", ""),
                        domain=hypothesis.domain,
                        status=HypothesisStatus.GENERATED,
                        parent_hypothesis_id=hypothesis.id,
                        generation=hypothesis.generation + 1,
                        evolution_history=[
                            {
                                "action": "spawned",
                                "based_on_result": result.id,
                                "relationship": variant_data.get("relationship", ""),
                                "timestamp": datetime.utcnow().isoformat(),
                            }
                        ],
                    )

                    self._track_lineage(variant, hypothesis, "spawned", [result.id])
                    variants.append(variant)

                logger.info(f"Spawned {len(variants)} variant hypotheses")
                return variants
            else:
                logger.warning("Could not parse variants JSON")
                return []

        except Exception as e:
            logger.error(f"Error spawning variants: {e}")
            return []

    # ========================================================================
    # HYPOTHESIS RETIREMENT
    # ========================================================================

    def retire_hypothesis(self, hypothesis: Hypothesis, rationale: str) -> Hypothesis:
        """
        Retire a hypothesis.

        Args:
            hypothesis: Hypothesis to retire
            rationale: Reason for retirement

        Returns:
            Hypothesis: Updated hypothesis with REJECTED status
        """
        hypothesis.status = HypothesisStatus.REJECTED
        hypothesis.updated_at = datetime.utcnow()
        hypothesis.evolution_history.append(
            {
                "action": "retired",
                "rationale": rationale,
                "timestamp": datetime.utcnow().isoformat(),
            }
        )

        logger.info(f"Retired hypothesis {hypothesis.id}: {rationale}")
        return hypothesis

    # ========================================================================
    # CONTRADICTION DETECTION
    # ========================================================================

    def detect_contradictions(
        self,
        hypotheses: list[Hypothesis],
        results: dict[str, list[ExperimentResult]],  # hypothesis_id -> results
    ) -> list[dict[str, Any]]:
        """
        Detect contradictions between hypotheses.

        Contradictions occur when:
        - Two similar hypotheses have opposite outcomes
        - Results support H1 but reject H2 where H1 and H2 are semantically similar

        Args:
            hypotheses: List of hypotheses to check
            results: Results for each hypothesis

        Returns:
            List of contradiction dicts
        """
        logger.info(f"Detecting contradictions among {len(hypotheses)} hypotheses")

        contradictions = []

        # Compare all pairs
        for i, hyp1 in enumerate(hypotheses):
            for hyp2 in hypotheses[i + 1 :]:
                # Check semantic similarity
                similarity = self._compute_semantic_similarity(hyp1.statement, hyp2.statement)

                if similarity >= self.similarity_threshold:
                    # Check if outcomes contradict
                    results1 = results.get(hyp1.id, [])
                    results2 = results.get(hyp2.id, [])

                    support1 = self._overall_support(results1)
                    support2 = self._overall_support(results2)

                    if support1 is not None and support2 is not None and support1 != support2:
                        # Contradiction: similar hypotheses, opposite outcomes
                        contradiction = {
                            "hypothesis1_id": hyp1.id,
                            "hypothesis2_id": hyp2.id,
                            "similarity": similarity,
                            "hypothesis1_statement": hyp1.statement,
                            "hypothesis2_statement": hyp2.statement,
                            "hypothesis1_supported": support1,
                            "hypothesis2_supported": support2,
                            "detected_at": datetime.utcnow().isoformat(),
                        }

                        contradictions.append(contradiction)
                        logger.warning(
                            f"Contradiction detected between {hyp1.id} and {hyp2.id} "
                            f"(similarity: {similarity:.2f})"
                        )

        return contradictions

    def _compute_semantic_similarity(self, text1: str, text2: str) -> float:
        """
        Compute semantic similarity between two texts.

        Uses vector DB if available, otherwise simple word overlap.

        Args:
            text1: First text
            text2: Second text

        Returns:
            float: Similarity score (0.0 - 1.0)
        """
        if self.vector_db and HAS_CHROMADB:
            try:
                # Use vector DB embeddings
                # (Simplified - actual implementation would query vector DB)
                # For now, use simple word overlap
                words1 = set(text1.lower().split())
                words2 = set(text2.lower().split())

                if not words1 or not words2:
                    return 0.0

                overlap = len(words1 & words2)
                union = len(words1 | words2)

                return overlap / union if union > 0 else 0.0

            except Exception as e:
                logger.warning(f"Error computing similarity with vector DB: {e}")
                return 0.0
        else:
            # Simple word overlap
            words1 = set(text1.lower().split())
            words2 = set(text2.lower().split())

            if not words1 or not words2:
                return 0.0

            overlap = len(words1 & words2)
            union = len(words1 | words2)

            return overlap / union if union > 0 else 0.0

    def _overall_support(self, results: list[ExperimentResult]) -> bool | None:
        """Determine overall support from multiple results."""
        if not results:
            return None

        supports = [r.supports_hypothesis for r in results if r.supports_hypothesis is not None]

        if not supports:
            return None

        # Majority vote
        support_count = sum(1 for s in supports if s is True)
        reject_count = sum(1 for s in supports if s is False)

        if support_count > reject_count:
            return True
        elif reject_count > support_count:
            return False
        else:
            return None  # Tied

    # ========================================================================
    # HYPOTHESIS MERGING
    # ========================================================================

    def merge_hypotheses(
        self, hypotheses: list[Hypothesis], rationale: str = "Merging similar supported hypotheses"
    ) -> Hypothesis:
        """
        Merge similar hypotheses that are both supported.

        Args:
            hypotheses: List of hypotheses to merge (should be 2-3)
            rationale: Reason for merging

        Returns:
            Hypothesis: Merged hypothesis
        """
        logger.info(f"Merging {len(hypotheses)} hypotheses")

        # Build inputs
        statements = "\n".join([f"- {h.statement}" for h in hypotheses])
        rationales = "\n".join([f"- {h.rationale}" for h in hypotheses])

        try:
            with dspy.context(lm=self.llm):
                predictor = dspy.Predict(HypothesisMergingSignature)
                response = predictor(statements=statements, rationales=rationales)

            # Parse response
            merged_statement = (
                response.merged_statement if hasattr(response, "merged_statement") else ""
            )
            merged_rationale = (
                response.merged_rationale if hasattr(response, "merged_rationale") else ""
            )
            synthesis_explanation = (
                response.synthesis_explanation if hasattr(response, "synthesis_explanation") else ""
            )

            # Create merged hypothesis with new ID
            merged = Hypothesis(
                id=f"hyp_{uuid.uuid4().hex[:12]}",
                research_question=hypotheses[0].research_question,
                statement=merged_statement,
                rationale=merged_rationale,
                domain=hypotheses[0].domain,
                status=HypothesisStatus.GENERATED,
                parent_hypothesis_id=hypotheses[0].id,  # First as parent
                generation=max(h.generation for h in hypotheses) + 1,
                evolution_history=[
                    {
                        "action": "merged",
                        "merged_from": [h.id for h in hypotheses],
                        "synthesis": synthesis_explanation,
                        "timestamp": datetime.utcnow().isoformat(),
                    }
                ],
            )

            logger.info("Created merged hypothesis")
            return merged

        except Exception as e:
            logger.error(f"Error merging hypotheses: {e}")
            return hypotheses[0]

    # ========================================================================
    # LINEAGE TRACKING
    # ========================================================================

    def _track_lineage(
        self, hypothesis: Hypothesis, parent: Hypothesis, action: str, evidence: list[str]
    ):
        """Track hypothesis lineage."""
        lineage = HypothesisLineage(
            hypothesis_id=hypothesis.id,
            parent_id=parent.id,
            generation=hypothesis.generation,
            refinement_reason=action,
            evidence_basis=evidence,
        )

        if hypothesis.id:
            self.lineage_tracking[hypothesis.id] = lineage

        # Update parent's children
        if parent.id and parent.id in self.lineage_tracking:
            parent_lineage = self.lineage_tracking[parent.id]
            if hypothesis.id and hypothesis.id not in parent_lineage.children_ids:
                parent_lineage.children_ids.append(hypothesis.id)

    def get_lineage(self, hypothesis_id: str) -> HypothesisLineage | None:
        """Get lineage for a hypothesis."""
        return self.lineage_tracking.get(hypothesis_id)

    def get_family_tree(self, hypothesis_id: str) -> dict[str, Any]:
        """
        Get complete family tree for a hypothesis.

        Returns:
            dict: Family tree with ancestors and descendants
        """
        lineage = self.get_lineage(hypothesis_id)
        if not lineage:
            return {"hypothesis_id": hypothesis_id, "ancestors": [], "descendants": []}

        # Get ancestors
        ancestors = []
        current_id = lineage.parent_id
        while current_id:
            current_lineage = self.get_lineage(current_id)
            if current_lineage:
                ancestors.append(current_id)
                current_id = current_lineage.parent_id
            else:
                break

        # Get descendants (recursive)
        def get_descendants(hyp_id: str) -> list[str]:
            lin = self.get_lineage(hyp_id)
            if not lin or not lin.children_ids:
                return []

            descendants = lin.children_ids.copy()
            for child_id in lin.children_ids:
                descendants.extend(get_descendants(child_id))

            return descendants

        descendants = get_descendants(hypothesis_id)

        return {
            "hypothesis_id": hypothesis_id,
            "generation": lineage.generation,
            "ancestors": ancestors,
            "descendants": descendants,
            "total_family_size": 1 + len(ancestors) + len(descendants),
        }

    # ========================================================================
    # UTILITIES
    # ========================================================================

    def _format_results_for_llm(self, results: list[ExperimentResult]) -> str:
        """Format results for LLM prompt."""
        formatted = []

        for i, result in enumerate(results, 1):
            formatted.append(
                f"Experiment {i}:\n"
                f"  - Supported: {result.supports_hypothesis}\n"
                f"  - P-value: {result.primary_p_value}\n"
                f"  - Effect size: {result.primary_effect_size}\n"
                f"  - Test: {result.primary_test}"
            )

        return "\n\n".join(formatted)
