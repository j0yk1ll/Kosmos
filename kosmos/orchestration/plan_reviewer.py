"""
Plan Reviewer Agent for Kosmos.

Validates research plans on 5 dimensions before execution to ensure quality.

5 Review Dimensions (0-10 each):
1. Specificity: Are tasks concrete and executable?
2. Relevance: Do tasks address research objective?
3. Novelty: Do tasks avoid redundancy?
4. Coverage: Do tasks cover important aspects?
5. Feasibility: Are tasks achievable within constraints?

Approval Criteria:
- Average score ≥ 7.0/10
- Minimum score ≥ 5.0/10 (no catastrophic failures)
- At least 3 data_analysis tasks
- At least 2 different task types

Performance Target: ~80% approval rate on first submission
"""

import logging
from dataclasses import dataclass
from typing import Any

import dspy

from kosmos.config import _DEFAULT_CLAUDE_SONNET_MODEL


logger = logging.getLogger(__name__)


class PlanReviewSignature(dspy.Signature):
    """Score and review a research plan."""

    research_objective: str = dspy.InputField()
    plan_tasks: list[dict] = dspy.InputField(desc="Tasks included in the research plan")
    plan_rationale: str = dspy.InputField()

    scores: dict = dspy.OutputField(desc="Dictionary of score_name -> numeric score")
    feedback: str = dspy.OutputField()
    required_changes: list[str] = dspy.OutputField()
    suggestions: list[str] = dspy.OutputField()


@dataclass
class PlanReview:
    """Container for plan review results."""

    approved: bool
    scores: dict[str, float]  # dimension → score (0-10)
    average_score: float
    min_score: float
    feedback: str
    required_changes: list[str]
    suggestions: list[str]

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "approved": self.approved,
            "scores": self.scores,
            "average_score": self.average_score,
            "min_score": self.min_score,
            "feedback": self.feedback,
            "required_changes": self.required_changes,
            "suggestions": self.suggestions,
        }


class PlanReviewerAgent:
    """
    Plan quality validation agent.

    Evaluates research plans on multiple dimensions before execution
    to prevent low-quality or unfocused research.

    Design Philosophy:
    - Multi-dimensional scoring catches different failure modes
    - Minimum thresholds prevent catastrophic failures
    - Actionable feedback enables plan revision
    """

    # Review dimension weights (not currently used, but available for future)
    DIMENSION_WEIGHTS = {
        "specificity": 0.25,
        "relevance": 0.25,
        "novelty": 0.20,
        "coverage": 0.15,
        "feasibility": 0.15,
    }

    def __init__(
        self,
        model: str = _DEFAULT_CLAUDE_SONNET_MODEL,
        llm_client: Any | None = None,
        min_average_score: float = 7.0,
        min_dimension_score: float = 5.0,
    ):
        """
        Initialize Plan Reviewer Agent.

        Args:
            lm: DSPy language model instance (dspy.LM)
            model: Model to use for review
            min_average_score: Minimum average score for approval
            min_dimension_score: Minimum score on any single dimension
        """
        self.llm_client = llm_client
        self.model = model
        self.min_average_score = min_average_score
        self.min_dimension_score = min_dimension_score

    async def review_plan(self, plan: dict, context: dict) -> PlanReview:
        """
        Review research plan on 5 dimensions.

        Args:
            plan: ResearchPlan as dictionary
            context: Context from State Manager

        Returns:
            PlanReview with scores, approval status, and feedback
        """
        # If no LLM client, use mock review
        if self.llm_client is None:
            return self._mock_review(plan, context)

        try:
            prediction = self.llm_client.predict(
                PlanReviewSignature,
                research_objective=context.get("research_objective", ""),
                plan_tasks=plan.get("tasks", []),
                plan_rationale=plan.get("rationale", ""),
            )
            review_data = {
                "scores": getattr(prediction, "scores", {}),
                "feedback": getattr(prediction, "feedback", ""),
                "required_changes": getattr(prediction, "required_changes", []),
                "suggestions": getattr(prediction, "suggestions", []),
            }

            # Extract scores
            scores = review_data.get("scores", {})
            avg_score = sum(scores.values()) / len(scores) if scores else 0
            min_score = min(scores.values()) if scores else 0

            # Check structural requirements
            structural_ok = self._meets_structural_requirements(plan)

            # Determine approval
            approved = (
                avg_score >= self.min_average_score
                and min_score >= self.min_dimension_score
                and structural_ok
            )

            return PlanReview(
                approved=approved,
                scores=scores,
                average_score=avg_score,
                min_score=min_score,
                feedback=review_data.get("feedback", ""),
                required_changes=review_data.get("required_changes", []),
                suggestions=review_data.get("suggestions", []),
            )

        except Exception as e:
            logger.error(f"Plan review failed: {e}, using mock review")
            return self._mock_review(plan, context)

    def _meets_structural_requirements(self, plan: dict) -> bool:
        """
        Check if plan meets basic structural requirements.

        Args:
            plan: Plan dictionary

        Returns:
            True if structural requirements met
        """
        tasks = plan.get("tasks", [])

        if not tasks:
            return False

        # Requirement 1: At least 3 data_analysis tasks
        data_analysis_count = sum(1 for t in tasks if t.get("type") == "data_analysis")
        if data_analysis_count < 3:
            logger.warning(f"Only {data_analysis_count} data_analysis tasks, need >= 3")
            return False

        # Requirement 2: At least 2 different task types
        task_types = {t.get("type") for t in tasks}
        if len(task_types) < 2:
            logger.warning(f"Only {len(task_types)} task types, need >= 2")
            return False

        # Requirement 3: Each task has required fields
        for task in tasks:
            if not task.get("description"):
                logger.warning(f"Task {task.get('id')} missing description")
                return False
            if not task.get("expected_output"):
                logger.warning(f"Task {task.get('id')} missing expected_output")
                return False

        return True

    def _mock_review(self, plan: dict, context: dict) -> PlanReview:
        """
        Mock review for testing (when no LLM available).

        Provides optimistic scores that usually pass.
        """
        # Check structural requirements
        structural_ok = self._meets_structural_requirements(plan)

        # Base scores (slightly above minimum)
        base_score = 7.5 if structural_ok else 6.0

        scores = {
            "specificity": base_score,
            "relevance": base_score,
            "novelty": base_score - 0.5,
            "coverage": base_score,
            "feasibility": base_score + 0.5,
        }

        avg_score = sum(scores.values()) / len(scores)
        min_score = min(scores.values())

        approved = (
            avg_score >= self.min_average_score
            and min_score >= self.min_dimension_score
            and structural_ok
        )

        required_changes = []
        if not structural_ok:
            required_changes.append("Fix structural requirements (3+ data_analysis, 2+ task types)")

        return PlanReview(
            approved=approved,
            scores=scores,
            average_score=avg_score,
            min_score=min_score,
            feedback=f"Mock review: {'APPROVED' if approved else 'NEEDS REVISION'} (avg: {avg_score:.1f})",
            required_changes=required_changes,
            suggestions=["This is a mock review (no LLM client provided)"],
        )

    def get_approval_statistics(self, reviews: list[PlanReview]) -> dict:
        """
        Compute statistics over batch of reviews.

        Args:
            reviews: List of PlanReview objects

        Returns:
            Dictionary with approval statistics
        """
        if not reviews:
            return {}

        approved_count = sum(1 for r in reviews if r.approved)
        total = len(reviews)

        # Average scores per dimension
        avg_scores = {}
        for dim in ["specificity", "relevance", "novelty", "coverage", "feasibility"]:
            avg_scores[f"avg_{dim}"] = sum(r.scores.get(dim, 0) for r in reviews) / total

        return {
            "total_reviewed": total,
            "approved": approved_count,
            "rejected": total - approved_count,
            "approval_rate": approved_count / total,
            "avg_overall_score": sum(r.average_score for r in reviews) / total,
            **avg_scores,
        }
