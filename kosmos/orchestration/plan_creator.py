"""
Plan Creator Agent for Kosmos.

Generates strategic research plans with exploration/exploitation balance.

Key Innovation: Adaptive strategy based on research cycle progress.

Exploration/Exploitation Ratios by Cycle:
- Early (cycles 1-7): 70% exploration (find new directions)
- Middle (cycles 8-14): 50% balanced
- Late (cycles 15-20): 30% exploration, 70% exploitation (deepen findings)

Performance Target: Generate plans with ~80% approval rate
"""

import logging
from dataclasses import dataclass
from typing import Any

import dspy

from kosmos.config import _DEFAULT_CLAUDE_SONNET_MODEL


logger = logging.getLogger(__name__)


class PlanCreationSignature(dspy.Signature):
    """Generate a set of research tasks and rationale."""

    research_objective: str = dspy.InputField()
    cycle: int = dspy.InputField(desc="Current research cycle number")
    exploration_ratio: float = dspy.InputField()
    recent_findings: list[str] = dspy.InputField()
    unsupported_hypotheses: list[str] = dspy.InputField()
    num_tasks: int = dspy.InputField()

    tasks: list[dict] = dspy.OutputField(desc="Ordered list of research tasks with metadata")
    rationale: str = dspy.OutputField()


@dataclass
class Task:
    """Container for a research task."""

    task_id: int
    task_type: str  # data_analysis, literature_review, hypothesis_generation
    description: str
    expected_output: str
    required_skills: list[str]
    exploration: bool
    target_hypotheses: list[str] | None = None
    priority: int = 1

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "id": self.task_id,
            "type": self.task_type,
            "description": self.description,
            "expected_output": self.expected_output,
            "required_skills": self.required_skills,
            "exploration": self.exploration,
            "target_hypotheses": self.target_hypotheses or [],
            "priority": self.priority,
        }


@dataclass
class ResearchPlan:
    """Container for a research plan (10 tasks + rationale)."""

    cycle: int
    tasks: list[Task]
    rationale: str
    exploration_ratio: float

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "cycle": self.cycle,
            "tasks": [t.to_dict() for t in self.tasks],
            "rationale": self.rationale,
            "exploration_ratio": self.exploration_ratio,
        }


class PlanCreatorAgent:
    """
    Strategic research planning agent.

    Generates 10 tasks per cycle that advance the research objective
    while balancing exploration of new directions with exploitation
    of promising findings.

    Design Philosophy:
    - Early cycles: Broad exploration to map the problem space
    - Middle cycles: Balance between deepening and branching
    - Late cycles: Focus on validating and extending key discoveries
    """

    def __init__(
        self,
        model: str = _DEFAULT_CLAUDE_SONNET_MODEL,
        llm_client: Any | None = None,
        default_num_tasks: int = 10,
    ):
        """
        Initialize Plan Creator Agent.

        Args:
            model: Model to use for plan generation
            default_num_tasks: Default number of tasks per cycle
        """
        self.model = model
        self.llm_client = llm_client
        self.default_num_tasks = default_num_tasks

    def _get_exploration_ratio(self, cycle: int) -> float:
        """
        Determine exploration vs. exploitation ratio.

        Args:
            cycle: Current research cycle

        Returns:
            Exploration ratio (0.0-1.0)
        """
        if cycle <= 7:
            return 0.70  # Early: explore widely
        elif cycle <= 14:
            return 0.50  # Middle: balanced
        else:
            return 0.30  # Late: exploit findings

    async def create_plan(
        self, research_objective: str, context: dict, num_tasks: int | None = None
    ) -> ResearchPlan:
        """
        Generate strategic research plan.

        Args:
            research_objective: Overall research goal
            context: Context from State Manager (findings, hypotheses, etc.)
            num_tasks: Number of tasks to generate (default: self.default_num_tasks)

        Returns:
            ResearchPlan with tasks and rationale
        """
        if num_tasks is None:
            num_tasks = self.default_num_tasks

        cycle = context.get("cycle", 1)
        exploration_ratio = self._get_exploration_ratio(cycle)

        # If no LLM client, use mock planning
        if self.llm_client is None:
            return self._create_mock_plan(
                cycle, research_objective, context, num_tasks, exploration_ratio
            )

        try:
            # Query LLM
            recent_findings = [
                finding.get("summary", "") for finding in context.get("recent_findings", [])
            ]
            unsupported_hypotheses = [
                hyp.get("statement", "") for hyp in context.get("unsupported_hypotheses", [])
            ]

            prediction = self.llm_client.predict(
                PlanCreationSignature,
                research_objective=research_objective,
                cycle=cycle,
                exploration_ratio=exploration_ratio,
                recent_findings=recent_findings,
                unsupported_hypotheses=unsupported_hypotheses,
                num_tasks=num_tasks,
            )
            plan_data = {
                "tasks": getattr(prediction, "tasks", []),
                "rationale": getattr(prediction, "rationale", ""),
            }

            # Validate and create ResearchPlan
            tasks = []
            for i, task_data in enumerate(plan_data.get("tasks", [])[:num_tasks], 1):
                task = Task(
                    task_id=i,
                    task_type=task_data.get("type", "data_analysis"),
                    description=task_data.get("description", ""),
                    expected_output=task_data.get("expected_output", ""),
                    required_skills=task_data.get("required_skills", []),
                    exploration=task_data.get("exploration", False),
                    target_hypotheses=task_data.get("target_hypotheses"),
                    priority=task_data.get("priority", 1),
                )
                tasks.append(task)

            # Ensure we have enough tasks
            while len(tasks) < num_tasks:
                tasks.append(self._create_generic_task(len(tasks) + 1))

            return ResearchPlan(
                cycle=cycle,
                tasks=tasks,
                rationale=plan_data.get("rationale", ""),
                exploration_ratio=exploration_ratio,
            )

        except Exception as e:
            logger.error(f"Plan generation failed: {e}, using mock plan")
            return self._create_mock_plan(
                cycle, research_objective, context, num_tasks, exploration_ratio
            )

    def _create_mock_plan(
        self,
        cycle: int,
        research_objective: str,
        context: dict,
        num_tasks: int,
        exploration_ratio: float,
    ) -> ResearchPlan:
        """Create mock plan for testing (when no LLM available)."""
        tasks = []
        num_exploration = int(num_tasks * exploration_ratio)

        # Task type rotation to ensure structural requirements are met
        # (Plan reviewer requires >= 2 task types, >= 3 data_analysis tasks)
        task_types = ["data_analysis", "literature_review", "hypothesis_generation"]

        # Create exploration tasks (mix of data_analysis and literature_review)
        for i in range(1, num_exploration + 1):
            # Ensure first 3 exploration tasks are data_analysis, then mix in literature_review
            if i <= 3:
                task_type = "data_analysis"
            else:
                task_type = task_types[(i - 1) % 2]  # Alternate data_analysis and literature_review

            tasks.append(
                Task(
                    task_id=i,
                    task_type=task_type,
                    description=f"Exploratory {task_type.replace('_', ' ')} {i} for {research_objective}",
                    expected_output=f"{'Statistical findings and visualizations' if task_type == 'data_analysis' else 'Literature synthesis report'}",
                    required_skills=(
                        ["pandas", "scipy"] if task_type == "data_analysis" else ["arxiv", "pubmed"]
                    ),
                    exploration=True,
                    priority=1,
                )
            )

        # Create exploitation tasks (mix including hypothesis_generation)
        for i in range(num_exploration + 1, num_tasks + 1):
            # Alternate between data_analysis and hypothesis_generation for exploitation
            task_type = (
                "data_analysis" if (i - num_exploration) % 2 == 1 else "hypothesis_generation"
            )

            tasks.append(
                Task(
                    task_id=i,
                    task_type=task_type,
                    description=f"Validation {task_type.replace('_', ' ')} {i} for existing findings",
                    expected_output=(
                        "Hypothesis test results"
                        if task_type == "data_analysis"
                        else "New testable hypotheses"
                    ),
                    required_skills=(
                        ["pandas", "statsmodels"] if task_type == "data_analysis" else []
                    ),
                    exploration=False,
                    priority=2,
                )
            )

        return ResearchPlan(
            cycle=cycle,
            tasks=tasks,
            rationale=f"Mock plan for cycle {cycle} (no LLM client provided)",
            exploration_ratio=exploration_ratio,
        )

    def _create_generic_task(self, task_id: int) -> Task:
        """Create a generic task to fill gaps."""
        return Task(
            task_id=task_id,
            task_type="data_analysis",
            description=f"Additional analysis task {task_id}",
            expected_output="Statistical findings",
            required_skills=["pandas", "scipy"],
            exploration=True,
            priority=3,
        )

    async def revise_plan(
        self, original_plan: ResearchPlan, review_feedback: dict, context: dict
    ) -> ResearchPlan:
        """
        Revise plan based on reviewer feedback.

        Args:
            original_plan: Original plan that was rejected
            review_feedback: Feedback from PlanReviewerAgent
            context: Current context

        Returns:
            Revised ResearchPlan
        """
        # Simple revision: regenerate with feedback in context
        feedback_text = review_feedback.get("feedback", "")
        required_changes = review_feedback.get("required_changes", [])

        # Add feedback to context
        context_with_feedback = context.copy()
        context_with_feedback["previous_plan_feedback"] = feedback_text
        context_with_feedback["required_changes"] = required_changes

        # Regenerate plan
        return await self.create_plan(
            research_objective=context.get("research_objective", ""),
            context=context_with_feedback,
            num_tasks=len(original_plan.tasks),
        )
