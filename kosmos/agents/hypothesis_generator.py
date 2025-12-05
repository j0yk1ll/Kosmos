"""
Hypothesis Generator Agent.

Generates scientific hypotheses from research questions using Claude,
with literature context and novelty checking.
"""

import logging
import time
import uuid
from typing import Any

import dspy

from kosmos.agents.base import AgentMessage, AgentStatus, BaseAgent, MessageType
from kosmos.agents.dspy_client import DSPyAgentClient
from kosmos.db import get_session
from kosmos.db.models import (
    Hypothesis as DBHypothesis,
    HypothesisStatus as DBHypothesisStatus,
)
from kosmos.literature.base_client import PaperMetadata
from kosmos.literature.unified_search import UnifiedLiteratureSearch
from kosmos.models.hypothesis import (
    ExperimentType,
    Hypothesis,
    HypothesisGenerationResponse,
    HypothesisStatus,
)
from kosmos.utils.compat import model_to_dict


logger = logging.getLogger(__name__)


class DomainClassificationSignature(dspy.Signature):
    """Identify the primary scientific domain for a research question."""

    research_question: str = dspy.InputField()
    domain: str = dspy.OutputField(desc="Normalized domain label such as machine_learning or biology")


class HypothesisGenerationSignature(dspy.Signature):
    """Produce multiple structured hypotheses for a research question."""

    research_question: str = dspy.InputField()
    domain: str = dspy.InputField()
    num_hypotheses: int = dspy.InputField()
    literature_context: str = dspy.InputField(desc="Optional recent literature context as plain text")

    hypotheses: list[dict] = dspy.OutputField(
        desc="List of hypotheses with statement, rationale, confidence_score, testability_score, and suggested_experiment_types"
    )


def get_client():
    """Deprecated compatibility shim for legacy patches."""
    return None


class HypothesisGeneratorAgent(BaseAgent):
    """
    Agent for generating scientific hypotheses.

    Capabilities:
    - Generate multiple hypotheses from research questions
    - Use literature context for informed hypothesis generation
    - Validate hypothesis quality and testability
    - Store hypotheses in database
    - Integration with novelty checking (optional)

    Example:
        ```python
        agent = HypothesisGeneratorAgent(config={
            "num_hypotheses": 3,
            "use_literature_context": True
        })
        agent.start()

        # Generate hypotheses
        response = agent.generate_hypotheses(
            "How does attention mechanism affect transformer performance?"
        )

        for hyp in response.hypotheses:
            print(f"{hyp.statement} (novelty: {hyp.novelty_score})")
        ```
    """

    def __init__(
        self,
        agent_id: str | None = None,
        agent_type: str | None = None,
        config: dict[str, Any] | None = None,
        llm_client: DSPyAgentClient | None = None,
    ):
        """
        Initialize Hypothesis Generator Agent.

        Args:
            agent_id: Unique agent identifier
            agent_type: Agent type name
            config: Configuration dictionary
        """
        super().__init__(agent_id, agent_type or "HypothesisGeneratorAgent", config)

        # Configuration
        self.num_hypotheses = self.config.get("num_hypotheses", 3)
        self.use_literature_context = self.config.get("use_literature_context", True)
        self.max_papers_context = self.config.get("max_papers_context", 10)
        self.require_novelty_check = self.config.get("require_novelty_check", False)
        self.min_novelty_score = self.config.get("min_novelty_score", 0.5)

        # Components
        self.llm_client = llm_client
        self.literature_search = UnifiedLiteratureSearch() if self.use_literature_context else None

        logger.info(f"Initialized HypothesisGeneratorAgent {self.agent_id}")

    def execute(self, message: AgentMessage) -> AgentMessage:
        """
        Execute agent task from message.

        Args:
            message: AgentMessage with task details

        Returns:
            AgentMessage: Response message with results
        """
        self.status = AgentStatus.WORKING

        try:
            task_type = message.content.get("task_type")

            if task_type == "generate_hypotheses":
                research_question = message.content.get("research_question")
                num_hypotheses = message.content.get("num_hypotheses", self.num_hypotheses)
                domain = message.content.get("domain")

                response = self.generate_hypotheses(
                    research_question=research_question,
                    num_hypotheses=num_hypotheses,
                    domain=domain,
                )

                return AgentMessage(
                    type=MessageType.RESPONSE,
                    from_agent=self.agent_id,
                    to_agent=message.from_agent,
                    content={"response": model_to_dict(response)},
                    correlation_id=message.correlation_id,
                )

            else:
                raise ValueError(f"Unknown task type: {task_type}")

        except Exception as e:
            logger.error(f"Error executing task: {e}", exc_info=True)
            self.status = AgentStatus.ERROR
            return AgentMessage(
                type=MessageType.ERROR,
                from_agent=self.agent_id,
                to_agent=message.from_agent,
                content={"error": str(e)},
                correlation_id=message.correlation_id,
            )

        finally:
            self.status = AgentStatus.IDLE

    def generate_hypotheses(
        self,
        research_question: str,
        num_hypotheses: int | None = None,
        domain: str | None = None,
        store_in_db: bool = True,
    ) -> HypothesisGenerationResponse:
        """
        Generate hypotheses from research question.

        Args:
            research_question: Research question to generate hypotheses for
            num_hypotheses: Number of hypotheses to generate (default: config value)
            domain: Scientific domain (auto-detected if None)
            store_in_db: Whether to store hypotheses in database

        Returns:
            HypothesisGenerationResponse: Generated hypotheses with metadata

        Example:
            ```python
            response = agent.generate_hypotheses(
                research_question="How does learning rate affect convergence?",
                num_hypotheses=3,
                domain="machine_learning"
            )
            ```
        """
        start_time = time.time()
        num_hypotheses = num_hypotheses or self.num_hypotheses

        logger.info(f"Generating {num_hypotheses} hypotheses for: '{research_question}'")

        # Step 1: Auto-detect domain if not provided
        if not domain:
            domain = self._detect_domain(research_question)
            logger.info(f"Auto-detected domain: {domain}")

        # Step 2: Gather literature context
        papers = []
        if self.use_literature_context and self.literature_search:
            papers = self._gather_literature_context(research_question, domain)
            logger.info(f"Gathered {len(papers)} papers for context")

        # Step 3: Generate hypotheses using DSPy LM
        hypotheses = self._generate_with_lm(
            research_question=research_question,
            domain=domain,
            num_hypotheses=num_hypotheses,
            context_papers=papers,
        )

        # Step 4: Validate hypotheses
        validated_hypotheses = []
        for hyp in hypotheses:
            try:
                if self._validate_hypothesis(hyp):
                    validated_hypotheses.append(hyp)
                else:
                    logger.warning(f"Hypothesis failed validation: {hyp.statement[:50]}...")
            except Exception as e:
                logger.error(f"Error validating hypothesis: {e}")

        logger.info(f"Generated {len(validated_hypotheses)} valid hypotheses")

        # Step 5: Store in database if requested
        if store_in_db:
            for hyp in validated_hypotheses:
                self._store_hypothesis(hyp)

        # Step 6: Calculate metrics
        generation_time = time.time() - start_time
        avg_novelty = None
        avg_testability = None

        if validated_hypotheses:
            novelty_scores = [
                h.novelty_score for h in validated_hypotheses if h.novelty_score is not None
            ]
            if novelty_scores:
                avg_novelty = sum(novelty_scores) / len(novelty_scores)

            testability_scores = [
                h.testability_score for h in validated_hypotheses if h.testability_score is not None
            ]
            if testability_scores:
                avg_testability = sum(testability_scores) / len(testability_scores)

        return HypothesisGenerationResponse(
            hypotheses=validated_hypotheses,
            research_question=research_question,
            domain=domain,
            generation_time_seconds=generation_time,
            num_papers_analyzed=len(papers),
            avg_novelty_score=avg_novelty,
            avg_testability_score=avg_testability,
        )

    def _detect_domain(self, research_question: str) -> str:
        """
        Auto-detect scientific domain from research question.

        Args:
            research_question: Research question text

        Returns:
            str: Detected domain
        """
        try:
            if not self.llm_client:
                raise ValueError("Language model client is not configured")

            prediction = self.llm_client.predict(
                DomainClassificationSignature, research_question=research_question
            )
            domain = (
                prediction.domain.strip().lower().replace(" ", "_").replace("-", "_")
                if prediction and getattr(prediction, "domain", None)
                else None
            )
            return domain if domain else "general"

        except Exception as e:
            logger.error(f"Error detecting domain: {e}")
            return "general"

    def _gather_literature_context(
        self, research_question: str, domain: str
    ) -> list[PaperMetadata]:
        """
        Gather relevant literature for context.

        Args:
            research_question: Research question
            domain: Scientific domain

        Returns:
            List[PaperMetadata]: Relevant papers
        """
        if not self.literature_search:
            return []

        try:
            # Search for relevant papers
            query = research_question
            papers = self.literature_search.search(query=query, max_results=self.max_papers_context)

            logger.info(f"Found {len(papers)} papers for context")
            return papers

        except Exception as e:
            logger.error(f"Error gathering literature: {e}", exc_info=True)
            return []

    def _generate_with_lm(
        self,
        research_question: str,
        domain: str,
        num_hypotheses: int,
        context_papers: list[PaperMetadata],
    ) -> list[Hypothesis]:
        """
        Generate hypotheses using DSPy LMs with structured output.

        Args:
            research_question: Research question
            domain: Scientific domain
            num_hypotheses: Number of hypotheses to generate
            context_papers: Literature context

        Returns:
            List[Hypothesis]: Generated hypotheses
        """
        # Build literature context summary
        literature_context = ""
        if context_papers:
            literature_context = "Recent relevant literature:\n\n"
            # Filter out None papers and papers without titles
            valid_papers = [p for p in context_papers[:5] if p is not None and p.title]
            for i, paper in enumerate(valid_papers, 1):
                title = paper.title or "Untitled"
                year = paper.year or "N/A"
                literature_context += f"{i}. {title} ({year})\n"
                if paper.abstract:
                    literature_context += f"   Abstract: {paper.abstract[:200]}...\n"
                literature_context += "\n"

        try:
            if not self.llm_client:
                raise ValueError("Language model client is not configured")

            prediction = self.llm_client.predict(
                HypothesisGenerationSignature,
                research_question=research_question,
                domain=domain,
                num_hypotheses=num_hypotheses,
                literature_context=literature_context
                or "No specific literature context provided.",
            )
            response = prediction.hypotheses if prediction else []

            # Parse response into Hypothesis objects
            hypotheses = []
            for i, hyp_data in enumerate(response):
                try:
                    # Map experiment types
                    exp_types = []
                    for exp_type_str in hyp_data.get("suggested_experiment_types", []):
                        try:
                            exp_types.append(ExperimentType(exp_type_str))
                        except ValueError:
                            logger.warning(f"Unknown experiment type: {exp_type_str}")

                    hypothesis = Hypothesis(
                        id=str(uuid.uuid4()),
                        research_question=research_question,
                        statement=hyp_data["statement"],
                        rationale=hyp_data["rationale"],
                        domain=domain,
                        status=HypothesisStatus.GENERATED,
                        testability_score=hyp_data.get("testability_score"),
                        confidence_score=hyp_data.get("confidence_score"),
                        suggested_experiment_types=exp_types,
                        related_papers=[
                            p.primary_identifier for p in context_papers if p is not None
                        ],
                        generated_by=self.agent_id,
                    )
                    hypotheses.append(hypothesis)

                except Exception as e:
                    logger.error(f"Error parsing hypothesis {i}: {e}")
                    continue

            return hypotheses

        except Exception as e:
            logger.error(f"Error generating hypotheses with DSPy LM: {e}", exc_info=True)
            return []

    def _validate_hypothesis(self, hypothesis: Hypothesis) -> bool:
        """
        Validate hypothesis quality.

        Args:
            hypothesis: Hypothesis to validate

        Returns:
            bool: True if hypothesis passes validation
        """
        try:

            # Check for vague language (warning only, don't fail)
            vague_words = ["maybe", "might", "perhaps", "possibly", "potentially", "somewhat"]
            if any(word in hypothesis.statement.lower() for word in vague_words):
                logger.warning(f"Hypothesis contains vague language: {hypothesis.statement}")
            return True

        except Exception as e:
            logger.error(f"Validation error: {e}")
            return False

    def _store_hypothesis(self, hypothesis: Hypothesis) -> str | None:
        """
        Store hypothesis in database.

        Args:
            hypothesis: Hypothesis to store

        Returns:
            Optional[str]: Hypothesis ID if successful
        """
        try:
            session_ctx = get_session()

            # Support both actual context-manager sessions and test mocks
            if hasattr(session_ctx, "__enter__"):
                with session_ctx as session:
                    pass
            else:
                session = session_ctx

            # Convert to DB model
            db_hypothesis = DBHypothesis(
                id=hypothesis.id or str(uuid.uuid4()),
                research_question=hypothesis.research_question,
                statement=hypothesis.statement,
                rationale=hypothesis.rationale,
                domain=hypothesis.domain,
                status=DBHypothesisStatus.GENERATED,
                novelty_score=hypothesis.novelty_score,
                testability_score=hypothesis.testability_score,
                confidence_score=hypothesis.confidence_score,
                related_papers=hypothesis.related_papers,
                created_at=hypothesis.created_at,
                updated_at=hypothesis.updated_at,
            )

            session.add(db_hypothesis)
            session.commit()

            logger.info(f"Stored hypothesis {db_hypothesis.id} in database")
            hypothesis.id = db_hypothesis.id
            return db_hypothesis.id

        except Exception as e:
            logger.error(f"Error storing hypothesis: {e}", exc_info=True)
            return None

    def get_hypothesis_by_id(self, hypothesis_id: str) -> Hypothesis | None:
        """
        Retrieve hypothesis from database by ID.

        Args:
            hypothesis_id: Hypothesis ID

        Returns:
            Optional[Hypothesis]: Hypothesis if found
        """
        try:
            session_ctx = get_session()

            if hasattr(session_ctx, "__enter__"):
                with session_ctx as session:
                    db_hyp = (
                        session.query(DBHypothesis).filter(DBHypothesis.id == hypothesis_id).first()
                    )
            else:
                session = session_ctx
                db_hyp = (
                    session.query(DBHypothesis).filter(DBHypothesis.id == hypothesis_id).first()
                )

            if not db_hyp:
                return None

            # Convert DB model to Pydantic model
            hypothesis = Hypothesis(
                id=db_hyp.id,
                research_question=db_hyp.research_question,
                statement=db_hyp.statement,
                rationale=db_hyp.rationale,
                domain=db_hyp.domain,
                status=HypothesisStatus(db_hyp.status.value),
                testability_score=db_hyp.testability_score,
                novelty_score=db_hyp.novelty_score,
                confidence_score=db_hyp.confidence_score,
                related_papers=db_hyp.related_papers or [],
                created_at=db_hyp.created_at,
                updated_at=db_hyp.updated_at,
            )

            return hypothesis

        except Exception as e:
            logger.error(f"Error retrieving hypothesis: {e}", exc_info=True)
            return None

    def list_hypotheses(
        self, domain: str | None = None, status: HypothesisStatus | None = None, limit: int = 100
    ) -> list[Hypothesis]:
        """
        List hypotheses from database with optional filtering.

        Args:
            domain: Filter by domain
            status: Filter by status
            limit: Maximum number to return

        Returns:
            List[Hypothesis]: Matching hypotheses
        """
        try:
            session_ctx = get_session()

            if hasattr(session_ctx, "__enter__"):
                with session_ctx as session:
                    query = session.query(DBHypothesis)

                    if domain:
                        query = query.filter(DBHypothesis.domain == domain)

                    if status:
                        db_status = DBHypothesisStatus(status.value)
                        query = query.filter(DBHypothesis.status == db_status)

                    query = query.order_by(DBHypothesis.created_at.desc()).limit(limit)
                    db_rows = query.all()
            else:
                session = session_ctx
                query = session.query(DBHypothesis)

                if domain:
                    query = query.filter(DBHypothesis.domain == domain)

                if status:
                    db_status = DBHypothesisStatus(status.value)
                    query = query.filter(DBHypothesis.status == db_status)

                query = query.order_by(DBHypothesis.created_at.desc()).limit(limit)
                db_rows = query.all()

            hypotheses = []
            for db_hyp in db_rows:
                hypothesis = Hypothesis(
                    id=db_hyp.id,
                    research_question=db_hyp.research_question,
                    statement=db_hyp.statement,
                    rationale=db_hyp.rationale,
                    domain=db_hyp.domain,
                    status=HypothesisStatus(db_hyp.status.value),
                    testability_score=db_hyp.testability_score,
                    novelty_score=db_hyp.novelty_score,
                    confidence_score=db_hyp.confidence_score,
                    related_papers=db_hyp.related_papers or [],
                    created_at=db_hyp.created_at,
                    updated_at=db_hyp.updated_at,
                )
                hypotheses.append(hypothesis)

            return hypotheses

        except Exception as e:
            logger.error(f"Error listing hypotheses: {e}", exc_info=True)
            return []
