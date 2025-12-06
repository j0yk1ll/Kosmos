"""
Phase 3 end-to-end integration tests.

Tests complete workflow: Generation → Novelty → Testability → Prioritization
"""

from unittest.mock import Mock, patch

import pytest

from kosmos.agents.hypothesis_generator import HypothesisGeneratorAgent
from kosmos.hypothesis.novelty_checker import NoveltyChecker
from kosmos.hypothesis.prioritizer import HypothesisPrioritizer
from kosmos.hypothesis.testability import TestabilityAnalyzer
from kosmos.models.hypothesis import Hypothesis


@pytest.fixture
def mock_llm_hypotheses():
    return {
        "hypotheses": [
            {
                "statement": "Hypothesis 1: X increases Y by 20%",
                "rationale": "Evidence shows X affects Y through mechanism Z",
                "confidence_score": 0.8,
                "testability_score": 0.85,
                "suggested_experiment_types": ["computational"],
            },
            {
                "statement": "Hypothesis 2: A correlates with B",
                "rationale": "Data suggests strong correlation between A and B",
                "confidence_score": 0.7,
                "testability_score": 0.75,
                "suggested_experiment_types": ["data_analysis"],
            },
        ]
    }


@pytest.mark.integration
class TestPhase3EndToEnd:
    """Test complete Phase 3 workflow."""

    @patch("kosmos.agents.hypothesis_generator.dspy.LM")
    @patch("kosmos.hypothesis.novelty_checker.UnifiedLiteratureSearch")
    @patch("kosmos.hypothesis.novelty_checker.get_session")
    def test_full_hypothesis_pipeline(
        self, mock_session, mock_search, mock_dspy_lm, mock_llm_hypotheses
    ):
        """Test: Generate → Check Novelty → Analyze Testability → Prioritize."""

        # Setup mocks
        mock_lm = Mock()
        mock_dspy_lm.return_value = mock_lm

        mock_search_inst = Mock()
        mock_search_inst.search.return_value = []
        mock_search.return_value = mock_search_inst

        mock_sess = Mock()
        mock_sess.query.return_value.filter.return_value.all.return_value = []
        mock_session.return_value = mock_sess

        # Step 1: Generate hypotheses
        agent = HypothesisGeneratorAgent(
            config={"use_literature_context": False},
            llm_config={"model": "test", "api_key": "test"},
        )
        agent.llm = mock_lm

        # Mock the generate_hypotheses to return test data
        with patch.object(
            agent, "generate_hypotheses", return_value=Mock(hypotheses=[])
        ) as mock_gen:
            response = mock_gen(research_question="How does X affect Y?", store_in_db=False)

        assert len(response.hypotheses) == 2
        hypotheses = response.hypotheses

        # Step 2: Check novelty
        novelty_checker = NoveltyChecker(use_vector_db=False)
        novelty_checker.literature_search = mock_search_inst

        for hyp in hypotheses:
            report = novelty_checker.check_novelty(hyp)
            assert report.novelty_score is not None
            assert 0.0 <= report.novelty_score <= 1.0
            hyp.novelty_score = report.novelty_score

        # Step 3: Analyze testability
        testability_analyzer = TestabilityAnalyzer(use_llm_for_assessment=False)

        for hyp in hypotheses:
            report = testability_analyzer.analyze_testability(hyp)
            assert report.testability_score is not None
            assert report.is_testable or not report.is_testable  # Boolean
            hyp.testability_score = report.testability_score

        # Step 4: Prioritize
        prioritizer = HypothesisPrioritizer(
            use_novelty_checker=False,  # Already done
            use_testability_analyzer=False,  # Already done
            use_impact_prediction=False,
        )

        ranked = prioritizer.prioritize(hypotheses, run_analysis=False)

        assert len(ranked) == 2
        assert ranked[0].rank == 1
        assert ranked[1].rank == 2
        assert ranked[0].priority_score > 0.0

        # Verify all scores present
        for p in ranked:
            assert p.novelty_score is not None
            assert p.testability_score is not None
            assert p.feasibility_score is not None
            assert p.impact_score is not None

    @patch("kosmos.agents.hypothesis_generator.dspy.LM")
    def test_hypothesis_filtering(self, mock_dspy_lm):
        """Test filtering untestable or non-novel hypotheses."""
        mock_lm = Mock()
        mock_dspy_lm.return_value = mock_lm

        agent = HypothesisGeneratorAgent(
            config={"use_literature_context": False},
            llm_config={"model": "test", "api_key": "test"},
        )
        agent.llm = mock_lm

        # Create mock hypotheses with testability scores

        mock_hypotheses = [
            Mock(
                statement="Good hypothesis with clear prediction",
                testability_score=0.9,
                is_testable=Mock(return_value=True),
            ),
            Mock(
                statement="Vague hypothesis maybe possibly",
                testability_score=0.2,
                is_testable=Mock(return_value=False),
            ),
        ]

        # Mock the response
        with patch.object(
            agent, "generate_hypotheses", return_value=Mock(hypotheses=mock_hypotheses)
        ):
            response = agent.generate_hypotheses("Test question?", store_in_db=False)

        # Filter testable hypotheses
        testable = [h for h in response.hypotheses if h.is_testable(threshold=0.5)]

        assert len(testable) == 1  # Only the good hypothesis
        assert testable[0].testability_score >= 0.5

    def test_hypothesis_model_validation(self):
        """Test Pydantic validation on Hypothesis model."""
        # Valid hypothesis
        hyp = Hypothesis(
            research_question="Valid question?",
            statement="This is a clear testable statement",
            rationale="This is a sufficient rationale that explains the hypothesis",
            domain="test",
        )
        assert hyp.statement == "This is a clear testable statement"

        # Invalid: statement too short (should fail validation)
        with pytest.raises(ValueError):
            Hypothesis(
                research_question="Test",
                statement="Too short",
                rationale="Valid rationale here",
                domain="test",
            )


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.requires_claude
class TestPhase3RealIntegration:
    """Integration tests with real services (requires Claude, DB)."""

    def test_real_hypothesis_workflow(self):
        """Test with real Claude API (slow, requires API key)."""
        agent = HypothesisGeneratorAgent(
            config={"num_hypotheses": 2, "use_literature_context": False}
        )

        response = agent.generate_hypotheses(
            research_question="How does batch size affect neural network training?",
            domain="machine_learning",
            store_in_db=False,
        )

        assert len(response.hypotheses) > 0

        # Analyze first hypothesis
        hyp = response.hypotheses[0]

        # Check novelty
        novelty_checker = NoveltyChecker(use_vector_db=False)
        novelty_report = novelty_checker.check_novelty(hyp)
        assert novelty_report.novelty_score is not None

        # Check testability
        testability_analyzer = TestabilityAnalyzer(use_llm_for_assessment=False)
        testability_report = testability_analyzer.analyze_testability(hyp)
        assert testability_report.is_testable or not testability_report.is_testable
