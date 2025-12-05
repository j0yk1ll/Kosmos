"""
Integration tests for kosmos.agents.hypothesis_generator module.

These tests require real services (Claude API, database, etc.) and are marked as integration tests.
"""

import pytest

from kosmos.agents.hypothesis_generator import HypothesisGeneratorAgent


@pytest.mark.integration
@pytest.mark.slow
class TestHypothesisGeneratorIntegration:
    """Integration tests (require real LLM and DB)."""

    @pytest.mark.requires_claude
    def test_real_hypothesis_generation(self):
        """Test real hypothesis generation with Claude."""
        agent = HypothesisGeneratorAgent(
            config={"num_hypotheses": 2, "use_literature_context": False}
        )

        response = agent.generate_hypotheses(
            research_question="How does learning rate affect neural network convergence?",
            domain="machine_learning",
            store_in_db=False,
        )

        assert len(response.hypotheses) > 0
        assert response.domain == "machine_learning"

        for hyp in response.hypotheses:
            assert len(hyp.statement) > 15
            assert len(hyp.rationale) > 30
            assert hyp.confidence_score is not None
