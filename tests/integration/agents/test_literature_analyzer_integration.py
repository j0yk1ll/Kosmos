"""
Integration tests for kosmos.agents.literature_analyzer module.

These tests require real services (Claude API, etc.) and are marked as integration tests.
"""

import pytest

from kosmos.agents.literature_analyzer import LiteratureAnalyzerAgent, PaperAnalysis


@pytest.mark.integration
@pytest.mark.requires_claude
class TestLiteratureAnalyzerIntegration:
    """Integration tests (requires Claude and services)."""

    def test_real_paper_summarization(self, sample_paper_metadata):
        """Test real paper summarization."""
        agent = LiteratureAnalyzerAgent(config={"use_knowledge_graph": False})

        agent.start()
        analysis = agent.summarize_paper(sample_paper_metadata)
        agent.stop()

        assert isinstance(analysis, PaperAnalysis)
        assert len(analysis.executive_summary) > 0
        assert len(analysis.key_findings) > 0
