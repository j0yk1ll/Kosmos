"""
Tests for kosmos.agents.literature_analyzer module.
"""

from unittest.mock import Mock, patch

import pytest

from kosmos.agents.literature_analyzer import LiteratureAnalyzerAgent, PaperAnalysis
from kosmos.literature.base_client import PaperMetadata


@pytest.fixture
def literature_analyzer(mock_llm_client, mock_knowledge_graph, mock_vector_db, tmp_path):
    """Create LiteratureAnalyzerAgent with mocked dependencies."""
    with (
        patch(
            "kosmos.agents.literature_analyzer.get_knowledge_graph",
            return_value=mock_knowledge_graph,
        ),
        patch("kosmos.agents.literature_analyzer.get_vector_db", return_value=mock_vector_db),
        patch("kosmos.agents.literature_analyzer.get_concept_extractor"),
    ):
        agent = LiteratureAnalyzerAgent(config={"use_knowledge_graph": False})
        agent.llm = mock_llm_client  # Mock the llm attribute
        agent.knowledge_graph = mock_knowledge_graph
        agent.vector_db = mock_vector_db

        agent.cache_dir = tmp_path / "test_cache"
        agent.cache_dir.mkdir(parents=True, exist_ok=True)

        return agent


@pytest.mark.unit
class TestLiteratureAnalyzerInit:
    """Test literature analyzer initialization."""

    def test_init_default(self, mock_llm_client):
        """Test default initialization."""
        with patch("kosmos.agents.literature_analyzer.get_concept_extractor"):
            agent = LiteratureAnalyzerAgent()
            assert agent.agent_type == "LiteratureAnalyzerAgent"

    def test_init_with_config(self, mock_llm_client, mock_knowledge_graph, mock_vector_db):
        """Test initialization with custom config."""
        config = {"use_knowledge_graph": True, "use_semantic_similarity": True}

        with (
            patch("kosmos.agents.literature_analyzer.get_concept_extractor"),
            patch(
                "kosmos.agents.literature_analyzer.get_knowledge_graph",
                return_value=mock_knowledge_graph,
            ),
            patch("kosmos.agents.literature_analyzer.get_vector_db", return_value=mock_vector_db),
        ):
            agent = LiteratureAnalyzerAgent(config=config)

            assert agent.use_knowledge_graph is True
            assert agent.use_semantic_similarity is True


@pytest.mark.unit
class TestPaperSummarization:
    """Test paper summarization."""

    def test_summarize_paper(self, literature_analyzer, sample_paper_metadata):
        """Test summarizing a paper."""
        with patch("kosmos.agents.literature_analyzer.dspy.Predict") as mock_predict:
            mock_response = Mock()
            mock_response.executive_summary = "This paper proposes the Transformer."
            mock_response.key_findings = ["Finding 1", "Finding 2"]
            mock_response.methodology = "Experiments on MT."
            mock_response.significance = "State-of-the-art results."
            mock_response.limitations = ["Computational cost"]
            mock_response.confidence_score = 0.9
            mock_predict.return_value.return_value = mock_response

            analysis = literature_analyzer.summarize_paper(sample_paper_metadata)

            assert isinstance(analysis, PaperAnalysis)
            assert analysis.executive_summary == "This paper proposes the Transformer."
            assert len(analysis.key_findings) == 2
            assert analysis.confidence_score == 0.9

    def test_summarize_paper_with_empty_abstract(self, literature_analyzer):
        """Test summarizing paper with no abstract but with full_text."""
        paper = PaperMetadata(
            id="test-002",
            title="Test Paper",
            authors=[],
            abstract="",
            full_text="Full text content of the paper...",
            year=2023,
            source="test",
        )

        with patch("kosmos.agents.literature_analyzer.dspy.Predict") as mock_predict:
            mock_response = Mock()
            mock_response.executive_summary = "Limited information."
            mock_response.key_findings = []
            mock_response.methodology = "Unknown"
            mock_response.significance = "Cannot assess"
            mock_response.limitations = []
            mock_response.confidence_score = 0.3
            mock_predict.return_value.return_value = mock_response

            analysis = literature_analyzer.summarize_paper(paper)

            assert analysis.confidence_score < 0.5


@pytest.mark.unit
class TestCitationNetworkAnalysis:
    """Test citation network analysis."""

    def test_analyze_citation_network(self, literature_analyzer):
        """Test analyzing citation network."""
        # Enable knowledge graph for this test
        literature_analyzer.use_knowledge_graph = True

        literature_analyzer.knowledge_graph.get_citations.return_value = [
            {"paper": {"id": "cited1", "title": "Cited Paper 1"}},
            {"paper": {"id": "cited2", "title": "Cited Paper 2"}},
        ]
        literature_analyzer.knowledge_graph.get_citing_papers.return_value = [
            {"paper": {"id": "citing1", "title": "Citing Paper 1"}},
        ]

        network_analysis = literature_analyzer.analyze_citation_network(
            "paper_123", depth=1, build_if_missing=False
        )

        assert "citation_count" in network_analysis
        assert "cited_by_count" in network_analysis
        assert network_analysis["citation_count"] == 2
        assert network_analysis["cited_by_count"] == 1

    def test_analyze_citation_network_with_build(self, literature_analyzer):
        """Test building citation network if missing."""
        literature_analyzer.knowledge_graph.get_citations.return_value = []

        network_analysis = literature_analyzer.analyze_citation_network(
            "paper_123", depth=1, build_if_missing=True
        )

        # Should attempt to build citation graph
        assert isinstance(network_analysis, dict)


@pytest.mark.unit
class TestAgentLifecycle:
    """Test agent lifecycle methods."""

    def test_agent_start(self, literature_analyzer):
        """Test starting the agent."""
        literature_analyzer.start()

        assert literature_analyzer.status == "running"

    def test_agent_stop(self, literature_analyzer):
        """Test stopping the agent."""
        literature_analyzer.start()
        literature_analyzer.stop()

        assert literature_analyzer.status == "stopped"

    def test_agent_execute(self, literature_analyzer, sample_paper_metadata):
        """Test agent execute method with message."""
        message = {
            "task_type": "summarize_paper",
            "paper": sample_paper_metadata,
        }

        with patch("kosmos.agents.literature_analyzer.dspy.Predict") as mock_predict:
            mock_response = Mock()
            mock_response.executive_summary = "Summary"
            mock_response.key_findings = []
            mock_response.methodology = "Methods"
            mock_response.significance = "Important"
            mock_response.limitations = []
            mock_response.confidence_score = 0.8
            mock_predict.return_value.return_value = mock_response

            response = literature_analyzer.execute(message)

            assert response is not None
            assert response["status"] == "success"
            assert "summary" in response


@pytest.mark.unit
class TestCorpusAnalysis:
    """Test corpus analysis functionality."""

    def test_analyze_corpus(self, literature_analyzer, sample_papers_list):
        """Test analyzing a corpus of papers."""
        # Test basic corpus analysis
        result = literature_analyzer.analyze_corpus(sample_papers_list[:3], generate_insights=False)

        assert result is not None
        assert "corpus_size" in result
        assert result["corpus_size"] == 3
        assert "temporal_distribution" in result
        assert "common_themes" in result
