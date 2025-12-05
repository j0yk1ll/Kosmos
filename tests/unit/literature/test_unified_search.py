"""
Tests for kosmos.literature.unified_search module.
"""

from unittest.mock import patch

import pytest

from kosmos.literature.base_client import PaperMetadata, PaperSource
from kosmos.literature.unified_search import UnifiedLiteratureSearch


@pytest.fixture
def unified_search():
    """Create UnifiedLiteratureSearch instance."""
    return UnifiedLiteratureSearch()


@pytest.fixture
def sample_papers_from_sources(sample_papers_list):
    """Create papers from different sources."""
    papers = sample_papers_list[:4]
    papers[0].source = "arxiv"
    papers[1].source = "semantic_scholar"
    papers[2].source = "pubmed"
    papers[3].source = "arxiv"  # Duplicate
    papers[3].title = papers[0].title  # Same title as first
    return papers


@pytest.mark.unit
class TestUnifiedSearchInit:
    """Test unified search initialization."""

    def test_init_default(self):
        """Test default initialization."""
        search = UnifiedLiteratureSearch()
        assert PaperSource.ARXIV in search.clients
        assert PaperSource.SEMANTIC_SCHOLAR in search.clients
        assert PaperSource.PUBMED in search.clients

    def test_init_with_custom_sources(self):
        """Test initialization with specific sources."""
        search = UnifiedLiteratureSearch(
            arxiv_enabled=True, semantic_scholar_enabled=True, pubmed_enabled=False
        )
        assert PaperSource.ARXIV in search.clients
        assert PaperSource.SEMANTIC_SCHOLAR in search.clients
        assert PaperSource.PUBMED not in search.clients


@pytest.mark.unit
class TestUnifiedSearch:
    """Test unified search functionality."""

    @patch("kosmos.literature.arxiv_client.ArxivClient.search")
    @patch("kosmos.literature.semantic_scholar.SemanticScholarClient.search")
    @patch("kosmos.literature.pubmed_client.PubMedClient.search")
    def test_search_all_sources(
        self, mock_pubmed, mock_s2, mock_arxiv, unified_search, sample_papers_list
    ):
        """Test searching across all sources."""
        # Mock responses from each source
        mock_arxiv.return_value = [sample_papers_list[0]]
        mock_s2.return_value = [sample_papers_list[1]]
        mock_pubmed.return_value = [sample_papers_list[2]]

        papers = unified_search.search("machine learning", max_results=10)

        assert len(papers) == 3
        assert mock_arxiv.called
        assert mock_s2.called
        assert mock_pubmed.called

    @patch("kosmos.literature.arxiv_client.ArxivClient.search")
    @patch("kosmos.literature.semantic_scholar.SemanticScholarClient.search")
    def test_search_specific_sources(self, mock_s2, mock_arxiv, unified_search, sample_papers_list):
        """Test searching specific sources only."""
        mock_arxiv.return_value = [sample_papers_list[0]]
        mock_s2.return_value = [sample_papers_list[1]]

        papers = unified_search.search(
            "test query", sources=["arxiv", "semantic_scholar"], max_results=10
        )

        assert len(papers) == 2
        assert mock_arxiv.called
        assert mock_s2.called

    @patch("kosmos.literature.arxiv_client.ArxivClient.search")
    @patch("kosmos.literature.semantic_scholar.SemanticScholarClient.search")
    @patch("kosmos.literature.pubmed_client.PubMedClient.search")
    def test_deduplication(self, mock_pubmed, mock_s2, mock_arxiv, unified_search):
        """Test that duplicate papers are removed."""
        # Create duplicate papers with same DOI
        paper1 = PaperMetadata(
            id="arxiv_123",
            source=PaperSource.ARXIV,
            title="Same Paper",
            authors=["Author"],
            abstract="Abstract",
            year=2023,
            doi="10.1234/same",
        )
        paper2 = PaperMetadata(
            id="s2_456",
            source=PaperSource.SEMANTIC_SCHOLAR,
            title="Same Paper",
            authors=["Author"],
            abstract="Abstract",
            year=2023,
            doi="10.1234/same",
        )

        mock_arxiv.return_value = [paper1]
        mock_s2.return_value = [paper2]
        mock_pubmed.return_value = []

        papers = unified_search.search("test", max_results=10)

        # Should only return one paper after deduplication
        assert len(papers) == 1

    @patch("kosmos.literature.arxiv_client.ArxivClient.search")
    def test_search_with_errors(self, mock_arxiv, unified_search):
        """Test handling of search errors."""
        mock_arxiv.side_effect = Exception("API Error")

        # Should return empty list instead of raising
        papers = unified_search.search("test query", sources=["arxiv"])
        assert papers == []


@pytest.mark.unit
class TestUnifiedSearchParallel:
    """Test parallel search functionality."""

    @patch("kosmos.literature.pubmed_client.PubMedClient.search")
    @patch("kosmos.literature.arxiv_client.ArxivClient.search")
    @patch("kosmos.literature.semantic_scholar.SemanticScholarClient.search")
    def test_parallel_execution(
        self, mock_s2, mock_arxiv, mock_pubmed, unified_search, sample_papers_list
    ):
        """Test that searches execute in parallel."""
        mock_arxiv.return_value = [sample_papers_list[0]]
        mock_s2.return_value = [sample_papers_list[1]]
        mock_pubmed.return_value = []

        papers = unified_search.search("test query", max_results=10, parallel=True)

        assert len(papers) == 2
        # All clients should be called
        assert mock_arxiv.called
        assert mock_s2.called
        assert mock_pubmed.called


@pytest.mark.unit
class TestUnifiedSearchDeduplication:
    """Test deduplication strategies."""

    def test_deduplicate_by_doi(self, unified_search):
        """Test deduplication by DOI."""
        papers = [
            PaperMetadata(
                id="arxiv_123",
                source=PaperSource.ARXIV,
                title="Paper 1",
                authors=[],
                abstract="",
                year=2023,
                doi="10.1234/test",
            ),
            PaperMetadata(
                id="s2_456",
                source=PaperSource.SEMANTIC_SCHOLAR,
                title="Paper 1 Duplicate",
                authors=[],
                abstract="",
                year=2023,
                doi="10.1234/test",
            ),
        ]

        deduplicated = unified_search._deduplicate_papers(papers)
        assert len(deduplicated) == 1

    def test_deduplicate_same_source_same_id(self, unified_search):
        """Test deduplication when same ID appears from same source."""
        papers = [
            PaperMetadata(
                id="2301.00001",
                source=PaperSource.ARXIV,
                title="Paper 1",
                authors=[],
                abstract="",
                year=2023,
            ),
            PaperMetadata(
                id="2301.00001",
                source=PaperSource.ARXIV,
                title="Paper 1",
                authors=[],
                abstract="",
                year=2023,
            ),
        ]

        deduplicated = unified_search._deduplicate_papers(papers)
        assert len(deduplicated) == 1

    def test_deduplicate_different_source_same_id(self, unified_search):
        """Test that same ID from different sources are kept separate."""
        papers = [
            PaperMetadata(
                id="2301.00001",
                source=PaperSource.ARXIV,
                title="Paper A",
                authors=[],
                abstract="",
                year=2023,
            ),
            PaperMetadata(
                id="2301.00001",  # Same ID
                source=PaperSource.SEMANTIC_SCHOLAR,  # Different source
                title="Paper B",  # Different title
                authors=[],
                abstract="",
                year=2023,
            ),
        ]

        deduplicated = unified_search._deduplicate_papers(papers)
        assert len(deduplicated) == 2  # Should keep both

    def test_deduplication_priority_order(self, unified_search):
        """Test deduplication priority: DOI > (source, id) > title."""
        # Test 1: DOI takes priority over everything
        papers_doi = [
            PaperMetadata(
                id="arxiv_123",
                source=PaperSource.ARXIV,
                doi="10.1234/paper",
                title="Paper Title",
                authors=[],
                abstract="",
                year=2023,
            ),
            PaperMetadata(
                id="s2_456",
                source=PaperSource.SEMANTIC_SCHOLAR,
                doi="10.1234/paper",  # Same DOI
                title="Different Title",  # Different title
                authors=[],
                abstract="",
                year=2023,
            ),
        ]
        deduplicated_doi = unified_search._deduplicate_papers(papers_doi)
        assert len(deduplicated_doi) == 1, "Papers with same DOI should deduplicate"

        # Test 2: (source, id) takes priority over title when no DOI
        papers_source_id = [
            PaperMetadata(
                id="2301.00001",
                source=PaperSource.ARXIV,
                title="Attention Is All You Need",
                authors=[],
                abstract="",
                year=2023,
            ),
            PaperMetadata(
                id="2301.00001",
                source=PaperSource.ARXIV,
                title="attention is all you need",  # Similar title, same source+id
                authors=[],
                abstract="",
                year=2023,
            ),
        ]
        deduplicated_source = unified_search._deduplicate_papers(papers_source_id)
        assert len(deduplicated_source) == 1, "Papers with same (source, id) should deduplicate"

        # Test 3: Title similarity only applies when DOI and (source, id) differ
        papers_title = [
            PaperMetadata(
                id="arxiv_123",
                source=PaperSource.ARXIV,
                title="Attention Is All You Need",
                authors=[],
                abstract="",
                year=2023,
            ),
            PaperMetadata(
                id="s2_456",
                source=PaperSource.SEMANTIC_SCHOLAR,
                title="Attention Is All You Need",  # Similar title, different source+id
                authors=[],
                abstract="",
                year=2023,
            ),
        ]
        deduplicated_title = unified_search._deduplicate_papers(papers_title)
        assert len(deduplicated_title) == 1, "Papers with similar titles should deduplicate"

        # Test 4: No deduplication when all identifiers differ
        papers_unique = [
            PaperMetadata(
                id="arxiv_123",
                source=PaperSource.ARXIV,
                doi="10.1234/paper1",
                title="First Paper",
                authors=[],
                abstract="",
                year=2023,
            ),
            PaperMetadata(
                id="s2_456",
                source=PaperSource.SEMANTIC_SCHOLAR,
                doi="10.1234/paper2",
                title="Second Paper",
                authors=[],
                abstract="",
                year=2023,
            ),
        ]
        deduplicated_unique = unified_search._deduplicate_papers(papers_unique)
        assert len(deduplicated_unique) == 2, "Completely different papers should not deduplicate"

    def test_deduplicate_by_title_similarity(self, unified_search):
        """Test fuzzy title-based deduplication."""
        papers = [
            PaperMetadata(
                id="1706.03762",
                source=PaperSource.ARXIV,
                title="Attention Is All You Need",
                authors=[],
                abstract="",
                year=2017,
            ),
            PaperMetadata(
                id="s2_attention",
                source=PaperSource.SEMANTIC_SCHOLAR,
                title="Attention is All You Need",
                authors=[],
                abstract="",
                year=2017,
            ),
        ]

        deduplicated = unified_search._deduplicate_papers(papers)
        # Should recognize these as duplicates despite minor differences
        assert len(deduplicated) == 1


@pytest.mark.integration
@pytest.mark.slow
class TestUnifiedSearchIntegration:
    """Integration tests."""

    def test_real_unified_search(self):
        """Test real unified search across sources."""
        search = UnifiedLiteratureSearch()
        papers = search.search("transformer neural network", max_results=5)

        assert len(papers) > 0
        assert all(isinstance(p, PaperMetadata) for p in papers)
        # Should have papers from multiple sources
        sources = {p.source for p in papers}
        assert len(sources) > 1
