"""
Tests for kosmos.knowledge.vector_db module.
"""

from unittest.mock import Mock, patch

import numpy as np
import pytest

from kosmos.knowledge.vector_db import PaperVectorDB
from kosmos.literature.base_client import PaperMetadata, PaperSource


@pytest.fixture
def vector_db():
    """Create PaperVectorDB instance with mocked ChromaDB."""
    with patch("kosmos.knowledge.vector_db.chromadb"):
        with patch("kosmos.knowledge.vector_db.get_embedder") as mock_embedder:
            mock_embedder.return_value = Mock()
            mock_embedder.return_value.embed_papers.return_value = np.array([[0.1] * 768])
            mock_embedder.return_value.embed_query.return_value = np.array([0.1] * 768)
            mock_embedder.return_value.embed_paper.return_value = np.array([0.1] * 768)
            db = PaperVectorDB(persist_directory="/tmp/test_vector_db")
            db.collection = Mock()
            return db


@pytest.mark.unit
class TestPaperVectorDBInit:
    """Test paper vector database initialization."""

    @patch("kosmos.knowledge.vector_db.chromadb")
    @patch("kosmos.knowledge.vector_db.get_embedder")
    def test_init_default(self, mock_embedder, mock_chromadb):
        """Test default initialization."""
        mock_embedder.return_value = Mock()
        db = PaperVectorDB(persist_directory="/tmp/test")
        assert db.collection_name == "papers"

    @patch("kosmos.knowledge.vector_db.chromadb")
    @patch("kosmos.knowledge.vector_db.get_embedder")
    def test_init_custom_collection(self, mock_embedder, mock_chromadb):
        """Test initialization with custom collection name."""
        mock_embedder.return_value = Mock()
        db = PaperVectorDB(collection_name="custom_papers", persist_directory="/tmp/test")
        assert db.collection_name == "custom_papers"


@pytest.mark.unit
class TestPaperVectorDBAdd:
    """Test adding papers to paper vector database."""

    def test_add_paper(self, vector_db, sample_paper_metadata):
        """Test adding a single paper."""
        with patch.object(vector_db, "embedder") as mock_emb:
            mock_emb.embed_papers.return_value = np.array([[0.1] * 768])

            vector_db.add_paper(sample_paper_metadata)

            vector_db.collection.add.assert_called_once()

    def test_add_papers_batch(self, vector_db, sample_papers_list):
        """Test adding multiple papers in batch."""
        with patch.object(vector_db, "embedder") as mock_emb:
            mock_emb.embed_papers.return_value = np.array([[0.1] * 768 for _ in sample_papers_list])

            vector_db.add_papers(sample_papers_list)

            vector_db.collection.add.assert_called()

    def test_add_paper_with_empty_abstract(self, vector_db):
        """Test adding paper with no abstract."""
        paper = PaperMetadata(
            id="test_123",
            source=PaperSource.MANUAL,
            title="Test",
            authors=[],
            abstract="",
            year=2023,
        )

        with patch.object(vector_db, "embedder") as mock_emb:
            mock_emb.embed_papers.return_value = np.array([[0.1] * 768])

            vector_db.add_paper(paper)

            mock_emb.embed_papers.assert_called_once()


@pytest.mark.unit
class TestPaperVectorDBSearch:
    """Test searching in paper vector database."""

    def test_search_by_text(self, vector_db):
        """Test searching by text query."""
        mock_results = {
            "ids": [["id1", "id2"]],
            "distances": [[0.1, 0.2]],
            "metadatas": [
                [
                    {"title": "Paper 1", "paper_id": "id1"},
                    {"title": "Paper 2", "paper_id": "id2"},
                ]
            ],
            "documents": [["doc1", "doc2"]],
        }
        vector_db.collection.query.return_value = mock_results

        with patch.object(vector_db, "embedder") as mock_emb:
            mock_emb.embed_query.return_value = np.array([0.1] * 768)

            results = vector_db.search("test query", top_k=2)

            assert len(results) == 2
            vector_db.collection.query.assert_called_once()

    def test_search_by_paper(self, vector_db, sample_paper_metadata):
        """Test searching by paper (find similar)."""
        mock_results = {
            "ids": [["id1"]],
            "distances": [[0.1]],
            "metadatas": [[{"title": "Similar Paper", "paper_id": "id1"}]],
            "documents": [["doc1"]],
        }
        vector_db.collection.query.return_value = mock_results

        with patch.object(vector_db, "embedder") as mock_emb:
            mock_emb.embed_paper.return_value = np.array([0.1] * 768)

            results = vector_db.search_by_paper(sample_paper_metadata, top_k=5)

            assert len(results) <= 5
            vector_db.collection.query.assert_called_once()

    def test_search_empty_results(self, vector_db):
        """Test search with no results."""
        vector_db.collection.query.return_value = {
            "ids": [[]],
            "distances": [[]],
            "metadatas": [[]],
            "documents": [[]],
        }

        with patch.object(vector_db, "embedder") as mock_emb:
            mock_emb.embed_query.return_value = np.array([0.1] * 768)

            results = vector_db.search("nonexistent query")

            assert results == []


@pytest.mark.unit
class TestPaperVectorDBCRUD:
    """Test CRUD operations."""

    def test_get_paper(self, vector_db):
        """Test getting a specific paper."""
        vector_db.collection.get.return_value = {
            "ids": ["paper_123"],
            "metadatas": [{"title": "Test Paper", "paper_id": "paper_123"}],
            "documents": ["doc1"],
        }

        paper_data = vector_db.get_paper("paper_123")

        assert paper_data is not None
        assert paper_data["id"] == "paper_123"

    def test_delete_paper(self, vector_db):
        """Test deleting a paper."""
        vector_db.delete_paper("paper_123")

        vector_db.collection.delete.assert_called_once_with(ids=["paper_123"])

    def test_count(self, vector_db):
        """Test getting total paper count."""
        vector_db.collection.count.return_value = 42

        paper_count = vector_db.count()

        assert paper_count == 42


@pytest.mark.unit
class TestPaperVectorDBStats:
    """Test database statistics."""

    def test_get_stats(self, vector_db):
        """Test getting database statistics."""
        vector_db.collection.count.return_value = 42
        vector_db.embedder = Mock()
        vector_db.embedder.embedding_dim = 768

        stats = vector_db.get_stats()

        assert stats["collection_name"] == "papers"
        assert stats["paper_count"] == 42
        assert stats["embedding_dim"] == 768


@pytest.mark.integration
@pytest.mark.requires_chromadb
class TestPaperVectorDBIntegration:
    """Integration tests (requires ChromaDB)."""

    def test_real_add_and_search(self, sample_papers_list, tmp_path):
        """Test real add and search operations."""
        db = PaperVectorDB(persist_directory=str(tmp_path / "test_db"))

        # Add papers
        db.add_papers(sample_papers_list[:3])

        # Search
        results = db.search("transformer attention", top_k=2)

        assert len(results) > 0

    def test_real_similarity_search(self, sample_paper_metadata, tmp_path):
        """Test real similarity search."""
        db = PaperVectorDB(persist_directory=str(tmp_path / "test_db"))

        db.add_paper(sample_paper_metadata)

        # Search for similar papers
        results = db.search_by_paper(sample_paper_metadata, top_k=1)

        # Results may be empty if only one paper in DB (self-match excluded)
        assert isinstance(results, list)
