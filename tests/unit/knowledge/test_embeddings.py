"""
Tests for kosmos.knowledge.embeddings module.
"""

from unittest.mock import patch

import numpy as np
import pytest

from kosmos.knowledge.embeddings import PaperEmbedder


@pytest.fixture
def paper_embedder():
    """Create PaperEmbedder instance."""
    with patch("kosmos.knowledge.embeddings.SentenceTransformer") as mock_st:
        mock_st.return_value.get_sentence_embedding_dimension.return_value = 768
        mock_st.return_value.device = "cpu"
        embedder = PaperEmbedder(model_name="allenai/specter")
        return embedder


@pytest.mark.unit
class TestPaperEmbedderInit:
    """Test paper embedder initialization."""

    @patch("kosmos.knowledge.embeddings.SentenceTransformer")
    def test_init_default(self, mock_st):
        """Test default initialization."""
        mock_st.return_value.get_sentence_embedding_dimension.return_value = 768
        mock_st.return_value.device = "cpu"
        embedder = PaperEmbedder()
        assert embedder.model_name == "allenai/specter"
        mock_st.assert_called_once()

    @patch("kosmos.knowledge.embeddings.SentenceTransformer")
    def test_init_custom_model(self, mock_st):
        """Test initialization with custom model."""
        mock_st.return_value.get_sentence_embedding_dimension.return_value = 768
        mock_st.return_value.device = "cpu"
        embedder = PaperEmbedder(model_name="custom-model")
        assert embedder.model_name == "custom-model"


@pytest.mark.unit
class TestEmbeddingGeneration:
    """Test embedding generation."""

    def test_embed_query(self, paper_embedder):
        """Test embedding a query."""
        with patch.object(paper_embedder.model, "encode") as mock_encode:
            mock_encode.return_value = np.array([0.1, 0.2, 0.3])

            embedding = paper_embedder.embed_query("test query")

            assert isinstance(embedding, np.ndarray)
            assert len(embedding) == 3
            mock_encode.assert_called_once()

    def test_embed_paper(self, paper_embedder, sample_paper_metadata):
        """Test embedding a paper."""
        with patch.object(paper_embedder.model, "encode") as mock_encode:
            mock_encode.return_value = np.array([0.1] * 768)

            embedding = paper_embedder.embed_paper(sample_paper_metadata)

            assert isinstance(embedding, np.ndarray)
            assert len(embedding) == 768
            # Should combine title and abstract
            mock_encode.assert_called_once()

    def test_embed_papers_batch(self, paper_embedder, sample_papers_list):
        """Test batch embedding of papers."""
        with patch.object(paper_embedder.model, "encode") as mock_encode:
            mock_encode.return_value = np.array([[0.1] * 768] * len(sample_papers_list))

            embeddings = paper_embedder.embed_papers(sample_papers_list)

            assert isinstance(embeddings, np.ndarray)
            assert len(embeddings) == len(sample_papers_list)

    def test_embed_empty_query(self, paper_embedder):
        """Test embedding empty query."""
        with patch.object(paper_embedder.model, "encode") as mock_encode:
            mock_encode.return_value = np.array([0.0] * 768)

            embedding = paper_embedder.embed_query("")

            assert isinstance(embedding, np.ndarray)


@pytest.mark.unit
class TestEmbeddingBehavior:
    """Test embedding behavior."""

    def test_multiple_queries(self, paper_embedder):
        """Test that multiple queries are handled correctly."""
        with patch.object(paper_embedder.model, "encode") as mock_encode:
            mock_encode.return_value = np.array([0.1, 0.2, 0.3])

            # Call multiple times
            emb1 = paper_embedder.embed_query("query 1")
            emb2 = paper_embedder.embed_query("query 2")

            # Should encode each query separately
            assert mock_encode.call_count == 2
            np.testing.assert_array_equal(emb1, emb2)  # Same mock return value


@pytest.mark.unit
class TestEmbeddingSimilarity:
    """Test similarity calculations."""

    def test_compute_similarity(self, paper_embedder):
        """Test similarity calculation."""
        vec1 = np.array([1.0, 0.0, 0.0])
        vec2 = np.array([1.0, 0.0, 0.0])

        similarity = paper_embedder.compute_similarity(vec1, vec2)

        assert 0.99 <= similarity <= 1.01  # Should be 1.0 (identical)

    def test_compute_similarity_orthogonal(self, paper_embedder):
        """Test similarity for orthogonal vectors."""
        vec1 = np.array([1.0, 0.0, 0.0])
        vec2 = np.array([0.0, 1.0, 0.0])

        similarity = paper_embedder.compute_similarity(vec1, vec2)

        assert -0.01 <= similarity <= 0.01  # Should be 0.0 (orthogonal)

    def test_find_most_similar(self, paper_embedder):
        """Test finding most similar papers."""
        # Create mock embeddings array
        paper_embeddings = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.9, 0.1, 0.0],
                [0.0, 1.0, 0.0],
            ]
        )

        query_embedding = np.array([1.0, 0.0, 0.0])
        similar = paper_embedder.find_most_similar(query_embedding, paper_embeddings, top_k=2)

        assert len(similar) <= 2
        assert all(isinstance(item, tuple) for item in similar)
        # First result should be most similar (index 0)
        assert similar[0][0] == 0


@pytest.mark.integration
@pytest.mark.slow
class TestPaperEmbedderIntegration:
    """Integration tests (requires model download)."""

    def test_real_embedding_generation(self):
        """Test real embedding generation with SPECTER."""
        embedder = PaperEmbedder()

        query = "Machine learning is a field of artificial intelligence."
        embedding = embedder.embed_query(query)

        assert isinstance(embedding, np.ndarray)
        assert len(embedding) == 768  # SPECTER embedding dimension

    def test_real_paper_embedding(self, sample_paper_metadata):
        """Test real paper embedding."""
        embedder = PaperEmbedder()

        embedding = embedder.embed_paper(sample_paper_metadata)

        assert isinstance(embedding, np.ndarray)
        assert len(embedding) == 768
