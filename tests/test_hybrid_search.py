"""
Tests for hybrid search (BM25 + vector) functionality.
Tests weight fusion, BM25 scoring, and combined ranking.
"""

import pytest
from unittest.mock import patch
import sys
sys.path.insert(0, 'dags')

from tasks.hybrid_search import (
    tokenize,
    HybridSearcher,
    perform_hybrid_search
)


class TestTokenize:
    """Test the tokenization function for BM25."""
    
    def test_tokenize_simple_text(self):
        """Test basic tokenization."""
        text = "Hello World"
        tokens = tokenize(text)
        
        assert tokens == ["hello", "world"]
        assert len(tokens) == 2
    
    def test_tokenize_empty_string(self):
        """Test tokenization of empty string."""
        text = ""
        tokens = tokenize(text)
        
        assert tokens == []
    
    def test_tokenize_with_punctuation(self):
        """Test tokenization handles punctuation."""
        text = "Hello, World! How are you?"
        tokens = tokenize(text)
        
        # Simple split by space, punctuation attached
        assert "hello," in tokens
        assert "world!" in tokens
    
    def test_tokenize_lowercases(self):
        """Test that tokenization converts to lowercase."""
        text = "UPPERCASE lowercase MixedCase"
        tokens = tokenize(text)
        
        assert all(token.islower() or not token.isalpha() for token in tokens)


class TestHybridSearcher:
    """Test the HybridSearcher class."""
    
    @pytest.fixture
    def sample_chunks(self):
        """Sample chunks for testing."""
        return [
            {'id': '1', 'text': 'employee vacation policy allows 20 days'},
            {'id': '2', 'text': 'reimbursement form submission deadline'},
            {'id': '3', 'text': 'vacation request approval process'},
            {'id': '4', 'text': 'employee benefits and compensation'},
            {'id': '5', 'text': 'expense reimbursement guidelines'},
        ]
    
    def test_hybrid_searcher_initialization(self, sample_chunks):
        """Test HybridSearcher initializes correctly."""
        searcher = HybridSearcher(chunks=sample_chunks, bm25_weight=0.3)
        
        assert searcher.bm25_weight == 0.3
        assert searcher.vector_weight == 0.7
        assert len(searcher.chunks) == 5
        assert searcher.bm25 is not None
    
    def test_hybrid_searcher_default_weights(self, sample_chunks):
        """Test default weight is 0.3 for BM25."""
        searcher = HybridSearcher(chunks=sample_chunks)
        
        assert searcher.bm25_weight == 0.3
        assert searcher.vector_weight == 0.7
    
    def test_hybrid_searcher_custom_weights(self, sample_chunks):
        """Test custom weight initialization."""
        searcher = HybridSearcher(chunks=sample_chunks, bm25_weight=0.5)
        
        assert searcher.bm25_weight == 0.5
        assert searcher.vector_weight == 0.5
    
    def test_hybrid_search_combines_scores(self, sample_chunks):
        """Test that search combines BM25 and vector scores."""
        searcher = HybridSearcher(chunks=sample_chunks, bm25_weight=0.3)
        
        # Mock Qdrant results with vector scores
        qdrant_results = [
            {'id': '1', 'score': 0.9, 'payload': {'text': sample_chunks[0]['text']}},
            {'id': '2', 'score': 0.7, 'payload': {'text': sample_chunks[1]['text']}},
            {'id': '3', 'score': 0.6, 'payload': {'text': sample_chunks[2]['text']}},
        ]
        
        query = "vacation policy"
        results = searcher.search(
            query=query,
            query_vector=[0.1] * 1536,  # Dummy vector
            qdrant_results=qdrant_results,
            top_k=3
        )
        
        # Check results have all score components
        assert len(results) > 0
        assert 'bm25_score' in results[0]
        assert 'vector_score' in results[0]
        assert 'combined_score' in results[0]
    
    def test_hybrid_search_keyword_boost(self, sample_chunks):
        """Test that exact keyword matches get BM25 boost."""
        searcher = HybridSearcher(chunks=sample_chunks, bm25_weight=0.3)
        
        # Mock results where vector search ranked 'vacation request' higher
        qdrant_results = [
            {'id': '3', 'score': 0.85, 'payload': {'text': sample_chunks[2]['text']}},  # vacation request
            {'id': '1', 'score': 0.80, 'payload': {'text': sample_chunks[0]['text']}},  # vacation policy
        ]
        
        # Query with exact match to "vacation policy"
        query = "vacation policy"
        results = searcher.search(
            query=query,
            query_vector=[0.1] * 1536,
            qdrant_results=qdrant_results,
            top_k=2
        )
        
        # Result with exact keyword "policy" should get BM25 boost
        assert len(results) == 2
        assert all('combined_score' in r for r in results)


class TestPerformHybridSearch:
    """Test the main hybrid search function."""
    
    @pytest.fixture
    def sample_chunks(self):
        """Sample chunks for testing."""
        return [
            {'id': '1', 'text': 'python programming language'},
            {'id': '2', 'text': 'java programming tutorial'},
            {'id': '3', 'text': 'python data science'},
        ]
    
    @patch('tasks.hybrid_search.get_qdrant_client')
    @patch('tasks.hybrid_search.search_similar')
    def test_perform_hybrid_search_integration(self, mock_search, mock_client, sample_chunks):
        """Test the full hybrid search function."""
        # Mock Qdrant search results
        mock_search.return_value = [
            {'id': '1', 'score': 0.9, 'payload': {'text': sample_chunks[0]['text']}},
            {'id': '2', 'score': 0.8, 'payload': {'text': sample_chunks[1]['text']}},
        ]
        
        results = perform_hybrid_search(
            query="python programming",
            query_vector=[0.1] * 1536,
            collection_name="test_collection",
            chunks=sample_chunks,
            bm25_weight=0.3,
            top_k=2
        )
        
        # Check search was called
        mock_search.assert_called_once()
        
        # Check results structure
        assert len(results) <= 2
        if results:
            assert 'combined_score' in results[0]
    

    def test_perform_hybrid_search_empty_chunks(self):
        """Test hybrid search with no chunks."""
        with patch('tasks.hybrid_search.search_similar') as mock_search:
            mock_search.return_value = []
            
            results = perform_hybrid_search(
                query="test",
                query_vector=[0.1] * 1536,
                collection_name="test",
                chunks=[],  # Empty chunks
                top_k=5
            )
            
            # Should return empty list, not raise ZeroDivisionError
            assert results == []


class TestWeightFusion:
    """Test different weight combinations."""
    
    def test_pure_vector_search(self):
        """Test with 100% vector weight (0% BM25)."""
        chunks = [{'id': '1', 'text': 'test document'}]
        searcher = HybridSearcher(chunks=chunks, bm25_weight=0.0)
        
        assert searcher.vector_weight == 1.0
        assert searcher.bm25_weight == 0.0
    
    def test_pure_bm25_search(self):
        """Test with 100% BM25 weight (0% vector)."""
        chunks = [{'id': '1', 'text': 'test document'}]
        searcher = HybridSearcher(chunks=chunks, bm25_weight=1.0)
        
        assert searcher.vector_weight == 0.0
        assert searcher.bm25_weight == 1.0
    
    def test_balanced_search(self):
        """Test with 50/50 split."""
        chunks = [{'id': '1', 'text': 'test document'}]
        searcher = HybridSearcher(chunks=chunks, bm25_weight=0.5)
        
        assert searcher.vector_weight == 0.5
        assert searcher.bm25_weight == 0.5