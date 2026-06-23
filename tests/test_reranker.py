"""
Tests for cross-encoder reranking functionality.
Tests reranking logic, score improvements, and ranking changes.
"""

from unittest.mock import Mock, patch
import sys
sys.path.insert(0, 'dags')

from tasks.reranker import Reranker, rerank_results


class TestReranker:
    """Test the Reranker class."""
    
    @patch('tasks.reranker.CrossEncoder')
    def test_reranker_initialization(self, mock_cross_encoder):
        """Test Reranker initializes with correct model."""
        mock_model = Mock()
        mock_cross_encoder.return_value = mock_model
        
        reranker = Reranker(model_name='test-model')
        
        mock_cross_encoder.assert_called_once_with('test-model', device='cpu', max_length=512)
        assert reranker.model == mock_model
    
    @patch('tasks.reranker.CrossEncoder')
    def test_reranker_default_model(self, mock_cross_encoder):
        """Test Reranker uses default model if none specified."""
        mock_model = Mock()
        mock_cross_encoder.return_value = mock_model
        
        Reranker()
        
        mock_cross_encoder.assert_called_once_with('cross-encoder/ms-marco-MiniLM-L-6-v2', device='cpu', max_length=512)
    
    @patch('tasks.reranker.CrossEncoder')
    def test_rerank_empty_results(self, mock_cross_encoder):
        """Test reranking with no results."""
        mock_model = Mock()
        mock_cross_encoder.return_value = mock_model
        
        reranker = Reranker()
        results = reranker.rerank(query="test", results=[], top_k=5)
        
        assert results == []
        mock_model.predict.assert_not_called()
    
    @patch('tasks.reranker.CrossEncoder')
    def test_rerank_adds_scores(self, mock_cross_encoder):
        """Test that reranking adds reranker_score to results."""
        mock_model = Mock()
        mock_model.predict.return_value = [0.9, 0.7, 0.5]
        mock_cross_encoder.return_value = mock_model
        
        reranker = Reranker()
        
        initial_results = [
            {'id': '1', 'payload': {'text': 'First document'}, 'score': 0.6},
            {'id': '2', 'payload': {'text': 'Second document'}, 'score': 0.8},
            {'id': '3', 'payload': {'text': 'Third document'}, 'score': 0.7},
        ]
        
        results = reranker.rerank(query="test query", results=initial_results, top_k=3)
        
        # Check all results have reranker_score
        assert all('reranker_score' in r for r in results)
        assert all('original_rank' in r for r in results)
    
    @patch('tasks.reranker.CrossEncoder')
    def test_rerank_changes_order(self, mock_cross_encoder):
        """Test that reranking can change result order."""
        mock_model = Mock()
        # Third document gets highest reranker score
        mock_model.predict.return_value = [0.5, 0.6, 0.9]
        mock_cross_encoder.return_value = mock_model
        
        reranker = Reranker()
        
        initial_results = [
            {'id': '1', 'payload': {'text': 'First'}, 'score': 0.9},
            {'id': '2', 'payload': {'text': 'Second'}, 'score': 0.8},
            {'id': '3', 'payload': {'text': 'Third'}, 'score': 0.7},
        ]
        
        results = reranker.rerank(query="test", results=initial_results, top_k=3)
        
        # Third document should now be first
        assert results[0]['id'] == '3'
        assert results[0]['reranker_score'] == 0.9
        assert results[0]['original_rank'] == 3
    
    @patch('tasks.reranker.CrossEncoder')
    def test_rerank_respects_top_k(self, mock_cross_encoder):
        """Test that reranking returns only top_k results."""
        mock_model = Mock()
        mock_model.predict.return_value = [0.9, 0.8, 0.7, 0.6, 0.5]
        mock_cross_encoder.return_value = mock_model
        
        reranker = Reranker()
        
        initial_results = [
            {'id': str(i), 'payload': {'text': f'Doc {i}'}, 'score': 0.5}
            for i in range(5)
        ]
        
        results = reranker.rerank(query="test", results=initial_results, top_k=3)
        
        assert len(results) == 3
    
    @patch('tasks.reranker.CrossEncoder')
    def test_rerank_handles_missing_text(self, mock_cross_encoder):
        """Test reranking gracefully handles missing text in payload."""
        mock_model = Mock()
        mock_model.predict.return_value = [0.8]
        mock_cross_encoder.return_value = mock_model
        
        reranker = Reranker()
        
        # Result with missing text field
        initial_results = [
            {'id': '1', 'payload': {}, 'score': 0.8, 'text': 'Fallback text'},
        ]
        
        results = reranker.rerank(query="test", results=initial_results, top_k=1)
        
        # Should not crash, should use fallback 'text' field
        assert len(results) == 1
        mock_model.predict.assert_called_once()


class TestRerankResults:
    """Test the main rerank_results function."""
    
    @patch('tasks.reranker.Reranker')
    def test_rerank_results_integration(self, mock_reranker_class):
        """Test the main reranking function."""
        mock_reranker = Mock()
        mock_reranked = [
            {'id': '2', 'reranker_score': 0.9, 'original_rank': 2},
            {'id': '1', 'reranker_score': 0.7, 'original_rank': 1},
        ]
        mock_reranker.rerank.return_value = mock_reranked
        mock_reranker_class.return_value = mock_reranker
        
        initial_results = [
            {'id': '1', 'score': 0.8},
            {'id': '2', 'score': 0.7},
        ]
        
        results = rerank_results(
            query="test query",
            results=initial_results,
            model_name="test-model",
            top_k=2
        )
        
        # Check Reranker was initialized with correct model
        mock_reranker_class.assert_called_once_with(model_name="test-model")
        
        # Check rerank was called
        mock_reranker.rerank.assert_called_once_with(
            query="test query",
            results=initial_results,
            top_k=2
        )
        
        assert results == mock_reranked
    
    @patch('tasks.reranker.Reranker')
    def test_rerank_results_exports_metrics(self, mock_reranker_class):
        """Test that reranking exports performance metrics."""
        mock_reranker = Mock()
        mock_reranked = [
            {'id': '1', 'reranker_score': 0.9, 'combined_score': 0.7},
        ]
        mock_reranker.rerank.return_value = mock_reranked
        mock_reranker_class.return_value = mock_reranker
        
        with patch('tasks.reranker.export_histogram') as mock_export:
            rerank_results(
                query="test",
                results=[{'id': '1', 'score': 0.7}],
                top_k=1
            )
            
            # Check metric was exported
            mock_export.assert_called_once()
            call_args = mock_export.call_args[0]
            assert call_args[0] == 'reranker_score_improvement'


class TestRerankingImpact:
    """Test the impact of reranking on result quality."""
    
    @patch('tasks.reranker.CrossEncoder')
    def test_reranking_improves_mrr(self, mock_cross_encoder):
        """Test that reranking can improve MRR by promoting relevant docs."""
        mock_model = Mock()
        # Simulate cross-encoder recognizing doc 3 as most relevant
        mock_model.predict.return_value = [0.3, 0.4, 0.95, 0.2]
        mock_cross_encoder.return_value = mock_model
        
        reranker = Reranker()
        
        # Initial ranking has relevant doc at position 3
        initial_results = [
            {'id': '1', 'payload': {'text': 'Unrelated doc'}, 'score': 0.9},
            {'id': '2', 'payload': {'text': 'Another unrelated'}, 'score': 0.85},
            {'id': '3', 'payload': {'text': 'Highly relevant doc'}, 'score': 0.8},
            {'id': '4', 'payload': {'text': 'Not relevant'}, 'score': 0.75},
        ]
        
        results = reranker.rerank(query="relevant query", results=initial_results, top_k=4)
        
        # After reranking, doc 3 should be first
        assert results[0]['id'] == '3'
        assert results[0]['original_rank'] == 3  # Was at position 3
        
        # MRR improved from 1/3 = 0.33 to 1/1 = 1.0