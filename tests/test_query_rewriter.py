"""
Tests for query rewriting/expansion functionality.
Tests LLM-based and rule-based query expansion strategies.
"""

from unittest.mock import Mock, patch
import sys
sys.path.insert(0, 'dags')

from tasks.query_rewriter import (
    rewrite_query_with_llm,
    rewrite_query_rule_based,
    rewrite_query
)


class TestRuleBasedQueryRewriting:
    """Test rule-based query expansion."""
    
    def test_expand_simple_acronym(self):
        """Test expansion of common acronym."""
        query = "What is the PTO policy?"
        variations = rewrite_query_rule_based(query)
        
        # Should include original + expansions
        assert query in variations
        assert any('paid time off' in v.lower() for v in variations)
    
    def test_expand_multiple_acronyms(self):
        """Test expansion when multiple acronyms present."""
        query = "PTO and VPN setup"
        variations = rewrite_query_rule_based(query)
        
        assert len(variations) > 1
        # Should expand both acronyms
        assert any('paid time off' in v.lower() for v in variations)
        assert any('virtual private network' in v.lower() for v in variations)
    
    def test_no_expansion_for_unknown_terms(self):
        """Test that unknown terms don't cause errors."""
        query = "What is the XYZ123 policy?"
        variations = rewrite_query_rule_based(query)
        
        # Should still return original query
        assert query in variations
    
    def test_case_insensitive_matching(self):
        """Test that acronym matching is case-insensitive."""
        query = "what is pto?"
        variations = rewrite_query_rule_based(query)
        
        # Should match 'pto' (lowercase)
        assert any('paid time off' in v.lower() for v in variations)
    
    def test_or_variant_generation(self):
        """Test that OR variants are generated."""
        query = "PTO request"
        variations = rewrite_query_rule_based(query)
        
        # Should include "PTO OR paid time off"
        assert any('OR' in v for v in variations)
    
    def test_deduplication(self):
        """Test that duplicate variations are removed."""
        query = "PTO policy"
        variations = rewrite_query_rule_based(query)
        
        # No duplicates
        assert len(variations) == len(set(variations))
    
    def test_empty_query(self):
        """Test handling of empty query."""
        query = ""
        variations = rewrite_query_rule_based(query)
        
        assert variations == [""]


class TestLLMBasedQueryRewriting:
    """Test LLM-based query expansion."""
    
    @patch('tasks.query_rewriter.OpenAI')
    def test_llm_expansion_success(self, mock_openai):
        """Test successful LLM query expansion."""
        # Mock OpenAI response
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [
            Mock(message=Mock(content='["variation 1", "variation 2", "variation 3"]'))
        ]
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
        query = "How do I submit expenses?"
        variations = rewrite_query_with_llm(query, max_variations=3)
        
        # Should return original + LLM variations
        assert query in variations
        assert len(variations) == 4  # Original + 3 variations
    
    @patch('tasks.query_rewriter.OpenAI')
    def test_llm_expansion_respects_max_variations(self, mock_openai):
        """Test that max_variations parameter is respected."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [
            Mock(message=Mock(content='["var1", "var2"]'))
        ]
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
        query = "test query"
        variations = rewrite_query_with_llm(query, max_variations=2)
        
        # Should have original + 2 variations max
        assert len(variations) <= 3
    
    @patch('tasks.query_rewriter.OpenAI')
    def test_llm_expansion_handles_api_error(self, mock_openai):
        """Test fallback when OpenAI API fails."""
        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = Exception("API Error")
        mock_openai.return_value = mock_client
        
        query = "test query"
        variations = rewrite_query_with_llm(query)
        
        # Should return original query as fallback
        assert variations == [query]
    
    @patch('tasks.query_rewriter.OpenAI')
    def test_llm_expansion_handles_invalid_json(self, mock_openai):
        """Test handling of invalid JSON response from LLM."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [
            Mock(message=Mock(content='This is not valid JSON'))
        ]
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
        query = "test query"
        variations = rewrite_query_with_llm(query)
        
        # Should return original query as fallback
        assert variations == [query]
    
    @patch('tasks.query_rewriter.OpenAI')
    def test_llm_uses_correct_model(self, mock_openai):
        """Test that correct model is used for expansion."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [
            Mock(message=Mock(content='["variation"]'))
        ]
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
        query = "test"
        rewrite_query_with_llm(query)
        
        # Check that GPT-3.5-turbo was used
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs['model'] == 'gpt-3.5-turbo'


class TestRewriteQuery:
    """Test the main rewrite_query function."""
    
    @patch('tasks.query_rewriter.rewrite_query_with_llm')
    def test_rewrite_with_llm_enabled(self, mock_llm):
        """Test query rewriting with LLM enabled."""
        mock_llm.return_value = ["original", "variation 1", "variation 2"]
        
        result = rewrite_query(query="test", use_llm=True, max_variations=2)
        
        mock_llm.assert_called_once_with("test", 2)
        assert result == ["original", "variation 1", "variation 2"]
    
    @patch('tasks.query_rewriter.rewrite_query_rule_based')
    def test_rewrite_with_llm_disabled(self, mock_rule):
        """Test query rewriting with rule-based only."""
        mock_rule.return_value = ["original", "pto -> paid time off"]
        
        result = rewrite_query(query="PTO policy", use_llm=False)
        
        mock_rule.assert_called_once_with("PTO policy")
        assert len(result) == 2
    
    @patch('tasks.query_rewriter.export_counter')
    @patch('tasks.query_rewriter.rewrite_query_rule_based')
    def test_rewrite_exports_metrics(self, mock_rule, mock_export):
        """Test that query rewriting exports metrics."""
        mock_rule.return_value = ["original", "var1", "var2"]
        
        rewrite_query(query="test", use_llm=False)
        
        # Should export number of variations generated (excluding original)
        mock_export.assert_called_once_with('query_rewrites_total', 2)


class TestAcronymMapping:
    """Test the built-in acronym mappings."""
    
    def test_common_acronyms_covered(self):
        """Test that common business acronyms are mapped."""
        test_cases = [
            ("PTO", "paid time off"),
            ("API", "application programming interface"),
            ("VPN", "virtual private network"),
            ("HR", "human resources"),
            ("IT", "information technology"),
        ]
        
        for acronym, expansion in test_cases:
            query = f"What is {acronym}?"
            variations = rewrite_query_rule_based(query)
            
            # Should contain expansion
            assert any(expansion in v.lower() for v in variations), \
                f"Failed to expand {acronym} to {expansion}"
    
    def test_acronym_in_context(self):
        """Test acronym expansion works in sentence context."""
        query = "I need to access the VPN for remote work"
        variations = rewrite_query_rule_based(query)
        
        # Should expand VPN in context
        assert any('virtual private network' in v.lower() for v in variations)