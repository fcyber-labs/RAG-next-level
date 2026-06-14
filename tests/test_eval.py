"""
Test evaluation metrics: Recall@K and MRR computation.
Tests correctness of retrieval quality metrics.
"""

from dags.tasks.run_eval import _compute_recall_at_k, _compute_mrr


def test_recall_at_k_perfect():
    """
    Test Recall@K with perfect retrieval (all relevant docs in top-K).
    """
    retrieved = ['doc1', 'doc2', 'doc3', 'doc4', 'doc5']
    expected = ['doc1', 'doc2']
    
    recall = _compute_recall_at_k(retrieved, expected, k=5)
    
    # Both expected docs are in top-5, so recall = 2/2 = 1.0
    assert recall == 1.0, "Perfect retrieval should have recall of 1.0"


def test_recall_at_k_partial():
    """
    Test Recall@K with partial retrieval.
    """
    retrieved = ['doc1', 'doc3', 'doc5', 'doc7', 'doc9']
    expected = ['doc1', 'doc2', 'doc3']
    
    recall = _compute_recall_at_k(retrieved, expected, k=5)
    
    # 2 out of 3 expected docs are in top-5, so recall = 2/3 = 0.667
    assert abs(recall - 0.667) < 0.01, "Partial retrieval should have recall of ~0.667"


def test_recall_at_k_zero():
    """
    Test Recall@K with no relevant docs retrieved.
    """
    retrieved = ['doc5', 'doc6', 'doc7', 'doc8', 'doc9']
    expected = ['doc1', 'doc2', 'doc3']
    
    recall = _compute_recall_at_k(retrieved, expected, k=5)
    
    # No expected docs in top-5, so recall = 0/3 = 0.0
    assert recall == 0.0, "No relevant docs should have recall of 0.0"


def test_recall_at_k_with_k():
    """
    Test Recall@K with different K values.
    """
    retrieved = ['doc1', 'doc5', 'doc2', 'doc7', 'doc9']
    expected = ['doc1', 'doc2']
    
    recall_at_1 = _compute_recall_at_k(retrieved, expected, k=1)
    recall_at_3 = _compute_recall_at_k(retrieved, expected, k=3)
    recall_at_5 = _compute_recall_at_k(retrieved, expected, k=5)
    
    # At k=1: doc1 is in top-1, so 1/2 = 0.5
    assert recall_at_1 == 0.5, "Recall@1 should be 0.5"
    
    # At k=3: doc1 and doc2 are in top-3, so 2/2 = 1.0
    assert recall_at_3 == 1.0, "Recall@3 should be 1.0"
    
    # At k=5: same as k=3
    assert recall_at_5 == 1.0, "Recall@5 should be 1.0"


def test_mrr_first_position():
    """
    Test MRR when first relevant doc is at position 1.
    """
    retrieved = ['doc1', 'doc2', 'doc3']
    expected = ['doc1']
    
    mrr = _compute_mrr(retrieved, expected)
    
    # First relevant doc at position 1, so MRR = 1/1 = 1.0
    assert mrr == 1.0, "First position should give MRR of 1.0"


def test_mrr_third_position():
    """
    Test MRR when first relevant doc is at position 3.
    """
    retrieved = ['doc5', 'doc6', 'doc1', 'doc7']
    expected = ['doc1', 'doc2']
    
    mrr = _compute_mrr(retrieved, expected)
    
    # First relevant doc (doc1) at position 3, so MRR = 1/3 = 0.333
    assert abs(mrr - 0.333) < 0.01, "Third position should give MRR of ~0.333"


def test_mrr_no_relevant():
    """
    Test MRR when no relevant docs are retrieved.
    """
    retrieved = ['doc5', 'doc6', 'doc7']
    expected = ['doc1', 'doc2']
    
    mrr = _compute_mrr(retrieved, expected)
    
    # No relevant docs, so MRR = 0.0
    assert mrr == 0.0, "No relevant docs should give MRR of 0.0"


def test_mrr_multiple_relevant():
    """
    Test MRR with multiple relevant docs (only first position matters).
    """
    retrieved = ['doc3', 'doc1', 'doc2', 'doc4']
    expected = ['doc1', 'doc2']
    
    mrr = _compute_mrr(retrieved, expected)
    
    # First relevant doc (doc1) at position 2, so MRR = 1/2 = 0.5
    # Note: MRR only considers the FIRST relevant doc found
    assert mrr == 0.5, "Second position should give MRR of 0.5"


def test_empty_expected_docs():
    """
    Test metrics with empty expected docs list.
    """
    retrieved = ['doc1', 'doc2', 'doc3']
    expected = []
    
    recall = _compute_recall_at_k(retrieved, expected, k=5)
    mrr = _compute_mrr(retrieved, expected)
    
    assert recall == 0.0, "Empty expected should give recall of 0.0"
    assert mrr == 0.0, "Empty expected should give MRR of 0.0"