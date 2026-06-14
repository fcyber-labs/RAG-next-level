"""
Hybrid search combining BM25 (keyword) + Vector (semantic) search.
Provides better recall for exact matches and semantic queries.
"""

import logging
from typing import List, Dict, Any
from rank_bm25 import BM25Okapi
import numpy as np


from utils.vector_store import search_similar


logger = logging.getLogger(__name__)


def tokenize(text: str) -> List[str]:
    """
    Simple tokenization for BM25.
    
    Args:
        text: Input text
        
    Returns:
        List of lowercase tokens
    """
    # Simple word-level tokenization
    # In production, you might use spaCy or NLTK
    return text.lower().split()


class HybridSearcher:
    """
    Combines BM25 keyword search with vector semantic search.
    Uses weighted fusion to combine scores.
    """
    
    def __init__(self, chunks: List[Dict[str, Any]], bm25_weight: float = 0.3):
        """
        Initialize hybrid searcher.
        
        Args:
            chunks: List of chunk dictionaries with 'text' field
            bm25_weight: Weight for BM25 scores (0-1). Vector weight = 1 - bm25_weight
        """
        self.chunks = chunks
        self.bm25_weight = bm25_weight
        self.vector_weight = 1.0 - bm25_weight
        
        # Build BM25 index
        logger.info(f"Building BM25 index for {len(chunks)} chunks...")
        tokenized_corpus = [tokenize(chunk['text']) for chunk in chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)
        logger.info("BM25 index built successfully")
    
    def search(
        self,
        query: str,
        query_vector: List[float],
        qdrant_results: List[Dict[str, Any]],
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Perform hybrid search combining BM25 and vector scores.
        
        Args:
            query: Text query
            query_vector: Query embedding vector
            qdrant_results: Results from Qdrant vector search (with scores)
            top_k: Number of results to return
            
        Returns:
            List of results sorted by combined score
        """
        # Step 1: Get BM25 scores for all chunks
        query_tokens = tokenize(query)
        bm25_scores = self.bm25.get_scores(query_tokens)
        
        # Step 2: Normalize BM25 scores to 0-1 range
        bm25_max = np.max(bm25_scores) if len(bm25_scores) > 0 else 1.0
        if bm25_max > 0:
            bm25_scores_normalized = bm25_scores / bm25_max
        else:
            bm25_scores_normalized = bm25_scores
        
        # Step 3: Create chunk_id -> BM25 score mapping
        bm25_score_map = {}
        for idx, chunk in enumerate(self.chunks):
            chunk_id = chunk.get('id', str(idx))
            bm25_score_map[chunk_id] = bm25_scores_normalized[idx]
        
        # Step 4: Combine with vector scores
        combined_results = []
        for result in qdrant_results:
            chunk_id = result.get('id')
            vector_score = result.get('score', 0.0)
            bm25_score = bm25_score_map.get(chunk_id, 0.0)
            
            # Weighted fusion
            combined_score = (
                self.vector_weight * vector_score +
                self.bm25_weight * bm25_score
            )
            
            result['bm25_score'] = float(bm25_score)
            result['vector_score'] = float(vector_score)
            result['combined_score'] = float(combined_score)
            combined_results.append(result)
        
        # Step 5: Re-rank by combined score
        combined_results.sort(key=lambda x: x['combined_score'], reverse=True)
        
        # Step 6: Return top-k
        top_results = combined_results[:top_k]
        
        logger.info(
            f"Hybrid search complete: "
            f"BM25 weight={self.bm25_weight}, "
            f"Vector weight={self.vector_weight}, "
            f"Top result combined_score={top_results[0]['combined_score']:.3f}"
        )
        
        return top_results

def perform_hybrid_search(
    query: str,
    query_vector: List[float],
    collection_name: str,
    chunks: List[Dict[str, Any]],
    bm25_weight: float = 0.3,
    top_k: int = 10,
    **kwargs
) -> List[Dict[str, Any]]:
    """
    Main function to perform hybrid search.
    
    Args:
        query: Text query
        query_vector: Query embedding
        collection_name: Qdrant collection name
        chunks: All chunks with text (for BM25)
        bm25_weight: Weight for BM25 (default 0.3 = 30% BM25, 70% vector)
        top_k: Number of results
        
    Returns:
        Hybrid search results
    """
    
    
    # Handle empty chunks case
    if not chunks:
        logger.warning("No chunks provided, returning empty results")
        return []
    
    # Step 1: Get vector search results from Qdrant
    logger.info(f"Performing vector search in '{collection_name}'...")
    try:
        qdrant_results = search_similar(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=top_k * 2  # Get more results for re-ranking
        )
    except Exception as e:
        logger.error(f"Vector search failed: {e}")
        qdrant_results = []
    
    # Step 2: Initialize hybrid searcher (will fail if chunks empty, but we already checked)
    try:
        searcher = HybridSearcher(chunks=chunks, bm25_weight=bm25_weight)
        
        # Step 3: Combine scores
        hybrid_results = searcher.search(
            query=query,
            query_vector=query_vector,
            qdrant_results=qdrant_results,
            top_k=top_k
        )
    except ZeroDivisionError as e:
        logger.error(f"BM25 error (empty corpus?): {e}")
        # Fallback to vector results only
        hybrid_results = qdrant_results[:top_k]
    
    logger.info(f"Hybrid search returned {len(hybrid_results)} results")
    
    # Export metrics
    from utils.metrics_exporter import export_gauge
    export_gauge('hybrid_search_bm25_weight', bm25_weight)
    
    return hybrid_results