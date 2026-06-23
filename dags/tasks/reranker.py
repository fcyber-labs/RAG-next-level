"""
Reranking retrieved results using cross-encoder models.
Improves relevance by scoring query-document pairs directly.
"""

import logging
from typing import Any, Dict, List

from sentence_transformers import CrossEncoder

from utils.metrics_exporter import export_histogram   # module-level – mocks work correctly

logger = logging.getLogger(__name__)


class Reranker:
    """
    Reranks search results using a cross-encoder model.
    Cross-encoders are more accurate than bi-encoders for ranking.
    """

    def __init__(self, model_name: str = 'cross-encoder/ms-marco-MiniLM-L-6-v2'):
        """
        Initialize reranker.

        Args:
            model_name: HuggingFace cross-encoder model name
        """
        logger.info(f"Loading reranker model: {model_name}")
        # Force CPU explicitly. Without this, newer versions of transformers
        # load weights as meta-tensors (lazy, device-less) and then
        # CrossEncoder.predict() calls model.to(target_device) which raises:
        #   NotImplementedError: Cannot copy out of meta tensor; no data!
        # Pinning to CPU prevents the device-move entirely.
        self.model = CrossEncoder(model_name, device='cpu', max_length=512)
        logger.info("Reranker model loaded successfully")

    def rerank(
        self,
        query: str,
        results: List[Dict[str, Any]],
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Rerank search results.

        Args:
            query: User query
            results: Initial search results (from vector/hybrid search)
            top_k: Number of results to return after reranking

        Returns:
            Reranked results with new scores
        """
        if not results:
            return []

        # Step 1: Prepare query-document pairs
        pairs = []
        for result in results:
            text = result.get('payload', {}).get('text', '')
            if not text:
                text = result.get('text', '')
            pairs.append([query, text])

        # Step 2: Score all pairs
        logger.info(f"Reranking {len(pairs)} results...")
        scores = self.model.predict(pairs)

        # Step 3: Add reranker scores to results
        for result, score in zip(results, scores):
            result['reranker_score'] = float(score)
            result['original_rank'] = results.index(result) + 1

        # Step 4: Sort by reranker score
        reranked = sorted(results, key=lambda x: x['reranker_score'], reverse=True)

        # Step 5: Return top-k
        top_results = reranked[:top_k]

        logger.info(
            f"Reranking complete. "
            f"Top result score: {top_results[0]['reranker_score']:.4f} "
            f"(was rank #{top_results[0]['original_rank']})"
        )

        return top_results


_RERANKER_CACHE: dict = {}


def rerank_results(
    query: str,
    results: List[Dict[str, Any]],
    model_name: str = 'cross-encoder/ms-marco-MiniLM-L-6-v2',
    top_k: int = 10,
    **kwargs,
) -> List[Dict[str, Any]]:
    """
    Main reranking function.

    Args:
        query: User query
        results: Search results to rerank
        model_name: Cross-encoder model
        top_k: Number of results to return

    Returns:
        Reranked results
    """
    if not results:
        return []

    # Cache Reranker instances by model_name. Without this, every call
    # (e.g. once per query during evaluation) reloads the CrossEncoder
    # model from disk from scratch — the same cost-multiplying bug that
    # was fixed in embed.py's _get_local_embeddings, just not yet applied
    # here. The model never changes between calls within a run.
    if model_name not in _RERANKER_CACHE:
        _RERANKER_CACHE[model_name] = Reranker(model_name=model_name)
    reranker = _RERANKER_CACHE[model_name]
    reranked = reranker.rerank(query=query, results=results, top_k=top_k)

    # Export metrics via module-level reference so unit-test mocks are honoured.
    # (Do NOT re-import export_histogram here – that would shadow the mock.)
    if reranked:
        original_score = results[0].get('combined_score', results[0].get('score', 0))
        reranked_score = reranked[0].get('reranker_score', 0)
        score_change = reranked_score - original_score
        export_histogram('reranker_score_improvement', score_change)

    return reranked