"""
Hybrid search combining BM25 (keyword) + Vector (semantic) search.
Provides better recall for exact matches and semantic queries.

Fair split
----------
Dense and sparse candidates are retrieved in equal halves of top_k:
  dense_k  = ceil(top_k / 2)   e.g. top_k=6  → 3 dense
  sparse_k = top_k - dense_k   e.g. top_k=6  → 3 sparse
The two sets are unioned (deduplicated by ID), scored with weighted fusion,
and the top top_k results are returned. This guarantees keyword-only
matches always get representation in the final list, not just when they
happened to appear in the Qdrant ANN top-N.

Pre-built index
---------------
The Airflow `prepare_search_cache` task builds and pickles the BM25Okapi
index after every successful upsert and stores it in Redis.
`perform_hybrid_search` accepts an optional `bm25_index` parameter; when
supplied, the O(N) tokenise-and-build step is skipped entirely (~1-3s
saved on large corpora). Callers that don't pass a pre-built index (e.g.
`run_retrieval_evaluation`) still work correctly — the index is built
inline from `chunks` as before.
"""

import logging
import math
from typing import Any, Dict, List, Optional

import numpy as np
from rank_bm25 import BM25Okapi

from utils.vector_store import search_similar

logger = logging.getLogger(__name__)


# ─── tokeniser (shared with prepare_search_cache) ─────────────────────────────

def tokenize(text: str) -> List[str]:
    """Simple word-level tokeniser for BM25."""
    return text.lower().split()


# ─── searcher ─────────────────────────────────────────────────────────────────

class HybridSearcher:
    """
    Combines BM25 keyword search with Qdrant dense vector search using
    weighted score fusion and a fair dense/sparse candidate split.
    """

    def __init__(
        self,
        chunks: List[Dict[str, Any]],
        bm25_weight: float = 0.3,
        bm25_index: Optional[BM25Okapi] = None,
    ):
        """
        Args:
            chunks:      Full chunk list [{'id', 'text', 'payload'}].
                         Must be in the same order as the BM25 corpus.
            bm25_weight: BM25 fraction of the combined score (default 0.3).
            bm25_index:  Pre-built BM25Okapi from Redis. If None the index
                         is built from chunks here.
        """
        self.chunks        = chunks
        self.bm25_weight   = bm25_weight
        self.vector_weight = 1.0 - bm25_weight

        if bm25_index is not None:
            self.bm25 = bm25_index
            logger.info("Using pre-built BM25 index from Redis ⚡")
        else:
            logger.info(f"Building BM25 index for {len(chunks)} chunks...")
            tokenized_corpus = [tokenize(c.get('text', '')) for c in chunks]
            self.bm25 = BM25Okapi(tokenized_corpus)
            logger.info("BM25 index built successfully")

        # id → corpus index for fast lookups
        self._idx_by_id: Dict[str, int] = {
            str(c.get('id', str(i))): i for i, c in enumerate(chunks)
        }

    def search(
        self,
        query: str,
        query_vector: List[float],
        collection_name: str,
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve ceil(top_k/2) dense candidates from Qdrant and
        floor(top_k/2) sparse candidates from BM25, union them, fuse
        scores, and return the top top_k results.

        Args:
            query:           Raw query text (BM25 tokenisation).
            query_vector:    Query embedding (Qdrant ANN search).
            collection_name: Qdrant collection to search.
            top_k:           Total candidates to return after fusion.

        Returns:
            List of result dicts with id, vector_score, bm25_score,
            combined_score, and payload.
        """
        dense_k  = math.ceil(top_k / 2)
        sparse_k = top_k - dense_k

        # ── 1. Dense candidates from Qdrant ───────────────────────────────
        try:
            dense_results = search_similar(
                collection_name=collection_name,
                query_vector=query_vector,
                limit=dense_k,
            )
        except Exception as e:
            logger.error(f"Qdrant vector search failed: {e}")
            dense_results = []

        dense_by_id: Dict[str, Dict] = {str(r['id']): r for r in dense_results}

        # ── 2. Sparse candidates from BM25 ────────────────────────────────
        query_tokens = tokenize(query)
        bm25_raw     = self.bm25.get_scores(query_tokens)

        bm25_max  = float(np.max(bm25_raw)) if len(bm25_raw) > 0 else 1.0
        bm25_norm = (bm25_raw / bm25_max) if bm25_max > 0 else bm25_raw

        top_sparse_idx = np.argsort(bm25_norm)[-sparse_k:][::-1]
        sparse_by_id: Dict[str, float] = {
            str(self.chunks[i].get('id', str(i))): float(bm25_norm[i])
            for i in top_sparse_idx
        }

        # ── 3. Union ──────────────────────────────────────────────────────
        all_ids = set(dense_by_id.keys()) | set(sparse_by_id.keys())

        # ── 4. Score fusion ───────────────────────────────────────────────
        results = []
        for cid in all_ids:
            vs = float(dense_by_id[cid].get('score', 0.0)) if cid in dense_by_id else 0.0

            if cid in sparse_by_id:
                bs = sparse_by_id[cid]
            else:
                idx = self._idx_by_id.get(cid)
                bs  = float(bm25_norm[idx]) if idx is not None else 0.0

            combined = self.vector_weight * vs + self.bm25_weight * bs

            # Payload: prefer live Qdrant payload, fall back to cached chunk
            if cid in dense_by_id:
                payload = dense_by_id[cid].get('payload', {})
            else:
                idx     = self._idx_by_id.get(cid)
                chunk   = self.chunks[idx] if idx is not None else {}
                payload = chunk.get('payload', {'text': chunk.get('text', '')})

            results.append({
                'id':             cid,
                'score':          vs,
                'vector_score':   vs,
                'bm25_score':     bs,
                'combined_score': combined,
                'payload':        payload,
            })

        # ── 5. Sort and return ────────────────────────────────────────────
        results.sort(key=lambda x: x['combined_score'], reverse=True)
        top_results = results[:top_k]

        if top_results:
            logger.info(
                f"Hybrid search ({dense_k} dense + {sparse_k} sparse → "
                f"{len(results)} candidates): "
                f"returning top {len(top_results)}, "
                f"best combined={top_results[0]['combined_score']:.3f}"
            )

        return top_results


# ─── public entry point ───────────────────────────────────────────────────────

def perform_hybrid_search(
    query: str,
    query_vector: List[float],
    collection_name: str,
    chunks: List[Dict[str, Any]],
    bm25_index: Optional[BM25Okapi] = None,
    bm25_weight: float = 0.3,
    top_k: int = 10,
    **kwargs,
) -> List[Dict[str, Any]]:
    """
    Run hybrid search: ceil(top_k/2) dense + floor(top_k/2) sparse,
    score-fused and sorted.

    Args:
        query:           Query text.
        query_vector:    Query embedding.
        collection_name: Qdrant collection.
        chunks:          Full chunk corpus [{'id', 'text', 'payload'}].
        bm25_index:      Pre-built BM25Okapi from Redis (optional).
                         Pass None to build inline (backwards-compatible).
        bm25_weight:     BM25 fraction of combined score (default 0.3).
        top_k:           Total results to return.

    Returns:
        Ranked list with id, vector_score, bm25_score, combined_score,
        payload.
    """
    if not chunks:
        logger.warning("No chunks provided for hybrid search — returning empty results")
        return []

    try:
        searcher = HybridSearcher(
            chunks=chunks,
            bm25_weight=bm25_weight,
            bm25_index=bm25_index,
        )
        results = searcher.search(
            query=query,
            query_vector=query_vector,
            collection_name=collection_name,
            top_k=top_k,
        )
    except Exception as e:
        logger.error(f"Hybrid search failed, falling back to vector-only: {e}")
        results = search_similar(collection_name, query_vector, limit=top_k)

    from utils.metrics_exporter import export_gauge
    export_gauge('hybrid_search_bm25_weight', bm25_weight)

    return results