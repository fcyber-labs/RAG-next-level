"""
Redis-backed semantic cache for generated LLM answers.

SEMANTIC CACHING — how it works
--------------------------------
Instead of requiring the query string to match exactly (hash equality),
this cache embeds every incoming query and compares it against the
embeddings of previously-answered queries using cosine similarity.  Any
stored answer whose query embedding is within `similarity_threshold`
(default 0.95) of the current query's embedding is treated as a hit and
returned without calling the LLM.

This catches natural paraphrases automatically, e.g.:
  "What is the vacation policy?"   → cache miss  → LLM called, stored
  "Tell me about the vacation policy" → similarity 0.97 → cache HIT

Redis data layout (per collection + settings fingerprint)
---------------------------------------------------------
Each cached entry is stored at:
    rag:semantic_answer:{collection}:{uuid4}
as a JSON blob containing:
    {
      "answer":    str,
      "embedding": List[float],   ← the query embedding, stored for comparison
      "cached_at": float,         ← unix timestamp
      "settings":  {...}          ← retrieval settings fingerprint
    }

An index set per (collection, settings-hash) tracks which UUIDs belong
to which context so that:
  1. Lookup only scans entries with the SAME retrieval settings (a query
     answered with reranking=True must not hit a cache entry that was
     built with reranking=False, because different chunks → different answer).
  2. Invalidation only needs to delete the index + its member keys.

Index key:
    rag:semantic_index:{collection}:{settings_hash}

STALENESS
---------
Same two-layer approach as chunk_cache.py:
  - TTL (default 5 min) self-heals any missed invalidation.
  - Active invalidation: upsert_to_qdrant / promote_collection /
    rollback_collection all call invalidate_answer_cache() whenever the
    underlying collection changes, so a cache hit can never outlive the
    data it was generated from.
"""

import json
import logging
import math
import time
import uuid
from typing import Optional, Dict, Any, List

import redis

from .hash_store import get_redis_client

logger = logging.getLogger(__name__)

ENTRY_KEY_PREFIX  = "rag:semantic_answer:"
INDEX_KEY_PREFIX  = "rag:semantic_index:"
DEFAULT_TTL_SECONDS       = 5 * 60   # 5 minutes — same as chunk cache
SIMILARITY_THRESHOLD      = 0.8     # cosine similarity to count as a hit


# ─── helpers ──────────────────────────────────────────────────────────────────

def _cosine_similarity(a: List[float], b: List[float]) -> float:
    """Pure-Python cosine similarity — no numpy required."""
    dot  = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _settings_hash(
    use_hybrid_search: bool,
    use_query_rewriting: bool,
    use_reranking: bool,
    top_k: int,
) -> str:
    """
    Short deterministic identifier for the retrieval-settings combination.
    Entries with different settings never compete in the similarity scan.
    """
    import hashlib
    blob = json.dumps(
        {
            "use_hybrid_search":   bool(use_hybrid_search),
            "use_query_rewriting": bool(use_query_rewriting),
            "use_reranking":       bool(use_reranking),
            "top_k":               int(top_k),
        },
        sort_keys=True,
    )
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def _index_key(collection_name: str, s_hash: str) -> str:
    return f"{INDEX_KEY_PREFIX}{collection_name}:{s_hash}"


def _entry_key(entry_id: str) -> str:
    return f"{ENTRY_KEY_PREFIX}{entry_id}"


# ─── public API ───────────────────────────────────────────────────────────────

def get_cached_answer(
    collection_name: str,
    query_embedding: List[float],
    use_hybrid_search: bool,
    use_query_rewriting: bool,
    use_reranking: bool,
    top_k: int,
    similarity_threshold: float = SIMILARITY_THRESHOLD,
) -> Optional[Dict[str, Any]]:
    """
    Semantic cache lookup.

    Scans all stored answers for (collection_name + settings) and returns
    the best match if its cosine similarity to query_embedding is ≥
    similarity_threshold.

    Args:
        collection_name:       Qdrant collection being searched.
        query_embedding:       Embedding of the current query (already
                               computed in Step 2 of the search flow —
                               reused here, no extra LLM/embed call).
        use_hybrid_search:     Retrieval setting (part of settings fingerprint).
        use_query_rewriting:   Retrieval setting.
        use_reranking:         Retrieval setting.
        top_k:                 Retrieval setting.
        similarity_threshold:  Minimum cosine similarity to count as a hit
                               (default 0.95).

    Returns:
        {'answer': str, 'cached_at': float, 'similarity': float} on a hit,
        or None on a miss or Redis error.
    """
    try:
        rc      = get_redis_client()
        s_hash  = _settings_hash(use_hybrid_search, use_query_rewriting, use_reranking, top_k)
        idx_key = _index_key(collection_name, s_hash)

        # Fetch all entry IDs for this (collection, settings) combination
        entry_ids = rc.smembers(idx_key)
        if not entry_ids:
            return None

        best_sim   = -1.0
        best_entry = None

        for raw_id in entry_ids:
            eid = raw_id.decode() if isinstance(raw_id, bytes) else raw_id
            raw = rc.get(_entry_key(eid))
            if raw is None:
                # Entry expired via TTL but index set wasn't cleaned up yet
                rc.srem(idx_key, eid)
                continue

            entry = json.loads(raw)
            stored_embedding = entry.get("embedding")
            if not stored_embedding:
                continue

            sim = _cosine_similarity(query_embedding, stored_embedding)
            if sim > best_sim:
                best_sim   = sim
                best_entry = entry

        if best_entry is not None and best_sim >= similarity_threshold:
            logger.info(
                f"Semantic cache HIT for '{collection_name}' "
                f"(similarity={best_sim:.4f} ≥ {similarity_threshold})"
            )
            return {
                "answer":     best_entry["answer"],
                "cached_at":  best_entry["cached_at"],
                "similarity": best_sim,
            }

        logger.debug(
            f"Semantic cache miss for '{collection_name}' "
            f"(best similarity={best_sim:.4f} < {similarity_threshold})"
        )
        return None

    except (redis.RedisError, json.JSONDecodeError, ValueError) as e:
        logger.warning(f"Redis semantic-cache read failed: {e}")
        return None


def set_cached_answer(
    collection_name: str,
    query_embedding: List[float],
    use_hybrid_search: bool,
    use_query_rewriting: bool,
    use_reranking: bool,
    top_k: int,
    answer: str,
    ttl_seconds: int = DEFAULT_TTL_SECONDS,
) -> None:
    """
    Store a freshly-generated LLM answer together with the query embedding
    so future similar queries can match it semantically.

    The entry is registered in the index set for its (collection, settings)
    combination so invalidation can find and delete it.

    Failures are logged and swallowed — a failed cache write should never
    break a search that already succeeded.
    """
    try:
        rc      = get_redis_client()
        s_hash  = _settings_hash(use_hybrid_search, use_query_rewriting, use_reranking, top_k)
        idx_key = _index_key(collection_name, s_hash)
        eid     = str(uuid.uuid4())
        ekey    = _entry_key(eid)

        payload = json.dumps({
            "answer":    answer,
            "embedding": query_embedding,
            "cached_at": time.time(),
            "settings":  {
                "use_hybrid_search":   use_hybrid_search,
                "use_query_rewriting": use_query_rewriting,
                "use_reranking":       use_reranking,
                "top_k":               top_k,
            },
        })

        pipe = rc.pipeline()
        pipe.set(ekey, payload.encode("utf-8"), ex=ttl_seconds)
        pipe.sadd(idx_key, eid)
        pipe.expire(idx_key, ttl_seconds)
        pipe.execute()

        logger.debug(
            f"Stored semantic cache entry '{eid}' for '{collection_name}' "
            f"(TTL {ttl_seconds}s)"
        )

    except redis.RedisError as e:
        logger.warning(f"Redis semantic-cache write failed: {e}")


def invalidate_answer_cache(collection_name: str) -> None:
    """
    Drop every cached answer for a collection (all queries, all settings
    combinations). Call this whenever the collection's contents change —
    upsert, promote, rollback — same trigger points as
    chunk_cache.invalidate_chunk_cache().
    """
    try:
        rc = get_redis_client()

        # Find all index keys for this collection (all settings hashes)
        idx_pattern = f"{INDEX_KEY_PREFIX}{collection_name}:*"
        idx_keys    = rc.keys(idx_pattern)

        total_deleted = 0
        for idx_key in idx_keys:
            entry_ids = rc.smembers(idx_key)
            if entry_ids:
                ekeys = [_entry_key(
                    eid.decode() if isinstance(eid, bytes) else eid
                ) for eid in entry_ids]
                rc.delete(*ekeys)
                total_deleted += len(ekeys)
            rc.delete(idx_key)

        if total_deleted:
            logger.info(
                f"Invalidated {total_deleted} semantic cache entries "
                f"for '{collection_name}'"
            )

    except redis.RedisError as e:
        logger.warning(
            f"Redis semantic-cache invalidation failed for '{collection_name}': {e}"
        )