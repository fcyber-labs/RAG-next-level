"""
Redis-backed cache for generated LLM answers in the Streamlit search UI.

This is a companion to chunk_cache.py: chunk_cache avoids re-scrolling
Qdrant for the BM25 index, this avoids re-calling the (paid, slower) LLM
for a question that's already been answered recently with the same
retrieval settings against the same collection.

IMPORTANT — staleness is handled the same way as the chunk cache:
  - Short TTL as a safety net (default 5 minutes).
  - Active invalidation: upsert_to_qdrant / promote_collection /
    rollback_collection all call invalidate_answer_cache() for any
    collection whose contents just changed, so a cache hit can never
    outlive the data it was generated from.

The cache key is NOT just the raw query string — it also folds in
collection_name and every retrieval setting that can change which chunks
get passed to the LLM (use_hybrid_search, use_query_rewriting,
use_reranking, top_k). Two identical questions asked with different
toggle settings are treated as different cache entries, because they can
legitimately produce different answers from different context.

On a cache hit, the answer is returned WITHOUT calling the LLM — no
streaming, no fresh generation. The UI MUST make this visible (see the
"⚡ Cached answer" badge in streamlit_app/app.py); silently returning a
cached answer in place of a "live" one would be misleading.
"""

import hashlib
import json
import logging
import time
from typing import Optional, Dict, Any

import redis

from .hash_store import get_redis_client

logger = logging.getLogger(__name__)

ANSWER_CACHE_KEY_PREFIX = "rag:answer:"
DEFAULT_TTL_SECONDS = 5 * 60  # same window as the chunk cache


def _cache_key(
    collection_name: str,
    query: str,
    use_hybrid_search: bool,
    use_query_rewriting: bool,
    use_reranking: bool,
    top_k: int,
) -> str:
    """
    Build a deterministic cache key from the question AND every retrieval
    setting that can change which chunks the LLM sees. Two questions with
    different toggle combinations are deliberately different cache
    entries.
    """
    normalized_query = query.strip().lower()
    fingerprint = json.dumps(
        {
            "query": normalized_query,
            "use_hybrid_search": bool(use_hybrid_search),
            "use_query_rewriting": bool(use_query_rewriting),
            "use_reranking": bool(use_reranking),
            "top_k": int(top_k),
        },
        sort_keys=True,
    )
    digest = hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()
    return f"{ANSWER_CACHE_KEY_PREFIX}{collection_name}:{digest}"


def get_cached_answer(
    collection_name: str,
    query: str,
    use_hybrid_search: bool,
    use_query_rewriting: bool,
    use_reranking: bool,
    top_k: int,
) -> Optional[Dict[str, Any]]:
    """
    Look up a cached answer for this exact (collection, query, settings)
    combination.

    Returns:
        {'answer': str, 'cached_at': float} on a hit, or None on a miss
        or Redis error. On None, the caller should call the LLM as
        normal — caching is a performance/cost optimization, never a
        correctness requirement.
    """
    try:
        client = get_redis_client()
        key = _cache_key(
            collection_name, query, use_hybrid_search,
            use_query_rewriting, use_reranking, top_k,
        )
        raw = client.get(key)
        if raw is None:
            return None
        return json.loads(raw)
    except (redis.RedisError, json.JSONDecodeError, ValueError) as e:
        logger.warning(f"Redis answer-cache read failed: {e}")
        return None


def set_cached_answer(
    collection_name: str,
    query: str,
    use_hybrid_search: bool,
    use_query_rewriting: bool,
    use_reranking: bool,
    top_k: int,
    answer: str,
    ttl_seconds: int = DEFAULT_TTL_SECONDS,
) -> None:
    """
    Store a freshly-generated LLM answer, keyed by query + collection +
    retrieval settings. Failures are logged and swallowed — a failed
    cache write should never break an answer that already succeeded.
    """
    try:
        client = get_redis_client()
        key = _cache_key(
            collection_name, query, use_hybrid_search,
            use_query_rewriting, use_reranking, top_k,
        )
        payload = json.dumps({"answer": answer, "cached_at": time.time()})
        client.set(key, payload.encode("utf-8"), ex=ttl_seconds)
        logger.debug(f"Cached LLM answer for '{collection_name}' (TTL {ttl_seconds}s)")
    except redis.RedisError as e:
        logger.warning(f"Redis answer-cache write failed: {e}")


def invalidate_answer_cache(collection_name: str) -> None:
    """
    Drop every cached answer for a collection (all queries, all settings
    combinations). Call this whenever the collection's contents change —
    upsert, promote, rollback — the same trigger points used by
    chunk_cache.invalidate_chunk_cache(), so an answer can never outlive
    the data it was generated from.
    """
    try:
        client = get_redis_client()
        pattern = f"{ANSWER_CACHE_KEY_PREFIX}{collection_name}:*"
        keys = client.keys(pattern)
        if keys:
            client.delete(*keys)
            logger.info(f"Invalidated {len(keys)} cached answers for '{collection_name}'")
    except redis.RedisError as e:
        logger.warning(f"Redis answer-cache invalidation failed for '{collection_name}': {e}")