"""
Redis-backed cache for the full chunk list used to build the BM25 index
in the Streamlit search UI.

Why this exists:
Hybrid search needs the *entire* collection's text content to build a
BM25 index (perform_hybrid_search scores against all chunks, not just
the top-K vector hits). Without caching, that means a full
client.scroll() over every point in the collection on EVERY search
request — for a non-trivial knowledge base this is the single biggest
source of search latency, and it scales with collection size, not query
complexity. (This was previously a TODO left in streamlit_app/app.py:
"Get all chunks for BM25 (simplified - in production, cache this)".)

This cache stores the scrolled chunk list in Redis, keyed by collection
name, with a short TTL as a safety net plus active invalidation whenever
a task actually changes the collection's contents (upsert_to_qdrant,
promote_collection, rollback_collection) so stale results don't linger
longer than necessary.
"""

import json
import logging
import pickle
from typing import List, Dict, Any, Optional, Tuple

import redis

from .hash_store import get_redis_client

logger = logging.getLogger(__name__)

CHUNK_CACHE_KEY_PREFIX = "rag:chunks:"
BM25_CACHE_KEY_PREFIX  = "rag:bm25:"

# Short TTL: long enough to absorb a burst of searches against the same
# collection, short enough that a missed invalidation self-heals quickly
# rather than serving stale chunks indefinitely.
DEFAULT_TTL_SECONDS = 5 * 60


def get_cached_chunks(collection_name: str) -> Optional[List[Dict[str, Any]]]:
    """
    Fetch the cached chunk list for a collection, if present.

    Args:
        collection_name: Qdrant collection name

    Returns:
        The cached list of {'id': ..., 'text': ...} dicts, or None on a
        cache miss or Redis error. On None, the caller should fall back
        to scrolling Qdrant directly — caching is a performance
        optimization, never a correctness requirement.
    """
    try:
        client = get_redis_client()
        key = f"{CHUNK_CACHE_KEY_PREFIX}{collection_name}"
        raw = client.get(key)
        if raw is None:
            return None
        return json.loads(raw)
    except (redis.RedisError, json.JSONDecodeError, ValueError) as e:
        logger.warning(f"Redis chunk-cache read failed for '{collection_name}': {e}")
        return None


def set_cached_chunks(
    collection_name: str,
    chunks: List[Dict[str, Any]],
    ttl_seconds: int = DEFAULT_TTL_SECONDS,
) -> None:
    """
    Store the chunk list for a collection in Redis with a TTL.

    Failures are logged and swallowed — a failed cache write should never
    break a search that otherwise succeeded.

    Args:
        collection_name: Qdrant collection name
        chunks: List of {'id': ..., 'text': ...} dicts to cache
        ttl_seconds: Cache lifetime (default 5 minutes)
    """
    try:
        client = get_redis_client()
        key = f"{CHUNK_CACHE_KEY_PREFIX}{collection_name}"
        client.set(key, json.dumps(chunks).encode('utf-8'), ex=ttl_seconds)
        logger.debug(
            f"Cached {len(chunks)} chunks for '{collection_name}' (TTL {ttl_seconds}s)"
        )
    except redis.RedisError as e:
        logger.warning(f"Redis chunk-cache write failed for '{collection_name}': {e}")


def invalidate_chunk_cache(collection_name: str) -> None:
    """
    Drop the cached chunk list for a collection.

    Call this whenever a task actually changes the collection's contents
    (upsert, promote, rollback) so the next search rebuilds from Qdrant
    instead of serving chunks that no longer match what's stored.

    Args:
        collection_name: Qdrant collection name
    """
    try:
        client = get_redis_client()
        key = f"{CHUNK_CACHE_KEY_PREFIX}{collection_name}"
        client.delete(key)
        logger.info(f"Invalidated chunk cache for '{collection_name}'")
    except redis.RedisError as e:
        logger.warning(
            f"Redis chunk-cache invalidation failed for '{collection_name}': {e}"
        )
    # Always invalidate the BM25 cache too — it was built from these same chunks
    invalidate_bm25_cache(collection_name)


# ─── BM25 index cache ─────────────────────────────────────────────────────────

def get_bm25_index(
    collection_name: str,
) -> Tuple[Optional[object], Optional[List[Dict[str, Any]]]]:
    """
    Load the pre-built BM25Okapi index and companion chunk list from Redis.

    Returns:
        (bm25_index, chunks) on a hit, or (None, None) on a miss / Redis error.
        On (None, None) the caller should fall back to building the index inline
        from a fresh Qdrant scroll — caching is a performance optimisation, not
        a correctness requirement.

        chunks is a list of {'id': str, 'text': str, 'payload': dict} dicts,
        i.e. the full payload is included so sparse-only hits can be displayed
        without an extra round-trip to Qdrant.
    """
    try:
        client = get_redis_client()
        key = f"{BM25_CACHE_KEY_PREFIX}{collection_name}"
        raw = client.get(key)
        if raw is None:
            return None, None
        data = pickle.loads(raw)
        return data['bm25'], data['chunks']
    except (redis.RedisError, pickle.UnpicklingError, KeyError) as e:
        logger.warning(f"Redis BM25-cache read failed for '{collection_name}': {e}")
        return None, None


def set_bm25_index(
    collection_name: str,
    bm25_index: object,
    chunks: List[Dict[str, Any]],
    ttl_seconds: int = DEFAULT_TTL_SECONDS,
) -> None:
    """
    Pickle and store a pre-built BM25Okapi index together with the chunk list.

    Called by the Airflow `prepare_search_cache` task at the end of each
    pipeline run so the index is warm before the first search arrives.
    Failures are logged and swallowed.

    Args:
        collection_name: Qdrant collection name
        bm25_index:      A fully-built BM25Okapi object
        chunks:          List of {'id', 'text', 'payload'} dicts (same order
                         as the BM25 corpus so index positions align)
        ttl_seconds:     Cache lifetime (default 5 min)
    """
    try:
        client = get_redis_client()
        key  = f"{BM25_CACHE_KEY_PREFIX}{collection_name}"
        data = pickle.dumps({'bm25': bm25_index, 'chunks': chunks})
        client.set(key, data, ex=ttl_seconds)
        logger.info(
            f"Cached BM25 index for '{collection_name}' "
            f"({len(chunks)} chunks, TTL {ttl_seconds}s)"
        )
    except redis.RedisError as e:
        logger.warning(f"Redis BM25-cache write failed for '{collection_name}': {e}")


def invalidate_bm25_cache(collection_name: str) -> None:
    """
    Drop the cached BM25 index for a collection. Called automatically by
    invalidate_chunk_cache and directly by upsert/promote/rollback tasks.

    Args:
        collection_name: Qdrant collection name
    """
    try:
        client = get_redis_client()
        key = f"{BM25_CACHE_KEY_PREFIX}{collection_name}"
        client.delete(key)
        logger.info(f"Invalidated BM25 cache for '{collection_name}'")
    except redis.RedisError as e:
        logger.warning(
            f"Redis BM25-cache invalidation failed for '{collection_name}': {e}"
        )