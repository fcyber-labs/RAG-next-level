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
from typing import List, Dict, Any, Optional

import redis

from .hash_store import get_redis_client

logger = logging.getLogger(__name__)

CHUNK_CACHE_KEY_PREFIX = "rag:chunks:"

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