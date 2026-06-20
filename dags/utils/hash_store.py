"""
Redis-based document hash store for deduplication.

TWO-PHASE DESIGN — this is the important part:

  1. document_hash_exists()  — READ-ONLY. Used early, in deduplicate_documents,
                                to decide whether a document needs (re)processing.
                                It never writes to Redis.

  2. confirm_document_hash() — WRITE. Used late, in upsert_to_qdrant, and
                                called ONLY after Qdrant has confirmed the
                                document's vectors were actually stored.

Why this matters:
Redis must only "remember" a document once the work it represents is truly
done. If deduplicate_documents wrote to Redis immediately (the old design),
and a later task (chunk / embed / upsert) then failed, Redis would
permanently remember a document that was never actually saved — a "ghost
duplicate". Every future run would skip it forever (until the 30-day TTL
expired) while the database stayed empty, with no way to tell the
difference between "nothing changed" and "the pipeline is broken".

With this two-phase design, a crash anywhere between dedup and upsert
leaves Redis untouched for that document, so the next run correctly
retries it instead of skipping it forever.
"""

import logging
import os
import redis
import hashlib
import json

logger = logging.getLogger(__name__)

HASH_KEY_PREFIX = "rag:doc:hash:"
DEFAULT_EXPIRY_SECONDS = 60 * 60 * 24 * 30  # 30 days


def get_redis_client() -> redis.Redis:
    """Get a Redis client connection."""
    redis_host = os.getenv('REDIS_HOST', 'localhost')
    redis_port = int(os.getenv('REDIS_PORT', 6379))

    return redis.Redis(
        host=redis_host,
        port=redis_port,
        db=0,
        decode_responses=False,
    )


def compute_content_hash(content: str) -> str:
    """
    Compute SHA-256 hash of content.

    Args:
        content: Text content to hash

    Returns:
        Hex digest of hash
    """
    return hashlib.sha256(content.encode('utf-8')).hexdigest()


def document_hash_exists(content_hash: str) -> bool:
    """
    PHASE 1 — READ-ONLY check: has this content hash already been
    confirmed as stored?

    This never writes to Redis. It is safe to call before the document
    has actually been processed, since it never claims the key — that
    only happens in confirm_document_hash(), after a real success.

    Args:
        content_hash: SHA-256 hex digest of the document content

    Returns:
        True if the hash is already confirmed in Redis (a true duplicate
        that's really stored in Qdrant), False otherwise.
        On Redis error, returns False (treat as "not seen yet") so the
        document gets processed rather than silently dropped.
    """
    try:
        client = get_redis_client()
        redis_key = f"{HASH_KEY_PREFIX}{content_hash}"
        return bool(client.exists(redis_key))
    except redis.RedisError as e:
        logger.error(f"Redis error checking document hash {content_hash}: {e}")
        return False


def confirm_document_hash(
    content_hash: str,
    metadata: dict,
    expiry_seconds: int = DEFAULT_EXPIRY_SECONDS,
) -> bool:
    """
    PHASE 2 — WRITE: mark a content hash as confirmed/seen in Redis.

    Call this ONLY after the document's vectors have been successfully
    upserted into Qdrant. This is what makes the dedup cache trustworthy:
    a hash only appears here once the work it represents is truly done.

    Args:
        content_hash: SHA-256 hex digest of the document content
        metadata: small dict to store alongside the hash
                  (e.g. source_uri, filename, first_seen timestamp)
        expiry_seconds: TTL for the hash (default 30 days)

    Returns:
        True if the hash was newly set, False if it already existed
        (e.g. confirmed by another run) or on error.
    """
    try:
        client = get_redis_client()
        redis_key = f"{HASH_KEY_PREFIX}{content_hash}"
        was_new = client.set(
            redis_key,
            json.dumps(metadata).encode('utf-8'),
            ex=expiry_seconds,
            nx=True,
        )
        return bool(was_new)
    except redis.RedisError as e:
        logger.error(f"Redis error confirming document hash {content_hash}: {e}")
        return False


def clear_old_hashes(pattern: str = "rag:doc:hash:*"):
    """
    Clear document hashes from Redis. Use with caution — mainly for
    testing or manual cleanup (e.g. unsticking "ghost duplicate" hashes
    that were written by an older, buggy version of this pipeline that
    wrote to Redis before confirming a successful upsert).

    Args:
        pattern: Redis key pattern to match
    """
    try:
        client = get_redis_client()
        keys = client.keys(pattern)
        if keys:
            client.delete(*keys)
            logger.info(f"Cleared {len(keys)} hash keys from Redis")
        else:
            logger.info("No hash keys found to clear")
    except redis.RedisError as e:
        logger.error(f"Error clearing hashes: {e}")


def get_hash_stats() -> dict:
    """
    Get statistics about stored hashes.

    Returns:
        Dictionary with hash count and memory usage
    """
    try:
        client = get_redis_client()
        keys = client.keys(f"{HASH_KEY_PREFIX}*")
        total_memory = sum(client.memory_usage(k) or 0 for k in keys) if keys else 0
        return {
            'total_hashes': len(keys),
            'memory_used_bytes': total_memory,
        }
    except redis.RedisError as e:
        logger.error(f"Error getting hash stats: {e}")
        return {'total_hashes': 0, 'memory_used_bytes': 0}