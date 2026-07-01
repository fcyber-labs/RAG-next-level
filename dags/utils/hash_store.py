"""
dags/utils/hash_store.py

Document deduplication via SHA-256 content hashing stored in Redis.

Public API
----------
compute_content_hash(content)                    -> str   SHA-256 hex digest
get_redis_client()                                -> Redis client (lazy singleton)
document_hash_exists(redis_client, content_hash) -> bool  True = duplicate
check_document_hash(content_hash)                -> bool  atomic check-and-store
store_document_hash(redis_client, content_hash)  -> None  unconditional store
delete_document_hash(redis_client, content_hash) -> bool  True = key existed
"""

from __future__ import annotations

import hashlib
import logging
import os

logger = logging.getLogger(__name__)

DEFAULT_TTL: int = 30 * 24 * 60 * 60  # 30 days in seconds
_KEY_PREFIX = "rag:doc_hash:"

# Lazily-created module-level Redis client, so tests can patch
# `dags.utils.hash_store.get_redis_client` without needing a live Redis.
_redis_client = None


def _make_key(content_hash: str) -> str:
    return f"{_KEY_PREFIX}{content_hash}"


# ---------------------------------------------------------------------------
# Hash computation
# ---------------------------------------------------------------------------

def compute_content_hash(content: str) -> str:
    """Return the SHA-256 hex digest of *content* (UTF-8 encoded)."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Redis client
# ---------------------------------------------------------------------------

def get_redis_client():
    """Return a lazily-created, module-level Redis client.

    Reads connection info from env vars REDIS_HOST / REDIS_PORT / REDIS_DB.
    Tests typically patch this function directly (e.g.
    ``mocker.patch("dags.utils.hash_store.get_redis_client")``) to avoid
    needing a real Redis instance.
    """
    global _redis_client
    if _redis_client is None:
        import redis  # imported lazily so this module is importable
        # without the redis package present (e.g. in DAG-syntax-only checks)
        _redis_client = redis.Redis(
            host=os.environ.get("REDIS_HOST", "localhost"),
            port=int(os.environ.get("REDIS_PORT", "6379")),
            db=int(os.environ.get("REDIS_DB", "0")),
            decode_responses=False,
        )
    return _redis_client


# ---------------------------------------------------------------------------
# Redis helpers
# ---------------------------------------------------------------------------

def document_hash_exists(redis_client, content_hash: str) -> bool:
    """Check whether *content_hash* already exists in Redis (read-only).

    Args:
        redis_client: Active ``redis.Redis`` (or compatible) client.
        content_hash: SHA-256 hex string from compute_content_hash().

    Returns:
        True  — hash found -> document is a duplicate, skip it.
        False — hash not found -> document is new, process it.
    """
    key = _make_key(content_hash)
    exists = bool(redis_client.exists(key))
    if exists:
        logger.debug("Duplicate hash detected: %s", content_hash[:16])
    return exists


def check_document_hash(content_hash: str, ttl: int = DEFAULT_TTL) -> bool:
    """Atomic check-and-store using the module-level Redis client.

    Internally calls get_redis_client() so callers don't need to manage
    a client themselves. Uses Redis SET NX for an atomic check-and-store.

    Returns:
        True  — duplicate (already existed, not stored again).
        False — new document (hash just stored with *ttl* seconds expiry).
    """
    client = get_redis_client()
    key = _make_key(content_hash)
    was_new = client.set(key, "1", ex=ttl, nx=True)
    if was_new:
        logger.debug("New hash stored: %s", content_hash[:16])
        return False
    logger.debug("Duplicate hash detected: %s", content_hash[:16])
    return True


def store_document_hash(redis_client, content_hash: str, ttl: int = DEFAULT_TTL) -> None:
    """Unconditionally store *content_hash* in Redis (overwrite if present)."""
    key = _make_key(content_hash)
    redis_client.set(key, "1", ex=ttl)
    logger.debug("Hash stored: %s", content_hash[:16])


def delete_document_hash(redis_client, content_hash: str) -> bool:
    """Remove *content_hash* from Redis so the document can be re-ingested."""
    key = _make_key(content_hash)
    return bool(redis_client.delete(key))