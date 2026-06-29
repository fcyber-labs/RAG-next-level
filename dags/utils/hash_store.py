"""
dags/utils/hash_store.py

Document deduplication via SHA-256 content hashing stored in Redis.

Public API
----------
compute_content_hash(content)                       -> str   SHA-256 hex digest
document_hash_exists(redis_client, content_hash)    -> bool  True = duplicate, skip
check_document_hash(redis_client, content_hash)     -> bool  atomic check-and-store
store_document_hash(redis_client, content_hash)     -> None  unconditional store
delete_document_hash(redis_client, content_hash)    -> bool  True = key existed
"""

from __future__ import annotations

import hashlib
import logging

logger = logging.getLogger(__name__)

DEFAULT_TTL: int = 30 * 24 * 60 * 60  # 30 days in seconds
_KEY_PREFIX = "rag:doc_hash:"


def _make_key(content_hash: str) -> str:
    return f"{_KEY_PREFIX}{content_hash}"


# ---------------------------------------------------------------------------
# Hash computation
# ---------------------------------------------------------------------------

def compute_content_hash(content: str) -> str:
    """Return the SHA-256 hex digest of *content* (UTF-8 encoded).

    Args:
        content: Raw document text.

    Returns:
        64-character lowercase hex string.
    """
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Redis helpers
# ---------------------------------------------------------------------------

def document_hash_exists(redis_client, content_hash: str) -> bool:
    """Check whether *content_hash* already exists in Redis (read-only).

    Used by deduplicate_documents() to decide whether to skip a document.
    Does NOT store the hash — call store_document_hash() separately after
    processing a new document.

    Args:
        redis_client: Active ``redis.Redis`` (or compatible) client.
        content_hash: SHA-256 hex string from compute_content_hash().

    Returns:
        True  — hash found → document is a duplicate, skip it.
        False — hash not found → document is new, process it.
    """
    key = _make_key(content_hash)
    exists = bool(redis_client.exists(key))
    if exists:
        logger.debug("Duplicate hash detected: %s", content_hash[:16])
    return exists


def check_document_hash(
    redis_client,
    content_hash: str,
    ttl: int = DEFAULT_TTL,
) -> bool:
    """Atomic check-and-store: check if hash exists, store if new.

    Unlike document_hash_exists(), this combines the check and the store
    in one atomic Redis SET NX operation.

    Returns:
        True  — duplicate (already existed, not stored again).
        False — new document (hash just stored with *ttl* seconds expiry).
    """
    key = _make_key(content_hash)
    was_new = redis_client.set(key, "1", ex=ttl, nx=True)
    if was_new:
        logger.debug("New hash stored: %s", content_hash[:16])
        return False
    logger.debug("Duplicate hash detected: %s", content_hash[:16])
    return True


def store_document_hash(
    redis_client,
    content_hash: str,
    ttl: int = DEFAULT_TTL,
) -> None:
    """Unconditionally store *content_hash* in Redis (overwrite if present).

    Call this after successfully processing a new document.

    Args:
        redis_client: Active ``redis.Redis`` client.
        content_hash: SHA-256 hex string from compute_content_hash().
        ttl:          Expiry in seconds (default 30 days).
    """
    key = _make_key(content_hash)
    redis_client.set(key, "1", ex=ttl)
    logger.debug("Hash stored: %s", content_hash[:16])


def delete_document_hash(redis_client, content_hash: str) -> bool:
    """Remove *content_hash* from Redis so the document can be re-ingested.

    Returns:
        True if the key existed and was deleted, False if not found.
    """
    key = _make_key(content_hash)
    return bool(redis_client.delete(key))