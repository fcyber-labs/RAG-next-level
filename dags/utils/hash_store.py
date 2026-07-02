"""
dags/utils/hash_store.py

Document deduplication via SHA-256 content hashing stored in Redis.

Design: CHECK vs CONFIRM are deliberately separate operations.
--------------------------------------------------------------
document_hash_exists() is READ-ONLY. It never writes to Redis.
confirm_document_hash() is the ONLY function that writes, and callers
must only invoke it AFTER a document has been fully and successfully
processed (chunked, embedded, and upserted into Qdrant).

This two-step design prevents "ghost duplicates": if the write happened
at check-time (like a naive SETNX-on-first-sight), a mid-pipeline crash
(e.g. embedding API failure, Qdrant outage) would leave Redis believing
a document was stored when it never was — causing every future run to
silently skip it for the full TTL window while the knowledge base stays
empty for that document. See dags/tasks/deduplicate.py's module docstring
for the full rationale.

Public API
----------
compute_content_hash(content)          -> str   SHA-256 hex digest
get_redis_client()                     -> Redis  lazy module-level singleton
document_hash_exists(content_hash)     -> bool   READ-ONLY check
confirm_document_hash(content_hash)    -> None   WRITE — call only after success
delete_document_hash(content_hash)     -> bool   admin/rollback use
"""

from __future__ import annotations

import hashlib
import logging
import os

logger = logging.getLogger(__name__)

DEFAULT_TTL: int = 30 * 24 * 60 * 60  # 30 days in seconds
_KEY_PREFIX = "rag:doc_hash:"

# Lazily-created module-level Redis client. Tests patch
# `dags.utils.hash_store.get_redis_client` directly to avoid needing a
# live Redis instance.
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
    """
    global _redis_client
    if _redis_client is None:
        import redis  # lazy import so this module stays importable without
        # the redis package present (e.g. in DAG-syntax-only checks)
        _redis_client = redis.Redis(
            host=os.environ.get("REDIS_HOST", "localhost"),
            port=int(os.environ.get("REDIS_PORT", "6379")),
            db=int(os.environ.get("REDIS_DB", "0")),
            decode_responses=False,
        )
    return _redis_client


# ---------------------------------------------------------------------------
# Check (read-only) / Confirm (write) — see module docstring
# ---------------------------------------------------------------------------

def document_hash_exists(content_hash: str) -> bool:
    """READ-ONLY check: does *content_hash* already exist in Redis?

    Does NOT write anything. Used by deduplicate_documents() to decide
    whether a document is a candidate-new document or an already-confirmed
    duplicate.

    Any Redis error (connection refused, timeout, etc.) is caught and
    treated as "not seen" (returns False) rather than propagating — a
    Redis outage must never silently drop a document from processing.

    Args:
        content_hash: SHA-256 hex string from compute_content_hash().

    Returns:
        True  — hash found -> document was previously confirmed stored.
        False — hash not found, OR Redis is unreachable -> document should
                be treated as new and processed.
    """
    client = get_redis_client()
    key = _make_key(content_hash)
    try:
        exists = bool(client.exists(key))
    except Exception as e:
        logger.warning(
            "Redis error checking hash existence, defaulting to False "
            "(treat as new so the document still gets processed): %s", e
        )
        return False
    if exists:
        logger.debug("Hash already confirmed: %s", content_hash[:16])
    return exists


def confirm_document_hash(content_hash: str, metadata: dict = None, ttl: int = DEFAULT_TTL) -> bool:
    """WRITE *content_hash* to Redis. Call ONLY after a confirmed success.

    This should only be invoked after the document has been fully
    processed AND successfully upserted into Qdrant — never at
    deduplication-check time. Calling this too early risks marking a
    document as "seen" when it was never actually stored (a "ghost
    duplicate").

    Uses Redis SET NX so an already-confirmed hash is never overwritten —
    the operation is idempotent and safe to call more than once for the
    same document.

    Args:
        content_hash: SHA-256 hex string from compute_content_hash().
        metadata:     Optional dict (e.g. {'filename': ...}) stored
                      alongside the confirmation, for debugging/audit.
        ttl:          Expiry in seconds (default 30 days).

    Returns:
        True  — this call newly confirmed the hash (it was not previously
                confirmed).
        False — the hash was already confirmed; this call was a no-op.
    """
    import json

    client = get_redis_client()
    key = _make_key(content_hash)
    value = json.dumps(metadata) if metadata else "1"

    was_new = client.set(key, value, ex=ttl, nx=True)

    if was_new:
        logger.debug("Hash confirmed and stored: %s", content_hash[:16])
    else:
        logger.debug("Hash already confirmed (no-op): %s", content_hash[:16])

    return bool(was_new)


def delete_document_hash(content_hash: str) -> bool:
    """Remove *content_hash* from Redis so the document can be re-ingested.

    Intended for admin/rollback use (e.g. dags/tasks/rollback.py).

    Returns:
        True if the key existed and was deleted, False if not found.
    """
    client = get_redis_client()
    key = _make_key(content_hash)
    return bool(client.delete(key))