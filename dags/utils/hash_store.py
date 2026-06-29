"""
dags/utils/hash_store.py

Document deduplication via SHA-256 content hashing stored in Redis.

Public API
──────────
compute_content_hash(content)              → str   (SHA-256 hex digest)
check_document_hash(redis_client, hash)    → bool  (True = duplicate, skip)
store_document_hash(redis_client, hash)    → None  (explicit store)
delete_document_hash(redis_client, hash)   → bool  (True = key existed)
"""

from __future__ import annotations

import hashlib
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Default TTL: 30 days in seconds
DEFAULT_TTL: int = 30 * 24 * 60 * 60   # 2 592 000 s

# Redis key prefix — namespace to avoid collisions with other apps
_KEY_PREFIX = "rag:doc_hash:"


# ─────────────────────────────────────────────────────────────────────────────
# Hash computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_content_hash(content: str) -> str:
    """Return the SHA-256 hex digest of *content*.

    Args:
        content: Raw document text (str).  The string is encoded as UTF-8
                 before hashing so results are deterministic across platforms.

    Returns:
        64-character lowercase hex string, e.g.
        ``'e3b0c44298fc1c149afb…'``
    """
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


# ─────────────────────────────────────────────────────────────────────────────
# Redis helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_key(content_hash: str) -> str:
    return f"{_KEY_PREFIX}{content_hash}"


def check_document_hash(
    redis_client,
    content_hash: str,
    ttl: int = DEFAULT_TTL,
) -> bool:
    """Check whether *content_hash* already exists in Redis.

    This is an **atomic check-and-store** operation using Redis ``SET NX``:

    * If the key does **not** exist → it is stored with *ttl* seconds expiry
      and ``False`` is returned (document is **new**, proceed with processing).
    * If the key **already exists** → ``True`` is returned (document is a
      **duplicate**, skip processing).

    Args:
        redis_client: An active ``redis.Redis`` (or compatible) client.
        content_hash: SHA-256 hex string from :func:`compute_content_hash`.
        ttl:          Key expiry in seconds (default 30 days).

    Returns:
        ``True``  — duplicate; the document has been seen before → **skip**.
        ``False`` — new document; hash stored; proceed with ingestion.

    Example::

        hash_val = compute_content_hash(doc_text)
        if check_document_hash(redis_client, hash_val):
            logger.info("Duplicate — skipping.")
        else:
            ingest(doc_text)
    """
    key = _make_key(content_hash)
    # SET key "1" EX ttl NX  — returns True when the key was NEW (set),
    # None/False when the key already existed (not set).
    was_new = redis_client.set(key, "1", ex=ttl, nx=True)
    if was_new:
        logger.debug("New document hash stored: %s", content_hash[:16])
        return False   # not a duplicate
    logger.debug("Duplicate document hash detected: %s", content_hash[:16])
    return True        # duplicate


def store_document_hash(
    redis_client,
    content_hash: str,
    ttl: int = DEFAULT_TTL,
) -> None:
    """Unconditionally store *content_hash* in Redis (overwrite if present).

    Use this when you want to mark a document as processed regardless of
    whether it existed before (e.g. after a forced re-index).

    Args:
        redis_client: An active ``redis.Redis`` (or compatible) client.
        content_hash: SHA-256 hex string from :func:`compute_content_hash`.
        ttl:          Key expiry in seconds (default 30 days).
    """
    key = _make_key(content_hash)
    redis_client.set(key, "1", ex=ttl)
    logger.debug("Document hash stored (forced): %s", content_hash[:16])


def delete_document_hash(
    redis_client,
    content_hash: str,
) -> bool:
    """Remove *content_hash* from Redis so the document can be re-ingested.

    Args:
        redis_client: An active ``redis.Redis`` (or compatible) client.
        content_hash: SHA-256 hex string from :func:`compute_content_hash`.

    Returns:
        ``True`` if the key existed and was deleted, ``False`` if not found.
    """
    key = _make_key(content_hash)
    deleted = redis_client.delete(key)
    existed = bool(deleted)
    if existed:
        logger.debug("Document hash deleted: %s", content_hash[:16])
    return existed