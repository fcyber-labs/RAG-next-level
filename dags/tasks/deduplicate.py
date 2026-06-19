"""
Document deduplication using Redis-based content hashing.
"""

import logging
import hashlib
from typing import List, Dict, Any
import redis
import os
import json
import ast

logger = logging.getLogger(__name__)


def _get_redis_client():
    """Get Redis client connection."""
    redis_host = os.getenv('REDIS_HOST', 'localhost')
    redis_port = int(os.getenv('REDIS_PORT', 6379))
    
    return redis.Redis(
        host=redis_host,
        port=redis_port,
        db=0,
        decode_responses=False,  # We'll handle encoding
    )


def _compute_content_hash(content: str) -> str:
    """
    Compute SHA-256 hash of document content.
    
    Args:
        content: Document text content
        
    Returns:
        Hex digest of content hash
    """
    return hashlib.sha256(content.encode('utf-8')).hexdigest()


def deduplicate_documents(documents: Any, **kwargs) -> List[Dict[str, Any]]:
    """
    Deduplicate documents using Redis-based content hashing.
    Only returns new or modified documents.
    
    Args:
        documents: List of document dictionaries or XCom reference
        
    Returns:
        List of unique/new documents
    """
    # Handle XCom pull
    if isinstance(documents, str):
        try:
            documents = ast.literal_eval(documents)
        except Exception as e:
            raise RuntimeError(
                f"deduplicate_documents could not parse its 'documents' XCom argument. "
                f"The value from extract_all_sources was not a valid Python literal. "
                f"Parse error: {e}"
            ) from e
    
    if not documents:
        logger.warning("No documents to deduplicate")
        return []
    
    logger.info(f"Starting deduplication for {len(documents)} documents")
    
    redis_client = _get_redis_client()
    new_documents = []
    duplicate_count = 0
    
    # Redis key prefix for document hashes
    HASH_KEY_PREFIX = "rag:doc:hash:"
    HASH_EXPIRY = 60 * 60 * 24 * 30  # 30 days
    
    for doc in documents:
        content = doc.get('content', '')
        if not content:
            continue
        
        # Compute content hash
        content_hash = _compute_content_hash(content)
        doc['content_hash'] = content_hash
        
        # Check if hash exists in Redis
        redis_key = f"{HASH_KEY_PREFIX}{content_hash}"
        
        try:
            # Try to set the hash with NX (only if not exists)
            # Returns 1 if key was set, 0 if key already existed
            is_new = redis_client.set(
                redis_key,
                json.dumps({
                    'source_uri': doc.get('source_uri', ''),
                    'filename': doc.get('filename', ''),
                    'first_seen': kwargs.get('ts', ''),
                }).encode('utf-8'),
                ex=HASH_EXPIRY,
                nx=True  # Only set if not exists
            )
            
            if is_new:
                new_documents.append(doc)
                logger.debug(f"New document: {doc.get('filename', 'unknown')}")
            else:
                duplicate_count += 1
                logger.debug(f"Duplicate document (skipped): {doc.get('filename', 'unknown')}")
        
        except redis.RedisError as e:
            logger.error(f"Redis error during deduplication: {e}")
            # On Redis error, include the document to be safe
            new_documents.append(doc)
    
    logger.info(
        f"Deduplication complete: {len(new_documents)} new, "
        f"{duplicate_count} duplicates skipped"
    )
    
    # Export metrics
    
    from utils.metrics_exporter import export_counter, export_gauge
    from utils.metadata_db import record_documents
    record_documents(new_documents)
    export_counter('documents_deduplicated_new', len(new_documents))
    export_counter('documents_deduplicated_skipped', duplicate_count)
    export_gauge('deduplication_cache_hit_rate', 
                 duplicate_count / len(documents) if documents else 0)
    
    return new_documents