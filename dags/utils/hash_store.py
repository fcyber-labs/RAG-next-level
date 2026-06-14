"""
Simple Redis helpers for document deduplication.
Uses content hashing with SETNX for fast duplicate detection.
"""

import logging
import os
import redis
import hashlib
import json

logger = logging.getLogger(__name__)


def get_redis_client():
    """
    Get a Redis client connection.
    
    Returns:
        Redis client instance
    """
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


def check_document_hash(content: str, metadata: dict, expiry_seconds: int = 2592000) -> bool:
    """
    Check if document hash exists in Redis. If not, store it.
    
    Args:
        content: Document content
        metadata: Document metadata to store
        expiry_seconds: TTL for hash (default 30 days)
        
    Returns:
        True if document is new, False if it's a duplicate
    """
    try:
        client = get_redis_client()
        
        # Compute hash
        content_hash = compute_content_hash(content)
        redis_key = f"rag:doc:hash:{content_hash}"
        
        # Try to set with NX (only if not exists)
        is_new = client.set(
            redis_key,
            json.dumps(metadata).encode('utf-8'),
            ex=expiry_seconds,
            nx=True
        )
        
        return bool(is_new)
    
    except Exception as e:
        logger.error(f"Error checking document hash: {e}")
        # On error, assume document is new to be safe
        return True


def clear_old_hashes(pattern: str = "rag:doc:hash:*"):
    """
    Clear old document hashes from Redis.
    Use with caution - mainly for testing or manual cleanup.
    
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
    
    except Exception as e:
        logger.error(f"Error clearing hashes: {e}")


def get_hash_stats() -> dict:
    """
    Get statistics about stored hashes.
    
    Returns:
        Dictionary with hash count and memory usage
    """
    try:
        client = get_redis_client()
        
        keys = client.keys("rag:doc:hash:*")
        
        return {
            'total_hashes': len(keys),
            'memory_used_bytes': client.memory_usage("rag:doc:hash:*") if keys else 0,
        }
    
    except Exception as e:
        logger.error(f"Error getting hash stats: {e}")
        return {'total_hashes': 0, 'memory_used_bytes': 0}