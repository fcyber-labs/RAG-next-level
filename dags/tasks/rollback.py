"""
Collection rollback and promotion logic for Qdrant.
"""

import logging
import os
from typing import Dict, Any
from qdrant_client import QdrantClient
import time


from qdrant_client.models import PointStruct 

logger = logging.getLogger(__name__)


def _get_qdrant_client() -> QdrantClient:
    """Get Qdrant client connection."""
    host = os.getenv('QDRANT_HOST', 'localhost')
    port = int(os.getenv('QDRANT_PORT', 6333))
    return QdrantClient(host=host, port=port)


def rollback_collection(
    staging_collection: str = "knowledge_base_staging",
    **kwargs
) -> Dict[str, Any]:
    """
    Rollback staging collection by deleting it.
    Production collection remains unchanged.
    
    Args:
        staging_collection: Name of staging collection to rollback
        
    Returns:
        Rollback status dictionary
    """
    logger.warning(f"Rolling back collection '{staging_collection}' due to quality check failure")
    
    client = _get_qdrant_client()
    
    try:
        # Delete the staging collection
        client.delete_collection(staging_collection)
        logger.info(f"Successfully deleted staging collection '{staging_collection}'")

        # The staging collection no longer exists — drop any cached BM25
        # chunk list and cached LLM answers for it so the Streamlit app
        # doesn't serve stale data for a collection that's gone.
        try:
            from utils.chunk_cache import invalidate_chunk_cache
            from utils.answer_cache import invalidate_answer_cache
            invalidate_chunk_cache(staging_collection)
            invalidate_answer_cache(staging_collection)
        except Exception as e:
            logger.warning(f"Could not invalidate caches for '{staging_collection}': {e}")
        
        return {
            'success': True,
            'action': 'rollback',
            'collection': staging_collection,
            'message': 'Staging collection deleted, production unchanged',
        }
    
    except Exception as e:
        logger.error(f"Error during rollback: {e}")
        return {
            'success': False,
            'action': 'rollback',
            'collection': staging_collection,
            'error': str(e),
        }


 

def promote_collection(
    staging_collection: str = "knowledge_base_staging",
    production_collection: str = "knowledge_base",
    **kwargs
) -> Dict[str, Any]:
    """
    Promote staging collection to production.
    
    Strategy:
    1. Create snapshot of current production (backup)
    2. Delete production collection
    3. Rename staging to production
    
    Args:
        staging_collection: Source collection (staging)
        production_collection: Target collection (production)
        
    Returns:
        Promotion status dictionary
    """
    logger.info(f"Promoting '{staging_collection}' to '{production_collection}'")
    
    client = _get_qdrant_client()
    backup_name = f"{production_collection}_backup_{int(time.time())}"
    
    # ===== STEP 1: Handle existing production collection =====
    production_exists = False
    try:
        client.get_collection(production_collection)
        production_exists = True
        logger.info(f"Production collection '{production_collection}' exists")
    except Exception as e:
        # Only catch 404 (Not Found) - production doesn't exist
        if "Not found" in str(e) or "404" in str(e):
            logger.info(f"No existing production collection found")
        else:
            # Some other error (connection, auth, etc.) - re-raise
            logger.error(f"Error checking production collection: {e}")
            raise
    
    # ===== STEP 2: Backup existing production =====
    if production_exists:
        try:
            logger.info(f"Creating backup of current production as snapshot '{backup_name}'")
            snapshot_info = client.create_snapshot(production_collection)
            logger.info(f"Created snapshot: {snapshot_info}")
        except Exception as e:
            logger.warning(f"Failed to create snapshot of production: {e}")
            # Continue with promotion (snapshot failure shouldn't block)
        
        # ===== STEP 3: Delete existing production =====
        try:
            client.delete_collection(production_collection)
            logger.info(f"Deleted old production collection '{production_collection}'")
        except Exception as e:
            logger.error(f"Failed to delete production collection: {e}")
            raise RuntimeError(f"Cannot promote: production deletion failed: {e}")
    
    # ===== STEP 4: Get staging config =====
    try:
        staging_info = client.get_collection(staging_collection)
        vector_config = staging_info.config.params.vectors
    except Exception as e:
        logger.error(f"Failed to get staging collection info: {e}")
        raise
    
    # ===== STEP 5: Create new production =====
    try:
        client.create_collection(
            collection_name=production_collection,
            vectors_config=vector_config,
        )
        logger.info(f"Created new production collection '{production_collection}'")
    except Exception as e:
        logger.error(f"Failed to create production collection: {e}")
        raise
    
    # ===== STEP 6: Copy points from staging to production =====
    offset = None
    batch_size = 100
    total_copied = 0
    
    try:
        while True:
            records, offset = client.scroll(
                collection_name=staging_collection,
                limit=batch_size,
                offset=offset,
                with_payload=True,
                with_vectors=True,
            )
            
            if not records:
                break
            
            # FIX: Convert Record objects to PointStruct objects
            points = [
                PointStruct(
                    id=record.id,
                    vector=record.vector,
                    payload=record.payload
                )
                for record in records
            ]
            
            client.upsert(
                collection_name=production_collection,
                points=points,  # Now passing PointStruct, not Record
            )
            
            total_copied += len(records)
            logger.debug(f"Copied {total_copied} points to production...")
            
            if offset is None:
                break
    except Exception as e:
        logger.error(f"Failed to copy points from staging to production: {e}")
        raise
    
    logger.info(f"Promotion complete: copied {total_copied} points to production")
    
    # ===== STEP 7: Clean up staging =====
    try:
        client.delete_collection(staging_collection)
        logger.info(f"Deleted staging collection '{staging_collection}'")
    except Exception as e:
        logger.warning(f"Failed to delete staging collection: {e}")
    
    # Export metrics
    from utils.metrics_exporter import export_counter
    export_counter('collection_promotions_total', 1)

    # Production now holds entirely new points, and staging no longer
    # exists — drop any cached BM25 chunk lists and cached LLM answers
    # for both so the next Streamlit search rebuilds from the current
    # state instead of serving stale data.
    try:
        from utils.chunk_cache import invalidate_chunk_cache
        from utils.answer_cache import invalidate_answer_cache
        invalidate_chunk_cache(production_collection)
        invalidate_chunk_cache(staging_collection)
        invalidate_answer_cache(production_collection)
        invalidate_answer_cache(staging_collection)
    except Exception as e:
        logger.warning(f"Could not invalidate caches after promotion: {e}")
    
    return {
        'success': True,
        'action': 'promote',
        'staging_collection': staging_collection,
        'production_collection': production_collection,
        'points_copied': total_copied,
        'message': f'Successfully promoted {total_copied} points to production',
    }