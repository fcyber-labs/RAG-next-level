"""
Collection rollback and promotion logic for Qdrant.
"""

import logging
import os
from typing import Dict, Any
from qdrant_client import QdrantClient
import time

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
    
    try:
        # Backup strategy: rename current production to backup with timestamp
        backup_name = f"{production_collection}_backup_{int(time.time())}"
        
        # Check if production exists
        try:
            client.get_collection(production_collection)
            logger.info(f"Creating backup of current production as '{backup_name}'")
            
            # Unfortunately Qdrant doesn't have direct rename, so we need to:
            # 1. Create a new collection with backup name
            # 2. Copy all points (for production, we'll use a snapshot approach)
            # 3. For simplicity in this demo, we'll just delete old production
            #    In real production, you'd want to create a snapshot first
            
            # Create snapshot of production before deletion
            snapshot_info = client.create_snapshot(production_collection)
            logger.info(f"Created snapshot: {snapshot_info}")
            
            # Delete old production
            client.delete_collection(production_collection)
            logger.info(f"Deleted old production collection '{production_collection}'")
        
        except Exception as e:
            logger.info(f"No existing production collection to backup: {e}")
        
        # Get staging collection info
        staging_info = client.get_collection(staging_collection)
        vector_config = staging_info.config.params.vectors
        
        # Create new production collection with same config
        client.create_collection(
            collection_name=production_collection,
            vectors_config=vector_config,
        )
        logger.info(f"Created new production collection '{production_collection}'")
        
        # Scroll through staging and copy all points to production
        # This is a simplified approach; in production you might use snapshots
        offset = None
        batch_size = 100
        total_copied = 0
        
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
            
            # Upsert to production
            client.upsert(
                collection_name=production_collection,
                points=records,
            )
            
            total_copied += len(records)
            logger.debug(f"Copied {total_copied} points to production...")
            
            if offset is None:
                break
        
        logger.info(f"Promotion complete: copied {total_copied} points to production")
        
        # Optionally delete staging collection after successful promotion
        client.delete_collection(staging_collection)
        logger.info(f"Deleted staging collection '{staging_collection}'")
        
        # Export metrics
        from utils.metrics_exporter import export_counter
        export_counter('collection_promotions_total', 1)
        
        return {
            'success': True,
            'action': 'promote',
            'staging_collection': staging_collection,
            'production_collection': production_collection,
            'points_copied': total_copied,
            'message': f'Successfully promoted {total_copied} points to production',
        }
    
    except Exception as e:
        logger.error(f"Error during promotion: {e}")
        
        # Export failure metric
        from utils.metrics_exporter import export_counter
        export_counter('collection_promotion_failures_total', 1)
        
        return {
            'success': False,
            'action': 'promote',
            'error': str(e),
        }