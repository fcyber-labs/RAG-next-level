"""
Upsert vectors to Qdrant vector database.
ENHANCED: Now includes document expiration metadata.
"""

import logging
import os
from typing import Dict, Any
import time
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
)
from qdrant_client.http.exceptions import UnexpectedResponse
import uuid
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


def _get_qdrant_client() -> QdrantClient:
    """Get Qdrant client connection."""
    host = os.getenv('QDRANT_HOST', 'localhost')
    port = int(os.getenv('QDRANT_PORT', 6333))
    
    return QdrantClient(host=host, port=port)


def _ensure_collection(
    client: QdrantClient,
    collection_name: str,
    vector_dimension: int,
) -> None:
    """Ensure collection exists, create if not."""
    try:
        client.get_collection(collection_name)
        logger.info(f"Collection '{collection_name}' already exists")
    except (UnexpectedResponse, Exception):
        logger.info(f"Creating collection '{collection_name}' with dimension {vector_dimension}")
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=vector_dimension,
                distance=Distance.COSINE,
            ),
        )


def upsert_to_qdrant(
    embedded_chunks: Any,
    collection_name: str = "knowledge_base_staging",
    batch_size: int = 100,
    expiration_days: int = 365,  # NEW: Default 1 year expiration
    **kwargs
) -> Dict[str, Any]:
    """
    Upsert embedded chunks to Qdrant vector database.
    
    ENHANCED: Now adds expiration metadata for document lifecycle management.
    
    Args:
        embedded_chunks: List of chunk dictionaries with embeddings
        collection_name: Qdrant collection name
        batch_size: Number of points to upsert per batch
        expiration_days: Days until document expires (0 = never)
        
    Returns:
        Summary statistics
    """
    # Handle XCom input
    if isinstance(embedded_chunks, str):
        try:
            embedded_chunks = eval(embedded_chunks)
        except Exception as e:
            logger.error(f"Could not parse embedded_chunks from XCom: {e}")
            return {'success': False, 'points_upserted': 0}
    
    if not embedded_chunks:
        logger.warning("No embedded chunks to upsert")
        return {'success': False, 'points_upserted': 0}
    
    # Parse expiration_days if string
    if isinstance(expiration_days, str):
        expiration_days = int(expiration_days)
    
    logger.info(f"Starting upsert of {len(embedded_chunks)} chunks to collection '{collection_name}'")
    logger.info(f"Document expiration: {expiration_days} days (0 = never)")
    
    client = _get_qdrant_client()
    start_time = time.time()
    
    # Get vector dimension from first chunk
    vector_dim = embedded_chunks[0].get('embedding_dimension', 1536)
    
    # Ensure collection exists
    _ensure_collection(client, collection_name, vector_dim)
    
    # Calculate expiration timestamp
    if expiration_days > 0:
        expires_at = datetime.now() + timedelta(days=expiration_days)
        expires_at_iso = expires_at.isoformat()
    else:
        expires_at_iso = None  # Never expires
    
    # Prepare points for upsert
    points = []
    for chunk in embedded_chunks:
        point_id = str(uuid.uuid4())
        
        point = PointStruct(
            id=point_id,
            vector=chunk['embedding'],
            payload={
                'text': chunk['text'],
                'source': chunk['source'],
                'source_uri': chunk['source_uri'],
                'filename': chunk['filename'],
                'content_hash': chunk.get('content_hash', ''),
                'chunk_index': chunk['chunk_index'],
                'total_chunks': chunk['total_chunks'],
                'embedding_model': chunk['embedding_model'],
                'metadata': chunk.get('metadata', {}),
                'ingestion_timestamp': kwargs.get('ts', datetime.now().isoformat()),
                'expires_at': expires_at_iso,  # NEW: Expiration timestamp
            }
        )
        points.append(point)
    
    # Upsert in batches
    points_upserted = 0
    for i in range(0, len(points), batch_size):
        batch = points[i:i + batch_size]
        
        try:
            client.upsert(
                collection_name=collection_name,
                points=batch,
            )
            points_upserted += len(batch)
            logger.debug(f"Upserted batch {i//batch_size + 1}/{(len(points)-1)//batch_size + 1}")
        
        except Exception as e:
            logger.error(f"Error upserting batch starting at index {i}: {e}")
            continue
    
    elapsed_time = time.time() - start_time
    
    logger.info(
        f"Upsert complete: {points_upserted} points in {elapsed_time:.2f}s "
        f"({points_upserted/elapsed_time:.2f} points/sec)"
    )
    
    # Get collection info
    try:
        collection_info = client.get_collection(collection_name)
        total_points = collection_info.points_count
        logger.info(f"Collection '{collection_name}' now contains {total_points} total points")
    except Exception as e:
        logger.error(f"Error getting collection info: {e}")
        total_points = points_upserted
    
    # Export metrics
    from utils.metrics_exporter import export_counter, export_histogram, export_gauge
    export_counter('vectors_upserted_total', points_upserted)
    export_histogram('upsert_latency_seconds', elapsed_time)
    export_gauge('collection_total_points', total_points)
    
    return {
        'success': True,
        'points_upserted': points_upserted,
        'collection_name': collection_name,
        'total_points': total_points,
        'elapsed_time': elapsed_time,
        'expiration_days': expiration_days,
        'expires_at': expires_at_iso,
    }