"""
Simple Qdrant vector store helpers.
Provides connection and common operations for the RAG pipeline.
"""

import logging
import os
from typing import List, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

logger = logging.getLogger(__name__)


def get_qdrant_client() -> QdrantClient:
    """
    Get a Qdrant client connection.
    
    Returns:
        QdrantClient instance
    """
    host = os.getenv('QDRANT_HOST', 'localhost')
    port = int(os.getenv('QDRANT_PORT', 6333))
    
    return QdrantClient(host=host, port=port)


def create_collection_if_not_exists(
    collection_name: str,
    vector_size: int = 384,
    distance: Distance = Distance.COSINE
):
    """
    Create a Qdrant collection if it doesn't exist.
    
    Args:
        collection_name: Name of the collection
        vector_size: Dimension of vectors
        distance: Distance metric (COSINE, EUCLID, DOT)
    """
    try:
        client = get_qdrant_client()
        
        # Check if collection exists
        try:
            client.get_collection(collection_name)
            logger.info(f"Collection '{collection_name}' already exists")
            return
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            pass
        
        # Create collection
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=distance,
            ),
        )
        
        logger.info(f"Created collection '{collection_name}' with size {vector_size}")
    
    except Exception as e:
        logger.error(f"Error creating collection: {e}")
        raise


def search_similar(
    collection_name: str,
    query_vector: List[float],
    limit: int = 10,
    score_threshold: float = 0.0
) -> List[Dict[str, Any]]:
    """
    Search for similar vectors in Qdrant.
    
    Args:
        collection_name: Collection to search
        query_vector: Query embedding vector
        limit: Number of results to return
        score_threshold: Minimum similarity score
        
    Returns:
        List of search results with payload and score
    """
    try:
        client = get_qdrant_client()
        
        results = client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=limit,
            score_threshold=score_threshold,
        )
        
        # Format results
        formatted_results = []
        for result in results:
            formatted_results.append({
                'id': result.id,
                'score': result.score,
                'payload': result.payload,
            })
        
        return formatted_results
    
    except Exception as e:
        logger.error(f"Error searching vectors: {e}")
        return []


def get_collection_info(collection_name: str) -> Dict[str, Any]:
    """
    Get information about a collection.
    
    Args:
        collection_name: Name of the collection
        
    Returns:
        Dictionary with collection stats
    """
    try:
        client = get_qdrant_client()
        
        info = client.get_collection(collection_name)
        
        return {
            'name': collection_name,
            'points_count': info.points_count,
            'vectors_count': info.vectors_count,
            'status': info.status,
        }
    
    except Exception as e:
        logger.error(f"Error getting collection info: {e}")
        return {}


def delete_collection(collection_name: str):
    """
    Delete a collection from Qdrant.
    
    Args:
        collection_name: Name of collection to delete
    """
    try:
        client = get_qdrant_client()
        client.delete_collection(collection_name)
        logger.info(f"Deleted collection '{collection_name}'")
    
    except Exception as e:
        logger.error(f"Error deleting collection: {e}")