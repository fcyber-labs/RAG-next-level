"""
Upsert vectors to Qdrant vector database.
ENHANCED: Now includes document expiration metadata.

ENHANCED (Redis hash confirmation): document hashes are now confirmed in
Redis HERE, after a successful upsert — never earlier. See
utils.hash_store and dags/tasks/deduplicate.py for the full explanation
of why the write moved here.
"""

import logging
import os
from typing import Dict, Any, List
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
import ast

from utils.hash_store import confirm_document_hash

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
            embedded_chunks = ast.literal_eval(embedded_chunks)
        except Exception as e:
            raise RuntimeError(
                f"upsert_to_qdrant could not parse its 'embedded_chunks' XCom argument. "
                f"The value from embed_chunks was not a valid Python literal. "
                f"Parse error: {e}"
            ) from e
    
    if not embedded_chunks:
        # `embedded_chunks` is empty for two very different reasons, and the
        # old code treated them identically (hard failure):
        #
        #   (a) STEADY STATE — this DAG runs every 6h (schedule_interval=
        #       '0 */6 * * *'). deduplicate_documents hashes every source
        #       document into Redis (30-day TTL) and only lets *new/changed*
        #       documents through. On most scheduled runs nothing in the
        #       source has changed since the last run, so dedupe correctly
        #       returns 0 documents, which correctly chains to 0 chunks and
        #       0 embeddings. That is NOT a failure — it's the pipeline
        #       working as designed. The knowledge base collection already
        #       holds everything from previous successful runs.
        #
        #   (b) GENUINE FAILURE — the very first ingestion run (collection
        #       never created / never populated) produces 0 usable chunks,
        #       e.g. because every source file extracted to empty text.
        #       There is nothing in Qdrant yet, so this *is* a real problem
        #       that should fail loudly.
        #
        # We tell these apart by checking whether the target collection
        # already has points in it.
        client = _get_qdrant_client()
        total_points = 0
        collection_already_populated = False
        try:
            collection_info = client.get_collection(collection_name)
            total_points = collection_info.points_count
            collection_already_populated = total_points > 0
        except Exception:
            pass  # collection doesn't exist yet -> definitely not case (a)

        if collection_already_populated:
            logger.info(
                f"No new/changed chunks this cycle — collection "
                f"'{collection_name}' already holds {total_points} points "
                f"from previous runs. Nothing to upsert; treating this as "
                f"a successful no-op so downstream evaluation can run "
                f"against the existing collection."
            )
            from utils.metrics_exporter import export_counter
            export_counter('upsert_cycles_skipped_no_new_documents', 1)
            return {
                'success': True,
                'points_upserted': 0,
                'collection_name': collection_name,
                'total_points': total_points,
                'elapsed_time': 0.0,
                'expiration_days': expiration_days,
                'expires_at': None,
                'skipped_no_new_documents': True,
            }

        # Collection is missing/empty AND we got 0 embedded chunks: this is
        # a genuine first-ingestion failure, not a benign re-run. Raise —
        # not a silent return. Returning {'success': False} without raising
        # means Airflow marks this task SUCCESS and the downstream
        # run_retrieval_eval starts, finds no collection in Qdrant, and
        # crashes with a confusing 404. Raising here fails the task at the
        # right place with the right message and prevents the downstream crash.
        raise ValueError(
            "upsert_to_qdrant received no embedded chunks and collection "
            f"'{collection_name}' does not exist yet (or is empty), so this "
            "isn't a benign 'nothing changed' re-run — it's the first "
            "ingestion producing no usable content. "
            "The upstream embed/chunk/extract stages produced no output — "
            "check that /opt/airflow/data/ contains documents and that "
            "extract_sources, deduplicate_documents, chunk_documents, and "
            "embed_chunks all completed with non-empty results."
        )
    
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
    
    # Upsert in batches. `points` and `embedded_chunks` were built in the
    # same order with no skipping, so points[i:i+batch_size] always lines
    # up 1:1 with embedded_chunks[i:i+batch_size].
    points_upserted = 0
    confirmed_chunks: List[Dict[str, Any]] = []  # chunks from batches that ACTUALLY succeeded
    for i in range(0, len(points), batch_size):
        batch_points = points[i:i + batch_size]
        batch_chunks = embedded_chunks[i:i + batch_size]
        
        try:
            client.upsert(
                collection_name=collection_name,
                points=batch_points,
            )
            points_upserted += len(batch_points)
            confirmed_chunks.extend(batch_chunks)
            logger.debug(f"Upserted batch {i//batch_size + 1}/{(len(points)-1)//batch_size + 1}")
        
        except Exception as e:
            logger.error(f"Error upserting batch starting at index {i}: {e}")
            continue
    
    elapsed_time = time.time() - start_time
    
    logger.info(
        f"Upsert complete: {points_upserted} points in {elapsed_time:.2f}s "
        f"({points_upserted/elapsed_time:.2f} points/sec)"
    )

    # ── Confirm document hashes in Redis — ONLY NOW, after Qdrant has ──
    # confirmed these chunks are actually stored. This is the only place
    # in the whole pipeline that WRITES to the dedup cache;
    # deduplicate_documents only reads from it. A document whose chunks
    # ended up in a FAILED batch above is correctly left unconfirmed, so
    # the next run will retry it instead of skipping it forever.
    confirmed_hashes = set()
    for chunk in confirmed_chunks:
        content_hash = chunk.get('content_hash', '')
        if not content_hash or content_hash in confirmed_hashes:
            continue
        confirmed_hashes.add(content_hash)
        confirm_document_hash(
            content_hash,
            metadata={
                'source_uri': chunk.get('source_uri', ''),
                'filename': chunk.get('filename', ''),
                'first_seen': kwargs.get('ts', datetime.now().isoformat()),
            },
        )
    if confirmed_hashes:
        logger.info(
            f"Confirmed {len(confirmed_hashes)} document hash(es) in Redis "
            f"after successful upsert"
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