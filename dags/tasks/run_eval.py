"""
Retrieval evaluation using benchmark queries.
ENHANCED: Now supports hybrid search and reranking.
Computes Recall@K and MRR (Mean Reciprocal Rank).
"""

import logging
import os
import json
from typing import List, Dict, Any
from qdrant_client import QdrantClient
import time
from datetime import datetime
from utils.metadata_db import record_eval_result
import ast

logger = logging.getLogger(__name__)


def _get_qdrant_client() -> QdrantClient:
    """Get Qdrant client connection."""
    host = os.getenv('QDRANT_HOST', 'localhost')
    port = int(os.getenv('QDRANT_PORT', 6333))
    return QdrantClient(host=host, port=port)


def _load_benchmark_queries(benchmark_path: str) -> List[Dict[str, Any]]:
    """Load benchmark queries from JSON file."""
    if not os.path.exists(benchmark_path):
        logger.error(f"Benchmark file not found: {benchmark_path}")
        return []
    
    with open(benchmark_path, 'r') as f:
        queries = json.load(f)
    
    return queries


def _get_query_embedding(query: str, model_name: str) -> List[float]:
    """Generate embedding for query using the same model as documents."""
    from tasks.embed import _get_openai_embeddings, _get_local_embeddings
    
    is_openai = model_name.startswith('text-embedding-')
    
    if is_openai:
        embeddings = _get_openai_embeddings([query], model_name)
        return embeddings[0]
    else:
        embeddings = _get_local_embeddings([query], model_name)
        return embeddings[0]


def _compute_recall_at_k(
    retrieved_docs: List[str],
    expected_docs: List[str],
    k: int
) -> float:
    """Compute Recall@K metric."""
    if not expected_docs:
        return 0.0
    
    top_k_docs = set(retrieved_docs[:k])
    relevant_docs = set(expected_docs)
    
    relevant_retrieved = top_k_docs.intersection(relevant_docs)
    recall = len(relevant_retrieved) / len(relevant_docs)
    
    return recall


def _compute_mrr(retrieved_docs: List[str], expected_docs: List[str]) -> float:
    """Compute Mean Reciprocal Rank (MRR)."""
    expected_set = set(expected_docs)
    
    for rank, doc in enumerate(retrieved_docs, start=1):
        if doc in expected_set:
            return 1.0 / rank
    
    return 0.0


def _filter_expired_documents(client: QdrantClient, collection_name: str) -> int:
    """
    Remove expired documents from collection.
    
    Args:
        client: Qdrant client
        collection_name: Collection to clean
        
    Returns:
        Number of expired documents removed
    """
    
    try:
        # Get current timestamp
        now = datetime.now().isoformat()
        
        # Delete points where expires_at < now
        # Note: Qdrant doesn't support direct delete by filter with date comparison
        # So we scroll through and delete manually
        
        expired_count = 0
        offset = None
        
        while True:
            # Scroll through collection
            records, offset = client.scroll(
                collection_name=collection_name,
                limit=100,
                offset=offset,
                with_payload=True,
                with_vectors=False
            )
            
            if not records:
                break
            
            # Find expired documents
            expired_ids = []
            for record in records:
                expires_at = record.payload.get('expires_at')
                if expires_at and expires_at < now:
                    expired_ids.append(record.id)
            
            # Delete expired
            if expired_ids:
                client.delete(
                    collection_name=collection_name,
                    points_selector=expired_ids
                )
                expired_count += len(expired_ids)
                logger.info(f"Deleted {len(expired_ids)} expired documents")
            
            if offset is None:
                break
        
        logger.info(f"Total expired documents removed: {expired_count}")
        return expired_count
    
    except Exception as e:
        logger.error(f"Error filtering expired documents: {e}")
        return 0


def run_retrieval_evaluation(
    collection_name: str,
    benchmark_path: str,
    model_name: str,
    use_hybrid_search: bool = False,  # NEW
    use_reranking: bool = False,  # NEW
    top_k: int = 10,
    **kwargs
) -> Dict[str, Any]:
    """
    Run retrieval evaluation against benchmark queries.
    
    ENHANCED: Now supports hybrid search and reranking.
    
    Args:
        collection_name: Qdrant collection to query
        benchmark_path: Path to benchmark queries JSON
        model_name: Embedding model name
        use_hybrid_search: Use hybrid BM25 + vector search
        use_reranking: Use cross-encoder reranking
        top_k: Number of results to retrieve per query
        
    Returns:
        Evaluation metrics dictionary
    """
    # Parse boolean parameters if they come as strings
    if isinstance(use_hybrid_search, str):
        use_hybrid_search = use_hybrid_search.lower() == 'true'
    if isinstance(use_reranking, str):
        use_reranking = use_reranking.lower() == 'true'
    
    logger.info(f"Starting retrieval evaluation on collection '{collection_name}'")
    logger.info(f"Hybrid search: {use_hybrid_search}, Reranking: {use_reranking}")
    
    client = _get_qdrant_client()

    # Step 0: Verify collection exists before doing anything else.
    # The upsert task runs before this task, so a missing collection means
    # the upstream task failed or was skipped — fail fast with a clear message.
    try:
        client.get_collection(collection_name)
    except Exception as e:
        logger.error(
            f"Collection '{collection_name}' does not exist in Qdrant. "
            f"Ensure the upsert_vectors task completed successfully before "
            f"running retrieval evaluation. Original error: {e}"
        )
        raise RuntimeError(
            f"Collection '{collection_name}' not found in Qdrant. "
            "Cannot run evaluation on a missing collection."
        ) from e

    # Step 1: Filter expired documents (safe — collection is confirmed to exist)
    expired_count = _filter_expired_documents(client, collection_name)
    
    # Load benchmark queries
    queries = _load_benchmark_queries(benchmark_path)
    if not queries:
        logger.error("No benchmark queries loaded")
        return {'success': False, 'error': 'No benchmark queries'}
    
    logger.info(f"Loaded {len(queries)} benchmark queries")
    
    # Get all chunks for hybrid search (if needed)
    all_chunks = []
    if use_hybrid_search:
        logger.info("Loading all chunks for BM25 index...")
        offset = None
        while True:
            records, offset = client.scroll(
                collection_name=collection_name,
                limit=100,
                offset=offset,
                with_payload=True,
                with_vectors=False
            )
            if not records:
                break
            for record in records:
                all_chunks.append({
                    'id': record.id,
                    'text': record.payload.get('text', '')
                })
            if offset is None:
                break
        logger.info(f"Loaded {len(all_chunks)} chunks for BM25")
    
    # Track metrics
    recall_at_1_scores = []
    recall_at_5_scores = []
    recall_at_10_scores = []
    mrr_scores = []
    query_latencies = []
    
    for query_obj in queries:
        query_text = query_obj['query']
        expected_docs = query_obj.get('expected_docs', [])
        
        try:
            start_time = time.time()
            
            # Generate query embedding
            query_embedding = _get_query_embedding(query_text, model_name)
            
            # Search
            if use_hybrid_search:
                # Use hybrid search
                from tasks.hybrid_search import perform_hybrid_search
                search_results = perform_hybrid_search(
                    query=query_text,
                    query_vector=query_embedding,
                    collection_name=collection_name,
                    chunks=all_chunks,
                    top_k=top_k * 2 if use_reranking else top_k
                )
            else:
                # Use standard vector search
                from utils.vector_store import search_similar
                search_results = search_similar(
                    collection_name=collection_name,
                    query_vector=query_embedding,
                    limit=top_k * 2 if use_reranking else top_k
                )
            
            # Rerank (if enabled)
            if use_reranking and search_results:
                from tasks.reranker import rerank_results
                search_results = rerank_results(
                    query=query_text,
                    results=search_results,
                    top_k=top_k
                )
            else:
                search_results = search_results[:top_k]
            
            query_latency = time.time() - start_time
            query_latencies.append(query_latency)
            
            # Extract retrieved document filenames
            retrieved_docs = [
                result.get('payload', {}).get('filename', '') 
                for result in search_results
            ]
            
            # Compute metrics
            recall_at_1 = _compute_recall_at_k(retrieved_docs, expected_docs, k=1)
            recall_at_5 = _compute_recall_at_k(retrieved_docs, expected_docs, k=5)
            recall_at_10 = _compute_recall_at_k(retrieved_docs, expected_docs, k=10)
            mrr = _compute_mrr(retrieved_docs, expected_docs)
            
            recall_at_1_scores.append(recall_at_1)
            recall_at_5_scores.append(recall_at_5)
            recall_at_10_scores.append(recall_at_10)
            mrr_scores.append(mrr)
            
            logger.debug(
                f"Query: '{query_text[:50]}...' | "
                f"Recall@5: {recall_at_5:.2f} | MRR: {mrr:.2f}"
            )
        
        except Exception as e:
            logger.error(f"Error evaluating query '{query_text}': {e}")
            continue
    
    # Aggregate results
    avg_recall_at_1 = sum(recall_at_1_scores) / len(recall_at_1_scores) if recall_at_1_scores else 0
    avg_recall_at_5 = sum(recall_at_5_scores) / len(recall_at_5_scores) if recall_at_5_scores else 0
    avg_recall_at_10 = sum(recall_at_10_scores) / len(recall_at_10_scores) if recall_at_10_scores else 0
    avg_mrr = sum(mrr_scores) / len(mrr_scores) if mrr_scores else 0
    avg_latency = sum(query_latencies) / len(query_latencies) if query_latencies else 0
    
    results = {
        'success': True,
        'collection_name': collection_name,
        'total_queries': len(queries),
        'recall@1': round(avg_recall_at_1, 4),
        'recall@5': round(avg_recall_at_5, 4),
        'recall@10': round(avg_recall_at_10, 4),
        'mrr': round(avg_mrr, 4),
        'avg_query_latency_ms': round(avg_latency * 1000, 2),
        'expired_docs_removed': expired_count,  # NEW
        'used_hybrid_search': use_hybrid_search,  # NEW
        'used_reranking': use_reranking,  # NEW
    }
    
    logger.info(f"Evaluation complete: {json.dumps(results, indent=2)}")
    
    # Export metrics
    from utils.metrics_exporter import export_gauge, export_histogram
    export_gauge('eval_recall_at_1', avg_recall_at_1)
    export_gauge('eval_recall_at_5', avg_recall_at_5)
    export_gauge('eval_recall_at_10', avg_recall_at_10)
    export_gauge('eval_mrr', avg_mrr)
    export_histogram('eval_query_latency_seconds', avg_latency)
    export_gauge('expired_docs_removed', expired_count)
    
    # Log to MLflow
    try:
        import mlflow
        mlflow.log_metrics({
            'recall@1': avg_recall_at_1,
            'recall@5': avg_recall_at_5,
            'recall@10': avg_recall_at_10,
            'mrr': avg_mrr,
            'avg_query_latency_ms': avg_latency * 1000,
            'expired_docs_removed': expired_count,
        })
    except Exception as e:
        logger.warning(f"Could not log to MLflow: {e}")

    try:
        record_eval_result(
            run_id=kwargs.get('run_id', 'unknown'),
            collection_name=collection_name,
            eval_results=results,
            threshold_value=float(kwargs.get('eval_threshold', 0.75)),
        )
    except Exception as e:
        logger.warning(f"Could not record eval result to metadata DB: {e}")
    
    return results