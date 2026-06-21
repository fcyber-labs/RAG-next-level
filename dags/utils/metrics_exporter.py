"""
Simple Prometheus metrics exporter using pushgateway.
Tracks pipeline metrics: docs, chunks, latency, scores.
"""

import logging
import os
from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, push_to_gateway

logger = logging.getLogger(__name__)

# Get Pushgateway URL from environment
PUSHGATEWAY_URL = os.getenv('PROMETHEUS_PUSHGATEWAY', 'prometheus-pushgateway:9091')

# Create a registry for this pipeline
registry = CollectorRegistry()

# ─── Counters ──────────────────────────────────────────────────────────────

documents_extracted_counter = Counter(
    'rag_documents_extracted_total',
    'Total number of raw documents pulled from sources (before deduplication)',
    registry=registry
)

documents_counter = Counter(
    'rag_documents_processed_total',
    'Total number of new (non-duplicate) documents processed',
    registry=registry
)

documents_skipped_counter = Counter(
    'rag_documents_deduplicated_skipped_total',
    'Total number of documents skipped because they were already seen (duplicates)',
    registry=registry
)

chunks_counter = Counter(
    'rag_chunks_created_total',
    'Total number of chunks created',
    registry=registry
)

embeddings_counter = Counter(
    'rag_embeddings_generated_total',
    'Total number of embeddings generated',
    registry=registry
)

vectors_counter = Counter(
    'rag_vectors_upserted_total',
    'Total number of vectors upserted',
    registry=registry
)

query_rewrites_counter = Counter(
    'rag_query_rewrites_total',
    'Total number of queries rewritten by the query-rewriting stage',
    registry=registry
)

collection_promotions_counter = Counter(
    'rag_collection_promotions_total',
    'Total number of successful staging -> production collection promotions',
    registry=registry
)

collection_promotion_failures_counter = Counter(
    'rag_collection_promotion_failures_total',
    'Total number of failed staging -> production collection promotions',
    registry=registry
)

embedding_cost_counter = Counter(
    'rag_embedding_cost_usd_total',
    'Estimated cumulative cost (USD) of OpenAI embedding calls',
    registry=registry
)

# ─── Histograms ────────────────────────────────────────────────────────────

embedding_latency = Histogram(
    'rag_embedding_latency_seconds',
    'Time spent generating embeddings',
    registry=registry
)

upsert_latency = Histogram(
    'rag_upsert_latency_seconds',
    'Time spent upserting vectors',
    registry=registry
)

eval_query_latency = Histogram(
    'rag_eval_query_latency_seconds',
    'Average per-query latency observed during retrieval evaluation',
    registry=registry
)

reranker_score_improvement = Histogram(
    'rag_reranker_score_improvement',
    'Change in the top result score after reranking (reranked_score - original_score)',
    buckets=(-1.0, -0.5, -0.25, -0.1, -0.05, 0.0, 0.05, 0.1, 0.25, 0.5, 1.0, float('inf')),
    registry=registry
)

# ─── Gauges ────────────────────────────────────────────────────────────────

eval_recall_at_1 = Gauge(
    'rag_eval_recall_at_1',
    'Recall@1 score from evaluation',
    registry=registry
)

eval_recall_at_5 = Gauge(
    'rag_eval_recall_at_5',
    'Recall@5 score from evaluation',
    registry=registry
)

eval_recall_at_10 = Gauge(
    'rag_eval_recall_at_10',
    'Recall@10 score from evaluation',
    registry=registry
)

eval_mrr = Gauge(
    'rag_eval_mrr',
    'Mean Reciprocal Rank from evaluation',
    registry=registry
)

expired_docs_removed_gauge = Gauge(
    'rag_expired_docs_removed',
    'Number of expired documents removed during the last evaluation run',
    registry=registry
)

dedup_cache_hit_rate = Gauge(
    'rag_dedup_cache_hit_rate',
    'Deduplication cache hit rate',
    registry=registry
)

collection_total_points_gauge = Gauge(
    'rag_collection_total_points',
    'Total number of points currently stored in the target collection',
    registry=registry
)

hybrid_search_bm25_weight_gauge = Gauge(
    'rag_hybrid_search_bm25_weight',
    'BM25 weight used in the most recent hybrid search call',
    registry=registry
)

average_chunks_per_document_gauge = Gauge(
    'rag_average_chunks_per_document',
    'Average number of chunks produced per document in the last chunking run',
    registry=registry
)


def export_counter(metric_name: str, value: float):
    """
    Export a counter metric to Prometheus.
    
    Args:
        metric_name: Name of the metric
        value: Value to add to counter
    """
    try:
        # Map metric names to actual counter objects
        metric_map = {
            'documents_extracted_total': documents_extracted_counter,
            'documents_deduplicated_new': documents_counter,
            'documents_deduplicated_skipped': documents_skipped_counter,
            'chunks_created_total': chunks_counter,
            'chunks_embedded_total': embeddings_counter,
            'vectors_upserted_total': vectors_counter,
            'query_rewrites_total': query_rewrites_counter,
            'collection_promotions_total': collection_promotions_counter,
            'collection_promotion_failures_total': collection_promotion_failures_counter,
            'embedding_cost_usd': embedding_cost_counter,
        }
        
        counter = metric_map.get(metric_name)
        if counter:
            counter.inc(value)
            logger.debug(f"Incremented counter {metric_name} by {value}")
        else:
            logger.warning(f"Unknown counter metric name '{metric_name}' — value not exported")
    
    except Exception as e:
        logger.error(f"Error exporting counter {metric_name}: {e}")


def export_gauge(metric_name: str, value: float):
    """
    Export a gauge metric to Prometheus.
    
    Args:
        metric_name: Name of the metric
        value: Current value to set
    """
    try:
        # Map metric names to actual gauge objects
        metric_map = {
            'eval_recall_at_1': eval_recall_at_1,
            'eval_recall_at_5': eval_recall_at_5,
            'eval_recall_at_10': eval_recall_at_10,
            'eval_mrr': eval_mrr,
            'expired_docs_removed': expired_docs_removed_gauge,
            'deduplication_cache_hit_rate': dedup_cache_hit_rate,
            'collection_total_points': collection_total_points_gauge,
            'hybrid_search_bm25_weight': hybrid_search_bm25_weight_gauge,
            'average_chunks_per_document': average_chunks_per_document_gauge,
        }
        
        gauge = metric_map.get(metric_name)
        if gauge:
            gauge.set(value)
            logger.debug(f"Set gauge {metric_name} to {value}")
        else:
            logger.warning(f"Unknown gauge metric name '{metric_name}' — value not exported")
    
    except Exception as e:
        logger.error(f"Error exporting gauge {metric_name}: {e}")


def export_histogram(metric_name: str, value: float):
    """
    Export a histogram metric to Prometheus.
    
    Args:
        metric_name: Name of the metric
        value: Value to observe
    """
    try:
        # Map metric names to actual histogram objects
        metric_map = {
            'embedding_latency_seconds': embedding_latency,
            'upsert_latency_seconds': upsert_latency,
            'eval_query_latency_seconds': eval_query_latency,
            'reranker_score_improvement': reranker_score_improvement,
        }
        
        histogram = metric_map.get(metric_name)
        if histogram:
            histogram.observe(value)
            logger.debug(f"Observed {value} for histogram {metric_name}")
        else:
            logger.warning(f"Unknown histogram metric name '{metric_name}' — value not exported")
    
    except Exception as e:
        logger.error(f"Error exporting histogram {metric_name}: {e}")


def push_metrics():
    """
    Push all metrics to Prometheus Pushgateway.
    Call this at the end of a task to send metrics.
    """
    try:
        push_to_gateway(
            PUSHGATEWAY_URL,
            job='rag_pipeline',
            registry=registry
        )
        logger.info("Pushed metrics to Prometheus Pushgateway")
    
    except Exception as e:
        logger.error(f"Error pushing metrics to Pushgateway: {e}")