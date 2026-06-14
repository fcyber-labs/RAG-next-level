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

# Define metrics
documents_counter = Counter(
    'rag_documents_processed_total',
    'Total number of documents processed',
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

eval_recall_at_5 = Gauge(
    'rag_eval_recall_at_5',
    'Recall@5 score from evaluation',
    registry=registry
)

eval_mrr = Gauge(
    'rag_eval_mrr',
    'Mean Reciprocal Rank from evaluation',
    registry=registry
)

dedup_cache_hit_rate = Gauge(
    'rag_dedup_cache_hit_rate',
    'Deduplication cache hit rate',
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
            'documents_extracted_total': documents_counter,
            'documents_deduplicated_new': documents_counter,
            'chunks_created_total': chunks_counter,
            'chunks_embedded_total': embeddings_counter,
            'vectors_upserted_total': vectors_counter,
        }
        
        counter = metric_map.get(metric_name)
        if counter:
            counter.inc(value)
            logger.debug(f"Incremented counter {metric_name} by {value}")
    
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
            'eval_recall_at_5': eval_recall_at_5,
            'eval_recall_at_1': eval_recall_at_5,  # Reuse for simplicity
            'eval_mrr': eval_mrr,
            'deduplication_cache_hit_rate': dedup_cache_hit_rate,
        }
        
        gauge = metric_map.get(metric_name)
        if gauge:
            gauge.set(value)
            logger.debug(f"Set gauge {metric_name} to {value}")
    
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
        }
        
        histogram = metric_map.get(metric_name)
        if histogram:
            histogram.observe(value)
            logger.debug(f"Observed {value} for histogram {metric_name}")
    
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