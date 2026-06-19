"""
Embedding generation — supports both OpenAI and local HuggingFace models.

Model selection is automatic:
  - model name starts with 'text-embedding-'  → OpenAI API
  - anything else (e.g. 'sentence-transformers/all-MiniLM-L6-v2') → local SentenceTransformer
"""

import logging
import os
import time
from typing import Any, Dict, List
import ast

from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def _get_openai_embeddings(texts: List[str], model: str) -> List[List[float]]:
    """Get embeddings from OpenAI API (lazy import — only used when OpenAI model selected)."""
    from openai import OpenAI  # lazy import — avoids parse-time failure when key not set

    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    response = client.embeddings.create(model=model, input=texts)
    return [item.embedding for item in response.data]


def _get_local_embeddings(texts: List[str], model_name: str) -> List[List[float]]:
    """Get embeddings from a local HuggingFace SentenceTransformer model."""
    from sentence_transformers import SentenceTransformer  # lazy import — heavy load

    logger.info(f"Loading SentenceTransformer model: {model_name}")
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, show_progress_bar=False)
    return embeddings.tolist()


def embed_chunks(
    chunks: Any,
    model_name: str = 'sentence-transformers/all-MiniLM-L6-v2',
    batch_size: int = 50,
    **kwargs,
) -> List[Dict[str, Any]]:
    """
    Generate embeddings for all text chunks.

    Args:
        chunks: List of chunk dicts (each must have a 'text' key), or XCom string.
        model_name: HuggingFace model name OR OpenAI model name.
        batch_size: Number of texts per embedding call.

    Returns:
        Same list with 'embedding', 'embedding_model', 'embedding_dimension' added.
    """
    # Handle XCom string input (Airflow passes XCom values as strings in templates)
    if isinstance(chunks, str):
        try:
            chunks = ast.literal_eval(chunks)
        except Exception as e:
            raise RuntimeError(
                f"embed_chunks could not parse its 'chunks' XCom argument. "
                f"The value from chunk_documents was not a valid Python literal. "
                f"Parse error: {e}"
            ) from e

    if not chunks:
        logger.warning("No chunks to embed — returning empty list")
        return []

    # Airflow may pass template strings if params weren't resolved
    if isinstance(model_name, str) and model_name.startswith('{{'):
        model_name = os.getenv('RAG_EMBEDDING_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')
    if isinstance(batch_size, str):
        batch_size = int(batch_size)

    is_openai = model_name.startswith('text-embedding-')
    logger.info(f"Embedding {len(chunks)} chunks with '{model_name}' ({'OpenAI' if is_openai else 'local'})")

    embedded_chunks: List[Dict] = []
    total_tokens = 0
    start_time = time.time()

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        batch_texts = [c.get('text', '') for c in batch]

        try:
            if is_openai:
                embeddings = _get_openai_embeddings(batch_texts, model_name)
                total_tokens += sum(len(t.split()) * 1.3 for t in batch_texts)
            else:
                embeddings = _get_local_embeddings(batch_texts, model_name)

            for chunk, emb in zip(batch, embeddings):
                chunk['embedding'] = emb
                chunk['embedding_model'] = model_name
                chunk['embedding_dimension'] = len(emb)
                embedded_chunks.append(chunk)

            logger.debug(f"Batch {i // batch_size + 1} done ({len(embedded_chunks)}/{len(chunks)} total)")

        except Exception as e:
            logger.error(f"Error embedding batch at index {i}: {e}")
            raise RuntimeError(
                f"Embedding failed for batch at index {i} using model '{model_name}'. "
                f"Check that the model is installed/accessible and the input texts are valid. "
                f"Error: {e}"
            ) from e

    elapsed = time.time() - start_time
    logger.info(
        f"Embedding complete: {len(embedded_chunks)}/{len(chunks)} chunks in {elapsed:.1f}s "
        f"({len(embedded_chunks) / max(elapsed, 0.1):.1f} chunks/sec)"
    )

    # Export metrics (lazy import — metrics_exporter is always available)
    try:
        from utils.metrics_exporter import export_counter, export_histogram
        export_counter('chunks_embedded_total', len(embedded_chunks))
        export_histogram('embedding_latency_seconds', elapsed)
        if is_openai and total_tokens > 0:
            cost = (total_tokens / 1_000_000) * 0.02  # $0.02 per 1M tokens
            export_counter('embedding_cost_usd', cost)
            logger.info(f"Estimated OpenAI cost: ${cost:.4f}")
    except Exception as e:
        logger.warning(f"Could not export metrics: {e}")

    return embedded_chunks