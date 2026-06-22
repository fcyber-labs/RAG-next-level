"""
Airflow task: pre-build and warm the search cache in Redis.

Called by the DAG after upsert_vectors succeeds so the BM25 index and
chunk list are ready in Redis before the first search request arrives.
This eliminates the O(N) rebuild that previously happened on every
Streamlit search (the "10 second" latency the user observed).

What this task does
-------------------
1. Scrolls the entire collection from Qdrant (with full payload).
2. Tokenises all chunk texts and builds a BM25Okapi index.
3. Stores the pickled (BM25 index + chunks) in Redis via set_bm25_index.
4. Stores the chunk list as JSON in Redis via set_cached_chunks
   (used by the fallback path in app.py for the non-BM25 scroll cache).

After this task, a Streamlit hybrid search cold-starts in ~1-2 s instead
of ~10 s because:
  - Qdrant scroll is skipped       (chunk list already in Redis)
  - BM25 build is skipped          (index already pickled in Redis)
  - Query embedding (~100 ms) and  Qdrant ANN search (~200 ms) still run
"""

import logging
import os
from typing import Dict, Any

from qdrant_client import QdrantClient

logger = logging.getLogger(__name__)


def prepare_search_cache(
    collection_name: str = "knowledge_base_staging",
    **kwargs,
) -> Dict[str, Any]:
    """
    Pre-build and cache the BM25 index + chunk list for a collection.

    Args:
        collection_name: Qdrant collection to index (default: staging).

    Returns:
        {'success': bool, 'collection_name': str, 'chunks_cached': int}
    """
    from rank_bm25 import BM25Okapi
    from utils.chunk_cache import set_cached_chunks, set_bm25_index
    from tasks.hybrid_search import tokenize

    host = os.getenv('QDRANT_HOST', 'localhost')
    port = int(os.getenv('QDRANT_PORT', 6333))
    client = QdrantClient(host=host, port=port)

    logger.info(f"[prepare_search_cache] Starting cache warm-up for '{collection_name}'")

    # ── 1. Scroll all chunks with full payload ──────────────────────────────
    all_chunks = []
    offset     = None

    try:
        while True:
            records, offset = client.scroll(
                collection_name=collection_name,
                limit=100,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            if not records:
                break
            for record in records:
                all_chunks.append({
                    'id':      str(record.id),
                    'text':    record.payload.get('text', ''),
                    'payload': record.payload,
                })
            if offset is None:
                break
    except Exception as e:
        logger.error(f"[prepare_search_cache] Scroll failed for '{collection_name}': {e}")
        return {'success': False, 'collection_name': collection_name, 'error': str(e)}

    logger.info(f"[prepare_search_cache] Scrolled {len(all_chunks)} chunks")

    if not all_chunks:
        logger.warning(f"[prepare_search_cache] No chunks in '{collection_name}' — skipping index build")
        return {'success': True, 'collection_name': collection_name, 'chunks_cached': 0}

    # ── 2. Build BM25 index ─────────────────────────────────────────────────
    try:
        tokenized   = [tokenize(c['text']) for c in all_chunks]
        bm25_index  = BM25Okapi(tokenized)
        logger.info(f"[prepare_search_cache] BM25 index built ({len(all_chunks)} documents)")
    except Exception as e:
        logger.error(f"[prepare_search_cache] BM25 build failed: {e}")
        return {'success': False, 'collection_name': collection_name, 'error': str(e)}

    # ── 3. Warm Redis ───────────────────────────────────────────────────────
    try:
        # Pickled BM25 + chunks (primary — used by hybrid search)
        set_bm25_index(collection_name, bm25_index, all_chunks)
        # JSON chunk list (fallback — used when BM25 cache is bypassed)
        set_cached_chunks(collection_name, all_chunks)
    except Exception as e:
        logger.error(f"[prepare_search_cache] Redis write failed: {e}")
        return {'success': False, 'collection_name': collection_name, 'error': str(e)}

    logger.info(
        f"[prepare_search_cache] Cache warm-up complete for '{collection_name}' "
        f"— {len(all_chunks)} chunks, BM25 index stored ✅"
    )

    return {
        'success':         True,
        'collection_name': collection_name,
        'chunks_cached':   len(all_chunks),
    }