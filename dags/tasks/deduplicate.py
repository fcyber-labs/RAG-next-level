"""
Document deduplication using Redis-based content hashing.

IMPORTANT: This task only CHECKS whether a document has already been
confirmed as stored (read-only, via utils.hash_store.document_hash_exists).
It does NOT write anything to Redis.

The "confirmed" write happens later, in upsert_to_qdrant.confirm_document_hash,
and ONLY after Qdrant has actually stored the document's vectors successfully.

Why this order matters:
The old version called Redis SETNX right here, the moment a document was
seen — before chunk_documents, embed_chunks, or upsert_to_qdrant had even
run. If any of those later tasks then failed, Redis would permanently
"remember" a document that was never actually saved (a "ghost duplicate").
Every future run would skip it forever (for the 30-day TTL) while the
knowledge base stayed empty for it — with upsert_to_qdrant having no way
to tell "nothing changed" apart from "the pipeline is broken".

Moving the write to upsert_to_qdrant, after a confirmed success, fixes
this: a crash anywhere between here and a successful upsert leaves Redis
untouched for that document, so the next run correctly retries it instead
of skipping it forever.
"""

import logging
from typing import List, Dict, Any
import ast

from utils.hash_store import compute_content_hash, document_hash_exists

logger = logging.getLogger(__name__)


def deduplicate_documents(documents: Any, **kwargs) -> List[Dict[str, Any]]:
    """
    Filter out documents that have already been CONFIRMED as stored in
    Qdrant. Only returns new or modified documents. Does NOT write to
    Redis — that happens later, in upsert_to_qdrant, after a confirmed
    success.

    Args:
        documents: List of document dictionaries or XCom reference

    Returns:
        List of candidate-new documents (each has 'content_hash' attached)
    """
    # Handle XCom pull
    if isinstance(documents, str):
        try:
            documents = ast.literal_eval(documents)
        except Exception as e:
            raise RuntimeError(
                f"deduplicate_documents could not parse its 'documents' XCom argument. "
                f"The value from extract_all_sources was not a valid Python literal. "
                f"Parse error: {e}"
            ) from e

    if not documents:
        logger.warning("No documents to deduplicate")
        return []

    logger.info(f"Starting deduplication for {len(documents)} documents")

    new_documents = []
    duplicate_count = 0

    for doc in documents:
        content = doc.get('content', '')
        if not content:
            continue

        # Compute content hash
        content_hash = compute_content_hash(content)
        doc['content_hash'] = content_hash

        # READ-ONLY check — does NOT claim/write the key
        if document_hash_exists(content_hash):
            duplicate_count += 1
            logger.debug(
                f"Duplicate document (already confirmed stored in Qdrant, "
                f"skipped): {doc.get('filename', 'unknown')}"
            )
        else:
            new_documents.append(doc)
            logger.debug(f"Candidate new document: {doc.get('filename', 'unknown')}")

    logger.info(
        f"Deduplication complete: {len(new_documents)} candidate new, "
        f"{duplicate_count} duplicates skipped"
    )

    # Export metrics
    from utils.metrics_exporter import export_counter, export_gauge
    from utils.metadata_db import record_documents
    record_documents(new_documents)
    export_counter('documents_deduplicated_new', len(new_documents))
    export_counter('documents_deduplicated_skipped', duplicate_count)
    export_gauge(
        'deduplication_cache_hit_rate',
        duplicate_count / len(documents) if documents else 0,
    )

    return new_documents