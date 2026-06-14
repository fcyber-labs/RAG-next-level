"""
Tasks package for RAG refresh pipeline.
Contains all task implementations for document processing, embedding, and evaluation.
"""

__all__ = [
    'extract_sources',
    'deduplicate_documents',
    'chunk_documents',
    'embed_chunks',
    'upsert_to_qdrant',
    'run_retrieval_evaluation',
    'rollback_collection',
    'promote_collection',
]