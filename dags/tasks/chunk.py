"""
Document chunking with overlapping windows.
"""

import logging
from typing import List, Dict, Any
import tiktoken

logger = logging.getLogger(__name__)


def _count_tokens(text: str, model: str = "gpt-3.5-turbo") -> int:
    """
    Count tokens in text using tiktoken.
    
    Args:
        text: Input text
        model: Model name for tokenizer
        
    Returns:
        Number of tokens
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    except Exception as e:
        logger.warning(f"Error counting tokens: {e}, falling back to character estimation")
        # Rough estimation: ~4 characters per token
        return len(text) // 4


def _split_text_into_chunks(
    text: str,
    chunk_size: int = 512,
    chunk_overlap: int = 50,
) -> List[str]:
    """
    Split text into overlapping chunks by tokens.
    
    Args:
        text: Input text to chunk
        chunk_size: Target chunk size in tokens
        chunk_overlap: Number of overlapping tokens between chunks
        
    Returns:
        List of text chunks
    """
    if not text or not text.strip():
        return []
    
    # Use tiktoken for accurate splitting
    try:
        encoding = tiktoken.get_encoding("cl100k_base")
        tokens = encoding.encode(text)
    except Exception as e:
        logger.error(f"Error encoding text: {e}")
        return []
    
    chunks = []
    start_idx = 0
    
    while start_idx < len(tokens):
        # Get chunk of tokens
        end_idx = start_idx + chunk_size
        chunk_tokens = tokens[start_idx:end_idx]
        
        # Decode back to text
        chunk_text = encoding.decode(chunk_tokens)
        chunks.append(chunk_text)
        
        # Move start position with overlap
        start_idx += chunk_size - chunk_overlap
        
        # Prevent infinite loop
        if chunk_overlap >= chunk_size:
            start_idx = end_idx
    
    return chunks


def chunk_documents(
    documents: Any,
    chunk_size: int = 512,
    chunk_overlap: int = 50,
    **kwargs
) -> List[Dict[str, Any]]:
    """
    Split documents into overlapping text chunks.
    
    Args:
        documents: List of document dictionaries or XCom reference
        chunk_size: Target size of each chunk in tokens
        chunk_overlap: Number of overlapping tokens between chunks
        
    Returns:
        List of chunk dictionaries with metadata
    """
    # Handle XCom or string input
    if isinstance(documents, str):
        try:
            documents = eval(documents)
        except Exception as e:
            logger.error(f"Could not parse documents from XCom:  {e}")
            return []
    
    
    if not documents:
        logger.warning("No documents to chunk")
        return []
    
    # Parse parameters if passed as strings
    if isinstance(chunk_size, str):
        chunk_size = int(chunk_size)
    if isinstance(chunk_overlap, str):
        chunk_overlap = int(chunk_overlap)
    
    logger.info(
        f"Starting chunking for {len(documents)} documents "
        f"(chunk_size={chunk_size}, overlap={chunk_overlap})"
    )
    
    all_chunks = []
    total_chunks = 0
    
    for doc in documents:
        content = doc.get('content', '')
        if not content:
            continue
        
        # Split into chunks
        text_chunks = _split_text_into_chunks(
            text=content,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        
        # Create chunk objects with metadata
        for idx, chunk_text in enumerate(text_chunks):
            chunk_obj = {
                'text': chunk_text,
                'chunk_index': idx,
                'total_chunks': len(text_chunks),
                'source': doc.get('source', 'unknown'),
                'source_uri': doc.get('source_uri', ''),
                'filename': doc.get('filename', ''),
                'content_hash': doc.get('content_hash', ''),
                'metadata': {
                    **doc.get('metadata', {}),
                    'chunk_size': chunk_size,
                    'chunk_overlap': chunk_overlap,
                    'token_count': _count_tokens(chunk_text),
                }
            }
            all_chunks.append(chunk_obj)
        
        total_chunks += len(text_chunks)
        logger.debug(
            f"Chunked '{doc.get('filename', 'unknown')}' "
            f"into {len(text_chunks)} chunks"
        )
    
    logger.info(f"Chunking complete: {total_chunks} chunks created from {len(documents)} documents")
    
    # Export metrics
    from utils.metrics_exporter import export_counter, export_gauge
    export_counter('chunks_created_total', total_chunks)
    export_gauge('average_chunks_per_document', total_chunks / len(documents))
    
    return all_chunks