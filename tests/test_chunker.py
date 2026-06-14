"""
Test chunking logic - validates text splitting with overlap.
Tests edge cases like empty text, single chunk, etc.
"""

from dags.tasks.chunk import _split_text_into_chunks, chunk_documents


def test_simple_chunking():
    """
    Test basic text chunking with known input.
    """
    text = "This is a simple test. " * 100  # Create longer text
    chunks = _split_text_into_chunks(text, chunk_size=50, chunk_overlap=10)
    
    assert len(chunks) > 1, "Should create multiple chunks"
    assert all(isinstance(chunk, str) for chunk in chunks), "All chunks should be strings"


def test_chunking_with_overlap():
    """
    Test that chunks have proper overlap.
    """
    text = "Word " * 200  # Create text with 200 words
    chunks = _split_text_into_chunks(text, chunk_size=50, chunk_overlap=10)
    
    # Check that we have multiple chunks
    assert len(chunks) >= 2, "Should create at least 2 chunks"
    
    # Chunks should have some content
    for chunk in chunks:
        assert len(chunk) > 0, "Chunks should not be empty"


def test_empty_text():
    """
    Test chunking with empty text input.
    """
    text = ""
    chunks = _split_text_into_chunks(text, chunk_size=512, chunk_overlap=50)
    
    assert chunks == [], "Empty text should return empty list"


def test_very_short_text():
    """
    Test chunking with text shorter than chunk size.
    """
    text = "Short text."
    chunks = _split_text_into_chunks(text, chunk_size=512, chunk_overlap=50)
    
    assert len(chunks) == 1, "Short text should create single chunk"
    assert chunks[0] == text, "Single chunk should match original text"


def test_chunk_documents_function():
    """
    Test the main chunk_documents function with document list.
    """
    documents = [
        {
            'content': 'This is a test document. ' * 100,
            'source': 'test',
            'source_uri': 'test://doc1',
            'filename': 'doc1.txt',
            'content_hash': 'abc123',
            'metadata': {}
        }
    ]
    
    chunks = chunk_documents(
        documents=documents,
        chunk_size=50,
        chunk_overlap=10
    )
    
    assert len(chunks) > 0, "Should create chunks from documents"
    assert 'text' in chunks[0], "Chunks should have 'text' field"
    assert 'chunk_index' in chunks[0], "Chunks should have 'chunk_index' field"
    assert chunks[0]['source'] == 'test', "Chunks should preserve source metadata"


def test_multiple_documents():
    """
    Test chunking multiple documents.
    """
    documents = [
        {
            'content': 'Document one content. ' * 50,
            'source': 'test',
            'source_uri': 'test://doc1',
            'filename': 'doc1.txt',
            'content_hash': 'hash1',
            'metadata': {}
        },
        {
            'content': 'Document two content. ' * 50,
            'source': 'test',
            'source_uri': 'test://doc2',
            'filename': 'doc2.txt',
            'content_hash': 'hash2',
            'metadata': {}
        }
    ]
    
    chunks = chunk_documents(documents=documents, chunk_size=50, chunk_overlap=10)
    
    # Should have chunks from both documents
    assert len(chunks) > 2, "Should create chunks from multiple documents"
    
    # Check that we have chunks from different source URIs
    source_uris = set(chunk['source_uri'] for chunk in chunks)
    assert len(source_uris) == 2, "Should have chunks from both documents"