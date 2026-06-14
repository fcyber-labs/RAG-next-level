"""
Test deduplication logic using mocked Redis.
Tests hash checking and duplicate detection.
"""

from unittest.mock import Mock, patch
from dags.tasks.deduplicate import deduplicate_documents
from dags.utils.hash_store import compute_content_hash, check_document_hash


def test_compute_content_hash():
    """
    Test that content hashing is deterministic.
    """
    content1 = "This is a test document."
    content2 = "This is a test document."
    content3 = "This is a different document."
    
    hash1 = compute_content_hash(content1)
    hash2 = compute_content_hash(content2)
    hash3 = compute_content_hash(content3)
    
    # Same content should produce same hash
    assert hash1 == hash2, "Same content should have same hash"
    
    # Different content should produce different hash
    assert hash1 != hash3, "Different content should have different hash"
    
    # Hash should be 64 characters (SHA-256 hex)
    assert len(hash1) == 64, "SHA-256 hash should be 64 hex characters"


@patch('dags.utils.hash_store.get_redis_client')
def test_check_document_hash_new(mock_redis_client):
    """
    Test checking hash for a new document (not in Redis).
    """
    # Mock Redis to return True (key was set, i.e., document is new)
    mock_client = Mock()
    mock_client.set.return_value = True
    mock_redis_client.return_value = mock_client
    
    content = "This is a new document."
    metadata = {'filename': 'test.txt'}
    
    is_new = check_document_hash(content, metadata)
    
    assert is_new, "New document should return True"
    mock_client.set.assert_called_once()


@patch('dags.utils.hash_store.get_redis_client')
def test_check_document_hash_duplicate(mock_redis_client):
    """
    Test checking hash for a duplicate document (already in Redis).
    """
    # Mock Redis to return False (key already exists, i.e., duplicate)
    mock_client = Mock()
    mock_client.set.return_value = False
    mock_redis_client.return_value = mock_client
    
    content = "This is a duplicate document."
    metadata = {'filename': 'test.txt'}
    
    is_new = check_document_hash(content, metadata)
    
    assert not is_new, "Duplicate document should return False"


@patch('dags.tasks.deduplicate._get_redis_client')
def test_deduplicate_documents_all_new(mock_redis_client):
    """
    Test deduplication when all documents are new.
    """
    # Mock Redis to always return True (all new)
    mock_client = Mock()
    mock_client.set.return_value = True
    mock_redis_client.return_value = mock_client
    
    documents = [
        {'content': 'Document 1', 'source_uri': 'uri1', 'filename': 'doc1.txt'},
        {'content': 'Document 2', 'source_uri': 'uri2', 'filename': 'doc2.txt'},
    ]
    
    new_docs = deduplicate_documents(documents)
    
    assert len(new_docs) == 2, "All documents should be new"


@patch('dags.tasks.deduplicate._get_redis_client')
def test_deduplicate_documents_some_duplicates(mock_redis_client):
    """
    Test deduplication when some documents are duplicates.
    """
    # Mock Redis to return True for first, False for second
    mock_client = Mock()
    mock_client.set.side_effect = [True, False]  # First new, second duplicate
    mock_redis_client.return_value = mock_client
    
    documents = [
        {'content': 'Document 1', 'source_uri': 'uri1', 'filename': 'doc1.txt'},
        {'content': 'Document 2', 'source_uri': 'uri2', 'filename': 'doc2.txt'},
    ]
    
    new_docs = deduplicate_documents(documents)
    
    assert len(new_docs) == 1, "Only one document should be new"
    assert new_docs[0]['filename'] == 'doc1.txt', "First document should be included"


def test_deduplicate_empty_list():
    """
    Test deduplication with empty document list.
    """
    documents = []
    new_docs = deduplicate_documents(documents)
    
    assert new_docs == [], "Empty input should return empty list"