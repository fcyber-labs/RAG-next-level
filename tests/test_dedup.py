"""
Test deduplication logic using mocked Redis.

Tests the two-phase dedup design:
  Phase 1 (read-only):  document_hash_exists()   — used in deduplicate_documents
  Phase 2 (write):       confirm_document_hash()  — used in upsert_to_qdrant,
                                                     only after a confirmed
                                                     successful upsert.
"""

from unittest.mock import Mock, patch
from dags.tasks.deduplicate import deduplicate_documents
from dags.utils.hash_store import (
    compute_content_hash,
    document_hash_exists,
    confirm_document_hash,
)


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
def test_document_hash_exists_new(mock_redis_client):
    """
    Phase 1: a hash never confirmed should read as "not seen yet" (False).
    """
    mock_client = Mock()
    mock_client.exists.return_value = 0  # key doesn't exist
    mock_redis_client.return_value = mock_client

    content_hash = compute_content_hash("This is a new document.")

    exists = document_hash_exists(content_hash)

    assert exists is False, "Unconfirmed hash should read as not-yet-seen"
    mock_client.exists.assert_called_once()
    # Phase 1 must NEVER write to Redis
    mock_client.set.assert_not_called()


@patch('dags.utils.hash_store.get_redis_client')
def test_document_hash_exists_duplicate(mock_redis_client):
    """
    Phase 1: a hash that was previously confirmed should read as True.
    """
    mock_client = Mock()
    mock_client.exists.return_value = 1  # key exists
    mock_redis_client.return_value = mock_client

    content_hash = compute_content_hash("This is a duplicate document.")

    exists = document_hash_exists(content_hash)

    assert exists is True, "Confirmed hash should read as already-seen"


@patch('dags.utils.hash_store.get_redis_client')
def test_document_hash_exists_redis_error_defaults_to_false(mock_redis_client):
    """
    Phase 1: a Redis error must default to False (treat as new), never
    silently drop a document because Redis happened to be unreachable.
    """
    import redis
    mock_client = Mock()
    mock_client.exists.side_effect = redis.RedisError("connection refused")
    mock_redis_client.return_value = mock_client

    content_hash = compute_content_hash("Some document.")
    exists = document_hash_exists(content_hash)

    assert exists is False, "Redis errors must default to 'not seen' so the document gets processed"


@patch('dags.utils.hash_store.get_redis_client')
def test_confirm_document_hash_new(mock_redis_client):
    """
    Phase 2: confirming a new hash should call SET NX and return True.
    """
    mock_client = Mock()
    mock_client.set.return_value = True  # NX succeeded — was new
    mock_redis_client.return_value = mock_client

    content_hash = compute_content_hash("A confirmed document.")
    was_new = confirm_document_hash(content_hash, {'filename': 'test.txt'})

    assert was_new is True
    mock_client.set.assert_called_once()
    # Must be called with nx=True — never overwrite an existing confirmation
    _, kwargs = mock_client.set.call_args
    assert kwargs.get('nx') is True


@patch('dags.utils.hash_store.get_redis_client')
def test_confirm_document_hash_already_confirmed(mock_redis_client):
    """
    Phase 2: confirming an already-confirmed hash should return False
    (idempotent — doesn't error, doesn't overwrite).
    """
    mock_client = Mock()
    mock_client.set.return_value = None  # NX failed — key already existed
    mock_redis_client.return_value = mock_client

    content_hash = compute_content_hash("An already-confirmed document.")
    was_new = confirm_document_hash(content_hash, {'filename': 'test.txt'})

    assert was_new is False


@patch('dags.tasks.deduplicate.document_hash_exists')
@patch('dags.utils.metadata_db.record_documents')
def test_deduplicate_documents_all_new(mock_record, mock_hash_exists):
    """
    Test deduplication when no documents have been confirmed yet.
    """
    mock_hash_exists.return_value = False  # nothing confirmed yet

    documents = [
        {'content': 'Document 1', 'source_uri': 'uri1', 'filename': 'doc1.txt'},
        {'content': 'Document 2', 'source_uri': 'uri2', 'filename': 'doc2.txt'},
    ]

    new_docs = deduplicate_documents(documents)

    assert len(new_docs) == 2, "All documents should be new"
    # Every returned doc must carry its content_hash for downstream tasks
    assert all('content_hash' in d for d in new_docs)


@patch('dags.tasks.deduplicate.document_hash_exists')
@patch('dags.utils.metadata_db.record_documents')
def test_deduplicate_documents_some_duplicates(mock_record, mock_hash_exists):
    """
    Test deduplication when one document was already confirmed as stored.
    """
    # First document: not confirmed (new). Second: already confirmed (duplicate).
    mock_hash_exists.side_effect = [False, True]

    documents = [
        {'content': 'Document 1', 'source_uri': 'uri1', 'filename': 'doc1.txt'},
        {'content': 'Document 2', 'source_uri': 'uri2', 'filename': 'doc2.txt'},
    ]

    new_docs = deduplicate_documents(documents)

    assert len(new_docs) == 1, "Only the unconfirmed document should be new"
    assert new_docs[0]['filename'] == 'doc1.txt', "First document should be included"


@patch('dags.tasks.deduplicate.document_hash_exists')
@patch('dags.utils.metadata_db.record_documents')
def test_deduplicate_documents_never_writes(mock_record, mock_hash_exists):
    """
    Structural guarantee: deduplicate_documents must be strictly read-only
    with respect to Redis. It calls document_hash_exists() (read) and must
    never call confirm_document_hash() (write) — that only happens later,
    in upsert_to_qdrant, after a confirmed success.
    """
    mock_hash_exists.return_value = False

    documents = [
        {'content': 'Document 1', 'source_uri': 'uri1', 'filename': 'doc1.txt'},
    ]

    with patch('dags.utils.hash_store.confirm_document_hash') as mock_confirm:
        deduplicate_documents(documents)
        mock_confirm.assert_not_called()


def test_deduplicate_empty_list():
    """
    Test deduplication with empty document list.
    """
    documents = []
    new_docs = deduplicate_documents(documents)

    assert new_docs == [], "Empty input should return empty list"