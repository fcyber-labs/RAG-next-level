"""
Test embedding generation with mocked API calls.
Tests OpenAI and local model embedding logic.
"""

from unittest.mock import Mock, patch
from dags.tasks.embed import embed_chunks, _get_openai_embeddings


@patch('dags.tasks.embed.OpenAI')
def test_get_openai_embeddings(mock_openai):
    """
    Test OpenAI embedding generation with mocked API.
    """
    # Mock OpenAI response
    mock_client = Mock()
    mock_response = Mock()
    mock_response.data = [
        Mock(embedding=[0.1, 0.2, 0.3]),
        Mock(embedding=[0.4, 0.5, 0.6]),
    ]
    mock_client.embeddings.create.return_value = mock_response
    mock_openai.return_value = mock_client
    
    texts = ["Text one", "Text two"]
    embeddings = _get_openai_embeddings(texts, "text-embedding-3-small")
    
    assert len(embeddings) == 2, "Should return 2 embeddings"
    assert len(embeddings[0]) == 3, "Each embedding should have 3 dimensions"
    assert embeddings[0] == [0.1, 0.2, 0.3], "First embedding should match mock"


@patch('dags.tasks.embed._get_openai_embeddings')
def test_embed_chunks_simple(mock_get_embeddings):
    """
    Test embedding chunks with mocked embedding function.
    """
    # Mock embeddings
    mock_get_embeddings.return_value = [
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
    ]
    
    chunks = [
        {'text': 'Chunk one', 'chunk_index': 0},
        {'text': 'Chunk two', 'chunk_index': 1},
    ]
    
    embedded = embed_chunks(
        chunks=chunks,
        model_name='text-embedding-3-small',
        batch_size=10
    )
    
    assert len(embedded) == 2, "Should embed all chunks"
    assert 'embedding' in embedded[0], "Chunks should have embedding field"
    assert len(embedded[0]['embedding']) == 3, "Embedding should have correct dimensions"


@patch('dags.tasks.embed._get_openai_embeddings')
def test_embed_chunks_batching(mock_get_embeddings):
    """
    Test that embedding handles batching correctly.
    """
    # Mock to return embeddings for each batch
    mock_get_embeddings.side_effect = [
        [[0.1, 0.2]] * 2,  # First batch
        [[0.3, 0.4]] * 2,  # Second batch
    ]
    
    chunks = [
        {'text': f'Chunk {i}', 'chunk_index': i}
        for i in range(4)
    ]
    
    embedded = embed_chunks(
        chunks=chunks,
        model_name='text-embedding-3-small',
        batch_size=2  # Process 2 at a time
    )
    
    assert len(embedded) == 4, "Should embed all chunks"
    assert mock_get_embeddings.call_count == 2, "Should make 2 batch calls"


def test_embed_empty_chunks():
    """
    Test embedding with empty chunk list.
    """
    chunks = []
    embedded = embed_chunks(chunks=chunks, model_name='test-model')
    
    assert embedded == [], "Empty input should return empty list"


@patch('dags.tasks.embed._get_openai_embeddings')
def test_embed_chunks_preserves_metadata(mock_get_embeddings):
    """
    Test that embedding preserves original chunk metadata.
    """
    mock_get_embeddings.return_value = [[0.1, 0.2, 0.3]]
    
    chunks = [{
        'text': 'Test chunk',
        'chunk_index': 0,
        'source': 'test',
        'filename': 'test.txt',
        'metadata': {'custom': 'value'}
    }]
    
    embedded = embed_chunks(chunks=chunks, model_name='text-embedding-3-small')
    
    assert embedded[0]['source'] == 'test', "Should preserve source"
    assert embedded[0]['filename'] == 'test.txt', "Should preserve filename"
    assert embedded[0]['metadata']['custom'] == 'value', "Should preserve metadata"