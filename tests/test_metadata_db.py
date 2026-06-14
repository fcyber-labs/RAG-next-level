"""
Tests for the Postgres metadata store (dags/utils/metadata_db.py).

All tests mock ``_get_connection`` so no real database is required.
The connection/cursor mocks use ``MagicMock`` so the
``with conn: with conn.cursor() as cur:`` context-manager pattern works.
"""

from unittest.mock import MagicMock, patch

from dags.utils.metadata_db import (
    get_latest_recall,
    get_pipeline_health,
    log_ingestion_complete,
    log_ingestion_start,
    record_chunks,
    record_documents,
    record_eval_result,
)


def _mock_connection(fetchone_return=None, description=None):
    """Build a MagicMock connection that supports the `with conn:` /
    `with conn.cursor() as cur:` pattern used throughout metadata_db.py."""
    conn = MagicMock()
    cursor = MagicMock()

    conn.__enter__.return_value = conn
    conn.cursor.return_value.__enter__.return_value = cursor

    cursor.fetchone.return_value = fetchone_return
    if description is not None:
        cursor.description = description

    return conn, cursor


# ---------------------------------------------------------------------------
# log_ingestion_start / log_ingestion_complete
# ---------------------------------------------------------------------------

@patch("dags.utils.metadata_db._get_connection")
def test_log_ingestion_start_success(mock_get_conn):
    conn, cursor = _mock_connection(fetchone_return=("abc-123",))
    mock_get_conn.return_value = conn

    log_id = log_ingestion_start(run_id="run_1", dag_id="rag_refresh_pipeline")

    assert log_id == "abc-123"
    cursor.execute.assert_called_once()
    args, _ = cursor.execute.call_args
    assert "INSERT INTO ingestion_log" in args[0]
    conn.close.assert_called_once()


@patch("dags.utils.metadata_db._get_connection")
def test_log_ingestion_start_db_unavailable_returns_none(mock_get_conn):
    mock_get_conn.side_effect = Exception("connection refused")

    log_id = log_ingestion_start(run_id="run_1", dag_id="rag_refresh_pipeline")

    assert log_id is None


@patch("dags.utils.metadata_db._get_connection")
def test_log_ingestion_complete_updates_row(mock_get_conn):
    conn, cursor = _mock_connection()
    mock_get_conn.return_value = conn

    ok = log_ingestion_complete(
        log_id="abc-123",
        documents_extracted=10,
        documents_deduplicated=8,
        chunks_created=40,
        chunks_embedded=40,
        vectors_upserted=40,
        status="success",
    )

    assert ok is True
    cursor.execute.assert_called_once()
    args, _kwargs = cursor.execute.call_args
    sql, params = args
    assert "UPDATE ingestion_log" in sql
    assert params[-1] == "abc-123"  # WHERE id = %s is the last param
    conn.close.assert_called_once()


def test_log_ingestion_complete_noop_without_log_id():
    # No DB call should happen at all when log_id is None
    with patch("dags.utils.metadata_db._get_connection") as mock_get_conn:
        ok = log_ingestion_complete(log_id=None, status="success")

    assert ok is False
    mock_get_conn.assert_not_called()


# ---------------------------------------------------------------------------
# record_documents
# ---------------------------------------------------------------------------

@patch("dags.utils.metadata_db._get_connection")
def test_record_documents_writes_rows_and_sets_db_id(mock_get_conn):
    conn, cursor = _mock_connection(fetchone_return=("doc-uuid-1",))
    mock_get_conn.return_value = conn

    documents = [
        {
            "source_type": "filesystem",
            "source_uri": "/data/documents/handbook.pdf",
            "filename": "handbook.pdf",
            "content_hash": "a" * 64,
            "file_size_bytes": 1024,
        }
    ]

    written = record_documents(documents)

    assert written == 1
    assert documents[0]["_metadata_db_id"] == "doc-uuid-1"
    cursor.execute.assert_called_once()
    assert "INSERT INTO documents" in cursor.execute.call_args[0][0]


def test_record_documents_empty_list_is_noop():
    with patch("dags.utils.metadata_db._get_connection") as mock_get_conn:
        written = record_documents([])

    assert written == 0
    mock_get_conn.assert_not_called()


@patch("dags.utils.metadata_db._get_connection")
def test_record_documents_skips_items_without_hash(mock_get_conn):
    conn, cursor = _mock_connection(fetchone_return=("doc-uuid-1",))
    mock_get_conn.return_value = conn

    documents = [{"source_type": "filesystem", "filename": "no_hash.pdf"}]

    written = record_documents(documents)

    assert written == 0
    cursor.execute.assert_not_called()


# ---------------------------------------------------------------------------
# record_chunks
# ---------------------------------------------------------------------------

@patch("dags.utils.metadata_db._get_connection")
def test_record_chunks_writes_one_row_per_chunk(mock_get_conn):
    conn, cursor = _mock_connection()
    mock_get_conn.return_value = conn

    chunks = [
        {"chunk_index": 0, "total_chunks": 2, "text": "first chunk", "token_count": 12},
        {"chunk_index": 1, "total_chunks": 2, "text": "second chunk", "token_count": 10},
    ]

    written = record_chunks("doc-uuid-1", chunks)

    assert written == 2
    assert cursor.execute.call_count == 2
    for call in cursor.execute.call_args_list:
        assert "INSERT INTO chunks" in call[0][0]


def test_record_chunks_noop_without_document_id():
    with patch("dags.utils.metadata_db._get_connection") as mock_get_conn:
        written = record_chunks("", [{"chunk_index": 0, "text": "x"}])

    assert written == 0
    mock_get_conn.assert_not_called()


def test_record_chunks_noop_without_chunks():
    with patch("dags.utils.metadata_db._get_connection") as mock_get_conn:
        written = record_chunks("doc-uuid-1", [])

    assert written == 0
    mock_get_conn.assert_not_called()


# ---------------------------------------------------------------------------
# record_eval_result
# ---------------------------------------------------------------------------

@patch("dags.utils.metadata_db._get_connection")
def test_record_eval_result_writes_row_when_passed(mock_get_conn):
    conn, cursor = _mock_connection()
    mock_get_conn.return_value = conn

    eval_results = {
        "success": True,
        "total_queries": 12,
        "recall@1": 0.80,
        "recall@5": 0.93,
        "recall@10": 0.96,
        "mrr": 0.82,
        "avg_query_latency_ms": 180.5,
    }

    ok = record_eval_result(
        run_id="run_1",
        collection_name="knowledge_base_staging",
        eval_results=eval_results,
        threshold_value=0.75,
    )

    assert ok is True
    cursor.execute.assert_called_once()
    args, _kwargs = cursor.execute.call_args
    sql, params = args
    assert "INSERT INTO eval_results" in sql
    # passed_threshold should be True since recall@5 (0.93) >= threshold (0.75)
    assert params[-2] is True


@patch("dags.utils.metadata_db._get_connection")
def test_record_eval_result_marks_failed_threshold(mock_get_conn):
    conn, cursor = _mock_connection()
    mock_get_conn.return_value = conn

    eval_results = {
        "success": True,
        "total_queries": 12,
        "recall@1": 0.40,
        "recall@5": 0.60,
        "recall@10": 0.70,
        "mrr": 0.50,
        "avg_query_latency_ms": 200.0,
    }

    ok = record_eval_result(
        run_id="run_1",
        collection_name="knowledge_base_staging",
        eval_results=eval_results,
        threshold_value=0.75,
    )

    assert ok is True
    args, _kwargs = cursor.execute.call_args
    _sql, params = args
    # passed_threshold should be False since recall@5 (0.60) < threshold (0.75)
    assert params[-2] is False


def test_record_eval_result_noop_when_unsuccessful():
    with patch("dags.utils.metadata_db._get_connection") as mock_get_conn:
        ok = record_eval_result(
            run_id="run_1",
            collection_name="knowledge_base_staging",
            eval_results={"success": False, "error": "no benchmark queries"},
            threshold_value=0.75,
        )

    assert ok is False
    mock_get_conn.assert_not_called()


# ---------------------------------------------------------------------------
# get_pipeline_health / get_latest_recall
# ---------------------------------------------------------------------------

@patch("dags.utils.metadata_db._get_connection")
def test_get_pipeline_health_returns_dict(mock_get_conn):
    description = [
        ("last_run_time",),
        ("last_status",),
        ("last_recall",),
        ("total_documents",),
        ("total_chunks",),
    ]
    conn, cursor = _mock_connection(
        fetchone_return=("2026-06-13T10:00:00", "success", 0.93, 120, 480),
        description=description,
    )
    mock_get_conn.return_value = conn

    health = get_pipeline_health()

    assert health == {
        "last_run_time": "2026-06-13T10:00:00",
        "last_status": "success",
        "last_recall": 0.93,
        "total_documents": 120,
        "total_chunks": 480,
    }


@patch("dags.utils.metadata_db._get_connection")
def test_get_pipeline_health_returns_none_when_empty(mock_get_conn):
    conn, cursor = _mock_connection(fetchone_return=None)
    mock_get_conn.return_value = conn

    assert get_pipeline_health() is None


@patch("dags.utils.metadata_db._get_connection")
def test_get_latest_recall_returns_float(mock_get_conn):
    conn, cursor = _mock_connection(fetchone_return=(0.93,))
    mock_get_conn.return_value = conn

    assert get_latest_recall() == 0.93


@patch("dags.utils.metadata_db._get_connection")
def test_get_latest_recall_returns_none_when_empty(mock_get_conn):
    conn, cursor = _mock_connection(fetchone_return=(None,))
    mock_get_conn.return_value = conn

    assert get_latest_recall() is None


@patch("dags.utils.metadata_db._get_connection")
def test_get_latest_recall_returns_none_on_db_error(mock_get_conn):
    mock_get_conn.side_effect = Exception("connection refused")

    assert get_latest_recall() is None