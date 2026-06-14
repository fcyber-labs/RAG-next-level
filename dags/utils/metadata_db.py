"""
PostgreSQL metadata store for the RAG pipeline.

This module is the missing link between the schema defined in
``sql/init.sql`` (documents, chunks, eval_results, ingestion_log) and the
Airflow tasks. Qdrant remains the source of truth for vectors and Redis
remains the source of truth for "have we seen this content hash before";
this module is the durable audit trail / reporting layer that Grafana and
the Streamlit dashboard read from (``get_pipeline_health()``,
``get_latest_recall()``).

Design notes
------------
* ``psycopg2`` is imported lazily inside each function (same pattern as
  ``mlflow_logger.py``) so that importing this module never breaks Airflow's
  DagBag parsing if the driver or DB is temporarily unavailable.
* Every public function swallows its own exceptions and logs them. Metadata
  logging must never fail the pipeline - if Postgres is down, the run should
  still complete and update Qdrant/MLflow/Slack as normal.
* Connection settings come from environment variables so the same module
  works locally (``localhost``) and inside docker-compose (``postgres``).
"""

import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _get_connection():
    """
    Open a new connection to the metadata database.

    Uses a dedicated database (default: ``rag_metadata``) on the same
    Postgres instance that backs Airflow, so the pipeline's audit tables
    stay separate from Airflow's internal tables.
    """
    import psycopg2

    return psycopg2.connect(
        host=os.getenv("RAG_METADATA_DB_HOST", os.getenv("POSTGRES_HOST", "postgres")),
        port=int(os.getenv("RAG_METADATA_DB_PORT", os.getenv("POSTGRES_PORT", 5432))),
        dbname=os.getenv("RAG_METADATA_DB_NAME", "rag_metadata"),
        user=os.getenv("RAG_METADATA_DB_USER", "airflow"),
        password=os.getenv("RAG_METADATA_DB_PASSWORD", "airflow"),
    )


# ---------------------------------------------------------------------------
# ingestion_log
# ---------------------------------------------------------------------------

def log_ingestion_start(
    run_id: str,
    dag_id: str,
    execution_date: Optional[str] = None,
) -> Optional[str]:
    """
    Insert a new row into ``ingestion_log`` marking the start of a pipeline run.

    Args:
        run_id: Airflow run id (e.g. ``{{ run_id }}``)
        dag_id: DAG id
        execution_date: ISO timestamp of the logical execution date

    Returns:
        The UUID of the new ``ingestion_log`` row as a string, or ``None``
        if the insert failed (e.g. DB unavailable). Callers should treat
        ``None`` as "metadata logging disabled for this run" and continue.
    """
    try:
        conn = _get_connection()
        try:
            with conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO ingestion_log
                            (run_id, dag_id, execution_date, status, started_at)
                        VALUES (%s, %s, %s, %s, NOW())
                        RETURNING id;
                        """,
                        (run_id, dag_id, execution_date or datetime.now(timezone.utc), "running"),
                    )
                    log_id = cur.fetchone()[0]
            logger.info(f"ingestion_log started: id={log_id} run_id={run_id}")
            return str(log_id)
        finally:
            conn.close()
    except Exception as e:
        logger.error(f"log_ingestion_start failed: {e}")
        return None


def log_ingestion_complete(
    log_id: Optional[str],
    documents_extracted: int = 0,
    documents_deduplicated: int = 0,
    chunks_created: int = 0,
    chunks_embedded: int = 0,
    vectors_upserted: int = 0,
    status: str = "success",
    error_message: Optional[str] = None,
) -> bool:
    """
    Update the ``ingestion_log`` row for this run with final counts/status.

    Args:
        log_id: The id returned by :func:`log_ingestion_start`. If ``None``
            (e.g. the start call failed earlier), this is a no-op.
        status: One of ``'success'``, ``'failed'``, ``'rolled_back'``.

    Returns:
        ``True`` if the row was updated, ``False`` otherwise.
    """
    if not log_id:
        logger.debug("log_ingestion_complete skipped: no log_id")
        return False

    try:
        conn = _get_connection()
        try:
            with conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        UPDATE ingestion_log
                        SET documents_extracted = %s,
                            documents_deduplicated = %s,
                            chunks_created = %s,
                            chunks_embedded = %s,
                            vectors_upserted = %s,
                            status = %s,
                            error_message = %s,
                            completed_at = NOW()
                        WHERE id = %s;
                        """,
                        (
                            documents_extracted,
                            documents_deduplicated,
                            chunks_created,
                            chunks_embedded,
                            vectors_upserted,
                            status,
                            error_message,
                            log_id,
                        ),
                    )
            logger.info(f"ingestion_log completed: id={log_id} status={status}")
            return True
        finally:
            conn.close()
    except Exception as e:
        logger.error(f"log_ingestion_complete failed: {e}")
        return False


# ---------------------------------------------------------------------------
# documents
# ---------------------------------------------------------------------------

def record_documents(documents: List[Dict[str, Any]]) -> int:
    """
    Upsert document metadata rows.

    Each ``documents`` item is expected to have at least:
    ``content_hash``, ``source_type``, ``source_uri``, ``filename``, and
    optionally ``file_size_bytes``.

    On conflict (same ``content_hash``), ``last_updated`` is refreshed -
    this lets ``ingestion_log`` and ``documents`` stay consistent even
    though Redis (not this table) is what actually drives deduplication.

    Args:
        documents: List of document dicts (already deduplicated, i.e. the
            output of ``deduplicate_documents``)

    Returns:
        Number of rows successfully written (0 on failure).
    """
    if not documents:
        return 0

    try:
        conn = _get_connection()
        try:
            written = 0
            with conn:
                with conn.cursor() as cur:
                    for doc in documents:
                        content_hash = doc.get("content_hash")
                        if not content_hash:
                            continue
                        cur.execute(
                            """
                            INSERT INTO documents
                                (source_type, source_uri, filename, content_hash, file_size_bytes)
                            VALUES (%s, %s, %s, %s, %s)
                            ON CONFLICT (content_hash)
                            DO UPDATE SET last_updated = NOW()
                            RETURNING id;
                            """,
                            (
                                doc.get("source_type", "unknown"),
                                doc.get("source_uri", ""),
                                doc.get("filename", ""),
                                content_hash,
                                doc.get("file_size_bytes"),
                            ),
                        )
                        row = cur.fetchone()
                        if row:
                            # Stash the DB id on the dict so record_chunks()
                            # can link chunks back to this document.
                            doc["_metadata_db_id"] = str(row[0])
                            written += 1
            logger.info(f"record_documents: wrote/updated {written} rows")
            return written
        finally:
            conn.close()
    except Exception as e:
        logger.error(f"record_documents failed: {e}")
        return 0


# ---------------------------------------------------------------------------
# chunks
# ---------------------------------------------------------------------------

def record_chunks(document_id: str, chunks: List[Dict[str, Any]]) -> int:
    """
    Insert chunk metadata rows for a single document.

    Each ``chunks`` item is expected to have at least: ``chunk_index``,
    ``total_chunks``, ``text`` (or ``chunk_text``), and optionally
    ``token_count``, ``embedding_model``, ``qdrant_point_id``.

    Args:
        document_id: UUID (as string) of the parent row in ``documents``.
            Typically ``doc["_metadata_db_id"]`` set by
            :func:`record_documents`.
        chunks: List of chunk dicts for this document.

    Returns:
        Number of chunk rows written (0 on failure or if document_id is empty).
    """
    if not document_id or not chunks:
        return 0

    try:
        conn = _get_connection()
        try:
            written = 0
            with conn:
                with conn.cursor() as cur:
                    for chunk in chunks:
                        cur.execute(
                            """
                            INSERT INTO chunks
                                (document_id, chunk_index, total_chunks, chunk_text,
                                 token_count, embedding_model, qdrant_point_id)
                            VALUES (%s, %s, %s, %s, %s, %s, %s);
                            """,
                            (
                                document_id,
                                chunk.get("chunk_index", 0),
                                chunk.get("total_chunks", 1),
                                chunk.get("text", chunk.get("chunk_text", "")),
                                chunk.get("token_count"),
                                chunk.get("embedding_model"),
                                chunk.get("qdrant_point_id"),
                            ),
                        )
                        written += 1
            logger.info(f"record_chunks: wrote {written} rows for document {document_id}")
            return written
        finally:
            conn.close()
    except Exception as e:
        logger.error(f"record_chunks failed: {e}")
        return 0


# ---------------------------------------------------------------------------
# eval_results
# ---------------------------------------------------------------------------

def record_eval_result(
    run_id: str,
    collection_name: str,
    eval_results: Dict[str, Any],
    threshold_value: float,
) -> bool:
    """
    Insert a row into ``eval_results`` summarizing one evaluation run.

    Args:
        run_id: Airflow run id / MLflow run id
        collection_name: Qdrant collection that was evaluated
        eval_results: The dict returned by
            ``run_eval.run_retrieval_evaluation`` (expects keys
            ``total_queries``, ``recall@1``, ``recall@5``, ``recall@10``,
            ``mrr``, ``avg_query_latency_ms``)
        threshold_value: The configured ``eval_threshold`` for this run

    Returns:
        ``True`` if the row was written, ``False`` otherwise.
    """
    if not eval_results or not eval_results.get("success"):
        logger.debug("record_eval_result skipped: empty or unsuccessful eval_results")
        return False

    recall_at_5 = eval_results.get("recall@5", 0)
    passed = recall_at_5 >= threshold_value

    try:
        conn = _get_connection()
        try:
            with conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO eval_results
                            (run_id, collection_name, total_queries,
                             recall_at_1, recall_at_5, recall_at_10, mrr,
                             avg_query_latency_ms, passed_threshold, threshold_value)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
                        """,
                        (
                            run_id,
                            collection_name,
                            eval_results.get("total_queries", 0),
                            eval_results.get("recall@1", 0),
                            recall_at_5,
                            eval_results.get("recall@10", 0),
                            eval_results.get("mrr", 0),
                            eval_results.get("avg_query_latency_ms", 0),
                            passed,
                            threshold_value,
                        ),
                    )
            logger.info(
                f"record_eval_result: run_id={run_id} recall@5={recall_at_5} "
                f"passed={passed}"
            )
            return True
        finally:
            conn.close()
    except Exception as e:
        logger.error(f"record_eval_result failed: {e}")
        return False


# ---------------------------------------------------------------------------
# Reporting helpers (used by the Streamlit dashboard / health checks)
# ---------------------------------------------------------------------------

def get_pipeline_health() -> Optional[Dict[str, Any]]:
    """
    Call the ``get_pipeline_health()`` SQL function defined in
    ``sql/init.sql`` and return the single result row as a dict.

    Returns:
        Dict with keys ``last_run_time``, ``last_status``, ``last_recall``,
        ``total_documents``, ``total_chunks``, or ``None`` on failure / no data.
    """
    try:
        conn = _get_connection()
        try:
            with conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT * FROM get_pipeline_health();")
                    row = cur.fetchone()
                    if not row:
                        return None
                    columns = [desc[0] for desc in cur.description]
                    return dict(zip(columns, row))
        finally:
            conn.close()
    except Exception as e:
        logger.error(f"get_pipeline_health failed: {e}")
        return None


def get_latest_recall() -> Optional[float]:
    """
    Call the ``get_latest_recall()`` SQL function and return the value.

    Returns:
        The most recent ``recall_at_5`` score, or ``None`` on failure / no data.
    """
    try:
        conn = _get_connection()
        try:
            with conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT get_latest_recall();")
                    row = cur.fetchone()
                    return float(row[0]) if row and row[0] is not None else None
        finally:
            conn.close()
    except Exception as e:
        logger.error(f"get_latest_recall failed: {e}")
        return None