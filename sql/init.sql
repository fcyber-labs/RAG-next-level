-- RAG Pipeline — PostgreSQL Metadata Schema
-- Run once against the rag_metadata database to create tables and functions.
-- Safe to re-run (all CREATE statements use IF NOT EXISTS / OR REPLACE).

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ── documents ────────────────────────────────────────────────────────────────
-- One row per unique document (deduplicated by content_hash).

CREATE TABLE IF NOT EXISTS documents (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    source_type     VARCHAR(50)  NOT NULL,   -- 'filesystem' | 's3' | 'url'
    source_uri      TEXT         NOT NULL,
    filename        VARCHAR(500),
    content_hash    VARCHAR(64)  NOT NULL,   -- SHA-256 of raw content
    file_size_bytes INTEGER,
    created_at      TIMESTAMP    DEFAULT NOW(),
    last_updated    TIMESTAMP    DEFAULT NOW(),
    UNIQUE(content_hash)
);

CREATE INDEX IF NOT EXISTS idx_documents_source_type  ON documents(source_type);
CREATE INDEX IF NOT EXISTS idx_documents_content_hash ON documents(content_hash);
CREATE INDEX IF NOT EXISTS idx_documents_created_at   ON documents(created_at DESC);

-- ── chunks ───────────────────────────────────────────────────────────────────
-- One row per text chunk; links back to the parent document and to Qdrant.

CREATE TABLE IF NOT EXISTS chunks (
    id               UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id      UUID REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index      INTEGER NOT NULL,
    total_chunks     INTEGER NOT NULL,
    chunk_text       TEXT    NOT NULL,
    token_count      INTEGER,
    embedding_model  VARCHAR(100),
    qdrant_point_id  VARCHAR(100),
    created_at       TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_chunks_created_at  ON chunks(created_at DESC);

-- ── eval_results ─────────────────────────────────────────────────────────────
-- One row per evaluation run (Recall@K, MRR, latency).

CREATE TABLE IF NOT EXISTS eval_results (
    id                   UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    run_id               VARCHAR(100),
    collection_name      VARCHAR(100) NOT NULL,
    total_queries        INTEGER      NOT NULL,
    recall_at_1          NUMERIC(5,4),
    recall_at_5          NUMERIC(5,4),
    recall_at_10         NUMERIC(5,4),
    mrr                  NUMERIC(5,4),
    avg_query_latency_ms NUMERIC(10,2),
    passed_threshold     BOOLEAN,
    threshold_value      NUMERIC(5,4),
    created_at           TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_eval_results_created_at   ON eval_results(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_eval_results_collection   ON eval_results(collection_name);

-- ── ingestion_log ─────────────────────────────────────────────────────────────
-- One row per pipeline run — the audit trail.

CREATE TABLE IF NOT EXISTS ingestion_log (
    id                     UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    run_id                 VARCHAR(100) NOT NULL,
    dag_id                 VARCHAR(100),
    execution_date         TIMESTAMP,
    documents_extracted    INTEGER DEFAULT 0,
    documents_deduplicated INTEGER DEFAULT 0,
    chunks_created         INTEGER DEFAULT 0,
    chunks_embedded        INTEGER DEFAULT 0,
    vectors_upserted       INTEGER DEFAULT 0,
    status                 VARCHAR(20),   -- 'running' | 'success' | 'failed' | 'rolled_back'
    error_message          TEXT,
    started_at             TIMESTAMP DEFAULT NOW(),
    completed_at           TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_ingestion_log_run_id     ON ingestion_log(run_id);
CREATE INDEX IF NOT EXISTS idx_ingestion_log_status     ON ingestion_log(status);
CREATE INDEX IF NOT EXISTS idx_ingestion_log_started_at ON ingestion_log(started_at DESC);

-- ── Helper: latest recall score ───────────────────────────────────────────────

CREATE OR REPLACE FUNCTION get_latest_recall()
RETURNS NUMERIC AS $$
    SELECT recall_at_5
    FROM   eval_results
    ORDER  BY created_at DESC
    LIMIT  1;
$$ LANGUAGE SQL;

-- ── Helper: pipeline health summary ──────────────────────────────────────────
-- FIX: original query mixed aggregates (COUNT) with non-aggregate columns
-- without GROUP BY, causing a PostgreSQL error.  Rewritten using subqueries.

CREATE OR REPLACE FUNCTION get_pipeline_health()
RETURNS TABLE (
    last_run_time    TIMESTAMP,
    last_status      VARCHAR,
    last_recall      NUMERIC,
    total_documents  BIGINT,
    total_chunks     BIGINT
) AS $$
    SELECT
        latest.completed_at                        AS last_run_time,
        latest.status                              AS last_status,
        (SELECT recall_at_5
         FROM   eval_results
         WHERE  run_id = latest.run_id
         ORDER  BY created_at DESC
         LIMIT  1)                                 AS last_recall,
        (SELECT COUNT(*) FROM documents)           AS total_documents,
        (SELECT COUNT(*) FROM chunks)              AS total_chunks
    FROM (
        SELECT run_id, status, completed_at
        FROM   ingestion_log
        WHERE  completed_at IS NOT NULL
        ORDER  BY completed_at DESC
        LIMIT  1
    ) latest;
$$ LANGUAGE SQL;

COMMENT ON TABLE documents      IS 'All processed documents with dedup metadata';
COMMENT ON TABLE chunks         IS 'Text chunks linked to documents and Qdrant points';
COMMENT ON TABLE eval_results   IS 'Retrieval evaluation metrics per pipeline run';
COMMENT ON TABLE ingestion_log  IS 'Audit log of every pipeline execution';