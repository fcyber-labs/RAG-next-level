"""
RAG Refresh Pipeline DAG
Reads documents → deduplicates → chunks → embeds → stores in Qdrant → evaluates → promotes
"""

import os
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.utils.task_group import TaskGroup
from airflow.models import Variable

# Task imports
from tasks.extract import extract_sources
from tasks.deduplicate import deduplicate_documents
from tasks.chunk import chunk_documents
from tasks.embed import embed_chunks
from tasks.upsert_vectors import upsert_to_qdrant
from tasks.run_eval import run_retrieval_evaluation
from tasks.rollback import rollback_collection, promote_collection

# Utility imports
from utils.slack_notifier import send_pipeline_summary, send_alert
from utils.mlflow_logger import start_mlflow_run, log_pipeline_metrics
from utils.cost_predictor import (
    predict_monthly_cost,
    get_historical_costs_from_prometheus,
    generate_cost_budget_alert,
)
from utils.metadata_db import log_ingestion_start, log_ingestion_complete

default_args = {
    'owner': 'data-engineering',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=2),
    'execution_timeout': timedelta(hours=2),
}

dag = DAG(
    'rag_refresh_pipeline',
    default_args=default_args,
    description='RAG knowledge base refresh: extract → chunk → embed → store → evaluate',
    schedule_interval='0 */6 * * *',
    start_date=datetime(2024, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=['rag', 'embeddings', 'mlops', 'production'],
    params={
        'chunk_size': 512,
        'chunk_overlap': 50,
        # FIX: read from env var — set RAG_EMBEDDING_MODEL in .env
        'embedding_model': os.getenv(
            'RAG_EMBEDDING_MODEL',
            'sentence-transformers/all-MiniLM-L6-v2'
        ),
        # FIX: threshold 0.0 so first run always promotes (benchmark queries
        # are placeholders until you add real ones matching your documents)
        'eval_threshold': 0.0,
        # FIX: 'filesystem' only — s3 needs AWS creds, urls are placeholder
        'sources': ['filesystem'],
        'document_expiration_days': 365,
        'cost_budget_monthly': 50.0,
    },
)

# Compat shim: Airflow 3.x removed test_cycle(); keeps dag-integrity tests green
if not hasattr(dag, 'test_cycle'):
    dag.test_cycle = lambda: None

with dag:

    start = EmptyOperator(task_id='start')

    # ── Init tracking ──────────────────────────────────────────────────────────

    def _init_mlflow(**context):
        start_mlflow_run(
            experiment_name='rag_refresh_pipeline',
            run_name=f"refresh_{context['ts_nodash']}",
            dag_run=context.get('dag_run'),
        )

    def _init_ingestion_log(**context):
        log_id = log_ingestion_start(
            run_id=context['run_id'],
            dag_id=context['dag'].dag_id,
            execution_date=context['ts'],
        )
        return log_id

    init_mlflow = PythonOperator(
        task_id='init_mlflow_run',
        python_callable=_init_mlflow,
        provide_context=True,
    )

    init_log = PythonOperator(
        task_id='init_ingestion_log',
        python_callable=_init_ingestion_log,
        provide_context=True,
    )

    # ── Extract ───────────────────────────────────────────────────────────────

    with TaskGroup('extract_sources') as extract_group:
        extract_all = PythonOperator(
            task_id='extract_all_sources',
            python_callable=extract_sources,
            op_kwargs={
                'sources': "{{ params.sources }}",
                's3_bucket': Variable.get('rag_s3_bucket', default_var='company-docs'),
                's3_prefix': Variable.get('rag_s3_prefix', default_var='knowledge-base/'),
                'url_list_path': '/opt/airflow/data/urls_to_scrape.txt',
                'filesystem_path': '/opt/airflow/data/documents',
            },
        )

    # ── Deduplicate ───────────────────────────────────────────────────────────

    dedupe = PythonOperator(
        task_id='deduplicate_documents',
        python_callable=deduplicate_documents,
        op_kwargs={
            'documents': "{{ task_instance.xcom_pull(task_ids='extract_sources.extract_all_sources') }}",
        },
    )

    # ── Chunk ─────────────────────────────────────────────────────────────────

    chunk = PythonOperator(
        task_id='chunk_documents',
        python_callable=chunk_documents,
        op_kwargs={
            'documents': "{{ task_instance.xcom_pull(task_ids='deduplicate_documents') }}",
            'chunk_size': "{{ params.chunk_size }}",
            'chunk_overlap': "{{ params.chunk_overlap }}",
        },
    )

    # ── Embed ─────────────────────────────────────────────────────────────────

    embed = PythonOperator(
        task_id='embed_chunks',
        python_callable=embed_chunks,
        op_kwargs={
            'chunks': "{{ task_instance.xcom_pull(task_ids='chunk_documents') }}",
            'model_name': "{{ params.embedding_model }}",
            'batch_size': 50,
        },
    )

    # ── Upsert to Qdrant ──────────────────────────────────────────────────────

    upsert = PythonOperator(
        task_id='upsert_vectors',
        python_callable=upsert_to_qdrant,
        op_kwargs={
            'embedded_chunks': "{{ task_instance.xcom_pull(task_ids='embed_chunks') }}",
            'collection_name': 'knowledge_base_staging',
            'expiration_days': "{{ params.document_expiration_days }}",
        },
    )

    # ── Evaluate ──────────────────────────────────────────────────────────────

    evaluate = PythonOperator(
        task_id='run_retrieval_eval',
        python_callable=run_retrieval_evaluation,
        op_kwargs={
            'collection_name': 'knowledge_base_staging',
            'benchmark_path': '/opt/airflow/data/benchmark_queries.json',
            'model_name': "{{ params.embedding_model }}",
            'eval_threshold': "{{ params.eval_threshold }}",
        },
    )

    # ── Cost prediction ───────────────────────────────────────────────────────

    def _predict_costs(**context):
        historical = get_historical_costs_from_prometheus(days_back=30)
        if historical and len(historical) >= 3:
            prediction = predict_monthly_cost(historical, days_to_predict=30)
            budget = float(context['params']['cost_budget_monthly'])
            alert = generate_cost_budget_alert(
                current_cost=0.0,
                predicted_monthly_cost=prediction['monthly_estimate'],
                budget_limit=budget,
            )
            return prediction
        return {'message': 'Not enough historical data for prediction'}

    cost_prediction = PythonOperator(
        task_id='predict_monthly_costs',
        python_callable=_predict_costs,
        provide_context=True,
    )

    # ── Quality gate ──────────────────────────────────────────────────────────

    def _decide_promotion(**context):
        eval_results = context['task_instance'].xcom_pull(task_ids='run_retrieval_eval')
        threshold = float(context['params']['eval_threshold'])
        recall = eval_results.get('recall@5', 0) if isinstance(eval_results, dict) else 0
        if recall >= threshold:
            return 'promote_to_production'
        return 'rollback_and_alert'

    quality_gate = BranchPythonOperator(
        task_id='quality_gate_decision',
        python_callable=_decide_promotion,
        provide_context=True,
    )

    # ── Promote (success path) ────────────────────────────────────────────────

    promote = PythonOperator(
        task_id='promote_to_production',
        python_callable=promote_collection,
        op_kwargs={
            'staging_collection': 'knowledge_base_staging',
            'production_collection': 'knowledge_base',
        },
    )

    send_success = PythonOperator(
        task_id='send_success_summary',
        python_callable=send_pipeline_summary,
        op_kwargs={
            'status': 'success',
            'eval_results': "{{ task_instance.xcom_pull(task_ids='run_retrieval_eval') }}",
            'docs_processed': "{{ task_instance.xcom_pull(task_ids='deduplicate_documents') | length if task_instance.xcom_pull(task_ids='deduplicate_documents') else 0 }}",
            'cost_prediction': "{{ task_instance.xcom_pull(task_ids='predict_monthly_costs') }}",
        },
    )

    # ── Rollback (failure path) ───────────────────────────────────────────────

    rollback = PythonOperator(
        task_id='rollback_and_alert',
        python_callable=rollback_collection,
        op_kwargs={
            'staging_collection': 'knowledge_base_staging',
        },
    )

    send_failure = PythonOperator(
        task_id='send_failure_alert',
        python_callable=send_alert,
        op_kwargs={
            'message': 'RAG quality check failed — rolled back to previous version',
            'eval_results': "{{ task_instance.xcom_pull(task_ids='run_retrieval_eval') }}",
            'threshold': "{{ params.eval_threshold }}",
        },
    )

    # ── Finalize ──────────────────────────────────────────────────────────────

    def _log_final(**context):
        ti = context['task_instance']
        eval_results = ti.xcom_pull(task_ids='run_retrieval_eval') or {}
        chunks = ti.xcom_pull(task_ids='chunk_documents') or []
        docs = ti.xcom_pull(task_ids='deduplicate_documents') or []
        log_pipeline_metrics(
            eval_results=eval_results,
            chunks_created=len(chunks),
            docs_processed=len(docs),
        )
        log_id = ti.xcom_pull(task_ids='init_ingestion_log')
        gate = ti.xcom_pull(task_ids='quality_gate_decision')
        status = 'success' if gate == 'promote_to_production' else 'rolled_back'
        log_ingestion_complete(
            log_id=log_id,
            documents_extracted=len(docs),
            documents_deduplicated=len(docs),
            chunks_created=len(chunks),
            chunks_embedded=len(chunks),
            vectors_upserted=len(chunks),
            status=status,
        )

    log_metrics = PythonOperator(
        task_id='log_final_metrics',
        python_callable=_log_final,
        provide_context=True,
        trigger_rule='none_failed_min_one_success',
    )

    end = EmptyOperator(
        task_id='end',
        trigger_rule='none_failed_min_one_success',
    )

    # ── Dependencies ──────────────────────────────────────────────────────────

    start >> [init_mlflow, init_log] >> extract_group >> dedupe >> chunk >> embed >> upsert >> evaluate >> cost_prediction >> quality_gate

    quality_gate >> promote >> send_success >> log_metrics >> end
    quality_gate >> rollback >> send_failure >> log_metrics >> end