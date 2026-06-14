"""
MLflow logging for RAG pipeline runs.

mlflow is imported lazily (inside each function body) so Airflow's DagBag can
parse this file without triggering the mlflow → pkg_resources import chain that
breaks in environments where setuptools is not installed.
"""

import logging
import os
from datetime import datetime

logger = logging.getLogger(__name__)


def start_mlflow_run(experiment_name: str, run_name: str, **kwargs):
    try:
        import mlflow
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
        mlflow.set_experiment(experiment_name)
        mlflow.start_run(run_name=run_name)
        mlflow.log_param("start_time", datetime.now().isoformat())
        mlflow.log_param(
            "airflow_dag_id",
            kwargs.get("dag_run", {}).get("dag_id", "unknown"),
        )
        logger.info(f"Started MLflow run: {run_name}")
    except Exception as e:
        logger.error(f"Error starting MLflow run: {e}")


def log_pipeline_metrics(eval_results, chunks_created, docs_processed, **kwargs):
    try:
        import mlflow

        if isinstance(eval_results, str):
            import ast
            eval_results = ast.literal_eval(eval_results)
        if isinstance(chunks_created, str):
            chunks_created = int(chunks_created)
        if isinstance(docs_processed, str):
            docs_processed = int(docs_processed)

        mlflow.log_metric("documents_processed", docs_processed)
        mlflow.log_metric("chunks_created", chunks_created)

        if eval_results:
            mlflow.log_metric("recall_at_1",   eval_results.get("recall@1", 0))
            mlflow.log_metric("recall_at_5",   eval_results.get("recall@5", 0))
            mlflow.log_metric("recall_at_10",  eval_results.get("recall@10", 0))
            mlflow.log_metric("mrr",           eval_results.get("mrr", 0))
            mlflow.log_metric(
                "avg_query_latency_ms",
                eval_results.get("avg_query_latency_ms", 0),
            )

        mlflow.set_tag("pipeline", "rag_refresh")
        mlflow.set_tag("status", "completed")
        mlflow.end_run()
        logger.info("Logged metrics to MLflow and ended run")

    except Exception as e:
        logger.error(f"Error logging to MLflow: {e}")
        try:
            import mlflow
            mlflow.end_run()
        except Exception:
            pass


def log_config_params(chunk_size: int, chunk_overlap: int, embedding_model: str):
    try:
        import mlflow
        mlflow.log_param("chunk_size", chunk_size)
        mlflow.log_param("chunk_overlap", chunk_overlap)
        mlflow.log_param("embedding_model", embedding_model)
        logger.info("Logged config parameters to MLflow")
    except Exception as e:
        logger.error(f"Error logging config to MLflow: {e}")