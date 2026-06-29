"""
DAG integrity tests.

FIX: Airflow 2.8.1 calls create_engine(..., encoding='utf-8') inside
settings.configure_orm().  SQLAlchemy 2.x removed that argument — it
raises TypeError before a single test runs.

Solution: patch sqlalchemy.create_engine to silently drop 'encoding'
BEFORE importing anything from airflow, so the binding that airflow's
settings.py picks up via `from sqlalchemy import create_engine` is
already the patched version.
"""

import os
import sys

# ── 1. Point Airflow at a throw-away SQLite DB ───────────────────────────────
os.environ.setdefault("AIRFLOW_HOME", "/tmp/airflow_dag_test")
os.environ.setdefault(
    "AIRFLOW__DATABASE__SQL_ALCHEMY_CONN",
    "sqlite:////tmp/airflow_dag_test/airflow.db",
)
os.environ.setdefault("AIRFLOW__CORE__LOAD_EXAMPLES", "false")
os.environ.setdefault("AIRFLOW__CORE__UNIT_TEST_MODE", "true")

# ── 2. Strip the deprecated `encoding` kwarg before airflow is imported ───────
import sqlalchemy  # noqa: E402  (must come after env vars)

_original_create_engine = sqlalchemy.create_engine


def _create_engine_compat(*args, **kwargs):
    kwargs.pop("encoding", None)       # removed in SQLAlchemy 2.x
    return _original_create_engine(*args, **kwargs)


# Patch at both levels so every `from sqlalchemy import create_engine` gets it
sqlalchemy.create_engine = _create_engine_compat
try:
    import sqlalchemy.engine.create as _ce
    _ce.create_engine = _create_engine_compat
except Exception:
    pass

# ── 3. Now it is safe to import Airflow ──────────────────────────────────────
import pytest                          # noqa: E402
from airflow.models import DagBag      # noqa: E402

DAG_FOLDER = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "dags",
)


@pytest.fixture(scope="module")
def dag_bag():
    """Load the DAG bag once for all tests in this module."""
    os.makedirs("/tmp/airflow_dag_test", exist_ok=True)
    # Run db init so Airflow doesn't complain about missing tables
    try:
        from airflow.utils.db import initdb
        initdb()
    except Exception:
        pass  # not critical for import tests
    return DagBag(dag_folder=DAG_FOLDER, include_examples=False)


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_no_import_errors(dag_bag):
    """All DAG files must load without Python import errors."""
    assert dag_bag.import_errors == {}, (
        f"DAG import errors detected:\n"
        + "\n".join(f"  {f}: {e}" for f, e in dag_bag.import_errors.items())
    )


def test_at_least_one_dag_loaded(dag_bag):
    """The dags/ folder must contain at least one valid DAG."""
    assert len(dag_bag.dags) > 0, (
        f"No DAGs found in {DAG_FOLDER}. "
        "Check that DAG files define a top-level `dag` variable or use @dag."
    )


def test_rag_pipeline_dag_exists(dag_bag):
    """The main RAG refresh pipeline DAG must be present."""
    assert "rag_refresh_pipeline" in dag_bag.dags, (
        f"Expected DAG 'rag_refresh_pipeline' not found. "
        f"Available DAGs: {sorted(dag_bag.dags.keys())}"
    )


def test_all_dags_have_tags(dag_bag):
    """Every DAG should carry at least one tag (good practice)."""
    missing_tags = [
        dag_id
        for dag_id, dag in dag_bag.dags.items()
        if not dag.tags
    ]
    assert not missing_tags, (
        f"DAG(s) missing tags: {missing_tags}. "
        "Add tags=[...] to the DAG definition."
    )


def test_all_dags_have_description(dag_bag):
    """Every DAG should have a non-empty description."""
    missing_desc = [
        dag_id
        for dag_id, dag in dag_bag.dags.items()
        if not dag.description
    ]
    assert not missing_desc, (
        f"DAG(s) missing description: {missing_desc}."
    )