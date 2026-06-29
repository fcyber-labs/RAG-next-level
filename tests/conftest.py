"""
tests/conftest.py

Stubs for heavy ML / infrastructure packages that are NOT installed in
the lightweight CI unit-test job (tiktoken, qdrant_client, groq, etc.).

MagicMock is used so that:
  - `from tiktoken import get_encoding` works (attribute access on mock)
  - `@retry(...)` decorators work (calling a mock returns a mock)
  - `QdrantClient(host=...)` works (instantiating a mock returns a mock)

This file is loaded by pytest automatically before any test module is
imported, so the stubs are in place when test_chunker.py, test_embed.py,
etc. do their top-level imports.
"""

import os
import sys
from unittest.mock import MagicMock

# ── Airflow environment (needed if test_dag_integrity.py is collected) ──────
os.environ.setdefault("AIRFLOW_HOME", "/tmp/airflow_test")
os.environ.setdefault(
    "AIRFLOW__DATABASE__SQL_ALCHEMY_CONN",
    "sqlite:////tmp/airflow_test/airflow.db",
)
os.environ.setdefault("AIRFLOW__CORE__LOAD_EXAMPLES", "false")
os.environ.setdefault("AIRFLOW__CORE__UNIT_TEST_MODE", "true")

# ── SQLAlchemy 2.x compatibility patch for Airflow 2.8.1 ────────────────────
# Airflow passes encoding='utf-8' to create_engine() which was removed in
# SQLAlchemy 2.x.  Strip it before Airflow imports.
try:
    import sqlalchemy

    _orig_create_engine = sqlalchemy.create_engine

    def _compat_create_engine(*args, **kwargs):
        kwargs.pop("encoding", None)
        return _orig_create_engine(*args, **kwargs)

    sqlalchemy.create_engine = _compat_create_engine
    try:
        import sqlalchemy.engine.create as _ce
        _ce.create_engine = _compat_create_engine
    except Exception:
        pass
except ImportError:
    pass


# ── Heavy package stubs ──────────────────────────────────────────────────────

def _make_stub(name: str) -> MagicMock:
    """Return a MagicMock that looks like an importable package."""
    stub = MagicMock()
    stub.__name__ = name
    stub.__path__ = []      # marks it as a package so sub-imports work
    stub.__spec__ = None
    return stub


# Every package that is imported at module level by dags/tasks/*.py or
# dags/utils/*.py but is NOT available in the CI lightweight job.
_STUBS = [
    "tiktoken",
    "tenacity",
    "qdrant_client",
    "qdrant_client.models",
    "qdrant_client.http",
    "qdrant_client.http.models",
    "qdrant_client.http.exceptions",
    "groq",
    "sentence_transformers",
    "torch",
    "torch.nn",
    "transformers",
    "openai",
    "langchain",
    "langchain.text_splitter",
    "langchain_community",
    "langchain_community.document_loaders",
    "langchain_community.document_loaders.pdf",
    "boto3",
    "botocore",
    "botocore.exceptions",
    "aiohttp",
    "prometheus_client",
    "psycopg2",
    "psycopg2.extras",
    "streamlit",
    "mlflow",
    "mlflow.tracking",
]

for _pkg in _STUBS:
    if _pkg not in sys.modules:
        sys.modules[_pkg] = _make_stub(_pkg)