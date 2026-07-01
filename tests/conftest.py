"""
tests/conftest.py

Two jobs this file does:
  1. Add repo root to sys.path so `from dags.xxx import yyy` works when
     running `pytest tests/` locally without PYTHONPATH set.
  2. Stub every heavy package (tiktoken, qdrant_client, groq, …) as
     MagicMock BEFORE any test module is imported so collection doesn't fail.

Key fix vs previous version:
  - tiktoken.get_encoding().encode() now returns a real list of ints
    proportional to the text length so _split_text_into_chunks() works.
  - MagicMock default for __len__ is 0, which broke every chunker test.
"""

import os
import sys
from unittest.mock import MagicMock

# ── 1. Repo root on sys.path (fixes local "No module named 'dags'") ─────────
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# ── 2. Airflow env ────────────────────────────────────────────────────────────
os.environ.setdefault("AIRFLOW_HOME", "/tmp/airflow_test")
os.environ.setdefault(
    "AIRFLOW__DATABASE__SQL_ALCHEMY_CONN",
    "sqlite:////tmp/airflow_test/airflow.db",
)
os.environ.setdefault("AIRFLOW__CORE__LOAD_EXAMPLES", "false")
os.environ.setdefault("AIRFLOW__CORE__UNIT_TEST_MODE", "true")

# ── 3. SQLAlchemy 2.x compat patch (Airflow 2.8.1 passes encoding= kwarg) ───
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


# ── 4. Tiktoken — needs a SMART stub ─────────────────────────────────────────
# Default MagicMock.__len__ returns 0, so chunk.py's token-counting loop
# never enters and _split_text_into_chunks() returns [].
# Fix: encode(text) must return a real list whose len() equals word count.
def _make_tiktoken_stub():
    stub = MagicMock()
    stub.__name__ = "tiktoken"
    stub.__path__ = []
    stub.__spec__ = None

    enc = MagicMock()
    # one int per whitespace-delimited word — close enough for chunk tests
    enc.encode.side_effect = (
        lambda text, *a, **kw: list(range(max(1, len(str(text).split()))))
    )
    enc.decode.side_effect = lambda tokens: " ".join(str(t) for t in tokens)
    enc.__len__ = lambda self: 100_000  # vocab size, not used in tests

    stub.get_encoding.return_value = enc
    stub.encoding_for_model.return_value = enc
    return stub


# ── 5. All heavy package stubs ───────────────────────────────────────────────
def _plain_stub(name):
    s = MagicMock()
    s.__name__ = name
    s.__path__ = []
    s.__spec__ = None
    return s


_STUBS = {
    "tiktoken": _make_tiktoken_stub(),          # smart stub — see above
    "tenacity": _plain_stub("tenacity"),
    "qdrant_client": _plain_stub("qdrant_client"),
    "qdrant_client.models": _plain_stub("qdrant_client.models"),
    "qdrant_client.http": _plain_stub("qdrant_client.http"),
    "qdrant_client.http.models": _plain_stub("qdrant_client.http.models"),
    "qdrant_client.http.exceptions": _plain_stub("qdrant_client.http.exceptions"),
    "groq": _plain_stub("groq"),
    "sentence_transformers": _plain_stub("sentence_transformers"),
    "torch": _plain_stub("torch"),
    "torch.nn": _plain_stub("torch.nn"),
    "transformers": _plain_stub("transformers"),
    "openai": _plain_stub("openai"),
    "langchain": _plain_stub("langchain"),
    "langchain.text_splitter": _plain_stub("langchain.text_splitter"),
    "langchain_community": _plain_stub("langchain_community"),
    "langchain_community.document_loaders": _plain_stub("langchain_community.document_loaders"),
    "langchain_community.document_loaders.pdf": _plain_stub("langchain_community.document_loaders.pdf"),
    "bs4": _plain_stub("bs4"),
    "boto3": _plain_stub("boto3"),
    "botocore": _plain_stub("botocore"),
    "botocore.exceptions": _plain_stub("botocore.exceptions"),
    "aiohttp": _plain_stub("aiohttp"),
    "prometheus_client": _plain_stub("prometheus_client"),
    "psycopg2": _plain_stub("psycopg2"),
    "psycopg2.extras": _plain_stub("psycopg2.extras"),
    "psycopg2.extensions": _plain_stub("psycopg2.extensions"),
    "streamlit": _plain_stub("streamlit"),
    "mlflow": _plain_stub("mlflow"),
    "mlflow.tracking": _plain_stub("mlflow.tracking"),
    "lxml": _plain_stub("lxml"),
    "markdownify": _plain_stub("markdownify"),
}

for _pkg, _stub in _STUBS.items():
    if _pkg not in sys.modules:
        sys.modules[_pkg] = _stub