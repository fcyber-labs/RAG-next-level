"""
tests/conftest.py

Stubs for heavy ML/infrastructure packages not installed in the
lightweight unit-test environment, plus sys.path setup so imports work
both locally and in CI.

KEY FIXES in this version:
  1. tenacity.retry is now a REAL passthrough decorator, not a generic
     MagicMock. A generic MagicMock stub silently replaces the decorated
     function's entire body with a mock object — e.g.
     dags/tasks/embed.py's `@retry(...)` on `_get_openai_embeddings` would
     destroy the function, so its `from openai import OpenAI` line never
     even executes. The passthrough decorator preserves the real function.
  2. tiktoken stub uses UTF-8 byte encode/decode — exactly invertible,
     so decode(encode(text)) == text.
  3. torch is NOT stubbed — see reasoning below (prevents a scipy/sklearn
     issubclass() crash when the real scikit-learn package is installed).
  4. dags/ is on sys.path (not just repo root) so `from tasks.xxx import`
     and `from utils.xxx import` — used throughout dags/tasks/*.py —
     resolve exactly like they do when Airflow loads the DAG folder.
"""

import os
import sys
from unittest.mock import MagicMock

# ── 1. sys.path setup ────────────────────────────────────────────────────────
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DAGS_DIR = os.path.join(_REPO_ROOT, "dags")

for _p in (_REPO_ROOT, _DAGS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── 2. Airflow env ────────────────────────────────────────────────────────────
os.environ.setdefault("AIRFLOW_HOME", "/tmp/airflow_test")
os.environ.setdefault(
    "AIRFLOW__DATABASE__SQL_ALCHEMY_CONN",
    "sqlite:////tmp/airflow_test/airflow.db",
)
os.environ.setdefault("AIRFLOW__CORE__LOAD_EXAMPLES", "false")
os.environ.setdefault("AIRFLOW__CORE__UNIT_TEST_MODE", "true")

# ── 3. SQLAlchemy 2.x compat patch (Airflow 2.8.1 passes encoding= kwarg) ────
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


# ── 4. Package stub helpers ──────────────────────────────────────────────────

def _plain_stub(name: str) -> MagicMock:
    s = MagicMock()
    s.__name__ = name
    s.__path__ = []
    s.__spec__ = None
    return s


def _make_tiktoken_stub() -> MagicMock:
    """UTF-8 byte-based encode/decode — exactly invertible."""
    stub = _plain_stub("tiktoken")
    enc = MagicMock()

    def _encode(text, *a, **kw):
        return list(str(text).encode("utf-8"))

    def _decode(tokens, *a, **kw):
        return bytes(tokens).decode("utf-8", errors="ignore")

    enc.encode.side_effect = _encode
    enc.decode.side_effect = _decode
    stub.get_encoding.return_value = enc
    stub.encoding_for_model.return_value = enc
    return stub


def _make_tenacity_stub() -> MagicMock:
    """@retry(...) must be a REAL passthrough decorator (see module docstring).

    stop_after_attempt / wait_exponential are only ever passed as kwargs
    to retry() and never called directly by our code, so they can stay as
    no-op callables that accept anything and return None.
    """
    stub = _plain_stub("tenacity")

    def _retry(*decorator_args, **decorator_kwargs):
        def _decorator(func):
            return func  # preserve the real function, unlike a MagicMock
        return _decorator

    stub.retry = _retry
    stub.stop_after_attempt = lambda *a, **kw: None
    stub.wait_exponential = lambda *a, **kw: None
    stub.wait_fixed = lambda *a, **kw: None
    stub.wait_random = lambda *a, **kw: None
    return stub


# ── 5. Stub registry ──────────────────────────────────────────────────────────
# NOTE: torch / torch.nn deliberately NOT included. Stubbing torch as a
# plain MagicMock makes sys.modules['torch'].Tensor a MagicMock instance
# rather than a real class. scipy's is_torch_array() then calls
# issubclass(numpy.ndarray, Tensor) and crashes with:
#   TypeError: issubclass() arg 2 must be a class, a tuple of classes...
# Not stubbing torch means sys.modules['torch'] raises KeyError, which
# scipy already handles safely (returns False). Verified empirically.
_STUBS = {
    "tiktoken": _make_tiktoken_stub(),
    "tenacity": _make_tenacity_stub(),
    "pypdf": _plain_stub("pypdf"),
    "qdrant_client": _plain_stub("qdrant_client"),
    "qdrant_client.models": _plain_stub("qdrant_client.models"),
    "qdrant_client.http": _plain_stub("qdrant_client.http"),
    "qdrant_client.http.models": _plain_stub("qdrant_client.http.models"),
    "qdrant_client.http.exceptions": _plain_stub("qdrant_client.http.exceptions"),
    "groq": _plain_stub("groq"),
    "sentence_transformers": _plain_stub("sentence_transformers"),
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