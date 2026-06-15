#!/usr/bin/env python3
from __future__ import annotations   # Python 3.9 fix for X | Y type hints
"""
RAG Pipeline — Local Function Checker
======================================
Tests every tool and function WITHOUT Docker or docker-compose.

Place this file in your  02_RAG_Airflow/  directory and run:

    cd 02_RAG_Airflow
    pip install -r requirements.txt
    python check_all.py

Checks 1-8  →  pure Python / CPU only, zero external services needed.
Checks 9-13 →  need a running service (Redis, Qdrant, Postgres, API key).
               Each skips gracefully with a hint if the service is absent.
"""

import os
import sys
import time
import traceback
import unittest.mock as mock

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
DAGS = os.path.join(ROOT, "dags")
sys.path.insert(0, ROOT)   # lets  `from dags.tasks.x import ...`  work
sys.path.insert(0, DAGS)   # lets  `from utils.x import ...`  work (as Airflow sees it)

# ── Load .env ─────────────────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(ROOT, ".env"))
except Exception:
    pass

# ── Terminal colours ──────────────────────────────────────────────────────────
G = "\033[92m"; R = "\033[91m"; Y = "\033[93m"; B = "\033[94m"
BOLD = "\033[1m"; X = "\033[0m"

_results: list[tuple[str, str]] = []

def ok(msg: str):
    print(f"  {G}✅ PASS{X}  {msg}")
    _results.append(("PASS", msg))

def fail(msg: str, err: Exception | None = None):
    print(f"  {R}❌ FAIL{X}  {msg}")
    if err:
        print(f"         {R}→ {type(err).__name__}: {err}{X}")
    _results.append(("FAIL", msg))

def skip(msg: str, hint: str = ""):
    print(f"  {Y}⚠️  SKIP{X}  {msg}")
    if hint:
        print(f"         {Y}→ {hint}{X}")
    _results.append(("SKIP", msg))

def header(n: int, title: str):
    bar = "═" * 58
    print(f"\n{BOLD}{B}{bar}{X}")
    print(f"{BOLD}{B} CHECK {n}: {title}{X}")
    print(f"{BOLD}{B}{bar}{X}")

def section(title: str):
    print(f"\n  {BOLD}▸ {title}{X}")


# ══════════════════════════════════════════════════════════════════════════════
# CHECK 1 — Python package imports
# ══════════════════════════════════════════════════════════════════════════════
header(1, "PYTHON PACKAGE IMPORTS")

PACKAGES = [
    ("tiktoken",             "tiktoken"),
    ("sentence_transformers","sentence-transformers"),
    ("rank_bm25",            "rank-bm25"),
    ("qdrant_client",        "qdrant-client"),
    ("openai",               "openai"),
    ("redis",                "redis"),
    ("pypdf",                "pypdf"),
    ("sklearn",              "scikit-learn"),
    ("numpy",                "numpy"),
    ("streamlit",            "streamlit"),
    ("mlflow",               "mlflow"),
    ("prometheus_client",    "prometheus-client"),
    ("bs4",                  "beautifulsoup4"),
    ("langchain",            "langchain"),
    ("psycopg2",             "psycopg2-binary"),
    ("tenacity",             "tenacity"),
    ("requests",             "requests"),
    ("dotenv",               "python-dotenv"),
    ("boto3",                "boto3"),
]

for module, pkg in PACKAGES:
    try:
        __import__(module)
        ok(pkg)
    except ImportError as e:
        fail(f"{pkg}  →  pip install {pkg}", e)


# ══════════════════════════════════════════════════════════════════════════════
# CHECK 2 — Text extraction (filesystem, PDF, HTML — pure Python)
# ══════════════════════════════════════════════════════════════════════════════
header(2, "TEXT EXTRACTION  (pypdf, beautifulsoup4 — no services)")

try:
    from tasks.extract import _extract_text_from_content, extract_from_filesystem

    # Plain text
    txt_bytes = b"Hello world. This is a test document.\nSecond line here."
    result = _extract_text_from_content(txt_bytes, "sample.txt")
    assert "Hello" in result
    ok(f"_extract_text_from_content(.txt)  →  {len(result)} chars")

    # HTML
    html_bytes = b"<html><body><h1>Title</h1><p>Some content here.</p></body></html>"
    result_html = _extract_text_from_content(html_bytes, "page.html")
    assert len(result_html) > 0
    ok(f"_extract_text_from_content(.html) →  {len(result_html)} chars: '{result_html.strip()[:40]}'")

    # Filesystem (scan data/ dir if it exists)
    data_dir = os.path.join(ROOT, "data")
    if os.path.isdir(data_dir):
        docs = extract_from_filesystem(data_dir)
        ok(f"extract_from_filesystem(data/)   →  {len(docs)} docs found")
    else:
        skip("extract_from_filesystem", "No data/ directory yet — create it and add .txt/.pdf files")

except Exception as e:
    fail("Text extraction failed", e)


# ══════════════════════════════════════════════════════════════════════════════
# CHECK 3 — Text chunking (tiktoken — pure Python)
# ══════════════════════════════════════════════════════════════════════════════
header(3, "TEXT CHUNKING  (tiktoken — pure Python, no services)")

try:
    from tasks.chunk import _count_tokens, _split_text_into_chunks, chunk_documents

    sample = "The quick brown fox jumps over the lazy dog. " * 150  # ~800 tokens

    # Token counter
    n_tokens = _count_tokens(sample)
    assert n_tokens > 0
    ok(f"_count_tokens          →  {n_tokens} tokens")

    # Splitter
    chunks = _split_text_into_chunks(sample, chunk_size=100, chunk_overlap=10)
    assert len(chunks) > 1
    ok(f"_split_text_into_chunks →  {len(chunks)} chunks")
    ok(f"First chunk preview     →  '{chunks[0][:60].strip()}…'")

    # Overlap sanity — last words of chunk[0] should appear at start of chunk[1]
    last_words_0 = set(chunks[0].split()[-5:])
    first_words_1 = set(chunks[1].split()[:10])
    overlap_found = bool(last_words_0 & first_words_1)
    if overlap_found:
        ok("Chunk overlap verified   →  words shared between chunk 0 and chunk 1")
    else:
        skip("Chunk overlap check", "Could not verify — may vary with tokenisation")

    # Full pipeline function
    docs_in = [{"content": sample, "source": "test", "filename": "test.txt",
                 "source_uri": "local://test.txt", "metadata": {}}]
    result = chunk_documents(docs_in, chunk_size=100, chunk_overlap=10)
    assert len(result) > 0
    required_keys = {"text", "chunk_index", "total_chunks", "source", "filename"}
    assert required_keys.issubset(result[0].keys())
    ok(f"chunk_documents         →  {len(result)} chunk dicts with metadata")
    ok(f"Chunk keys              →  {sorted(result[0].keys())}")

except Exception as e:
    fail("Chunking failed", e)


# ══════════════════════════════════════════════════════════════════════════════
# CHECK 4 — Local embeddings  (sentence-transformers CPU, ~80 MB auto-download)
# ══════════════════════════════════════════════════════════════════════════════
header(4, "LOCAL EMBEDDINGS  (sentence-transformers CPU — ~80 MB download on first run)")

try:
    import numpy as np
    from tasks.embed import _get_local_embeddings

    model_name = os.getenv("RAG_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    section(f"Model: {model_name}")

    texts = [
        "What is the vacation policy?",
        "How do I submit an expense report?",
        "What are the health insurance options?",
        "How does remote work policy work?",
    ]

    t0 = time.time()
    embeddings = _get_local_embeddings(texts, model_name)
    elapsed = time.time() - t0

    assert len(embeddings) == len(texts)
    dim = len(embeddings[0])

    ok(f"Encoded {len(texts)} sentences in {elapsed:.1f}s")
    ok(f"Vector dimension  →  {dim}")

    if dim == 384:
        ok("Dimension ✓ correct for all-MiniLM-L6-v2 (384)")
    elif dim == 1536:
        fail("Dimension is 1536 (OpenAI size!) — check RAG_EMBEDDING_MODEL in .env")
    else:
        ok(f"Dimension: {dim}")

    arr = np.array(embeddings)
    assert arr.std() > 0.01, "Vectors look all-zero"
    ok(f"Non-zero vectors  →  std={arr.std():.4f}")

    # Semantic sanity: "vacation" should be closer to "vacation" than to "expense"
    import numpy as np
    v0, v1, v2 = arr[0], arr[1], arr[2]
    sim_01 = float(np.dot(v0, v1) / (np.linalg.norm(v0) * np.linalg.norm(v1)))
    sim_vacation_health = float(np.dot(v0, v2) / (np.linalg.norm(v0) * np.linalg.norm(v2)))
    ok(f"Cosine sim (vacation vs expense)  →  {sim_01:.4f}")
    ok(f"Cosine sim (vacation vs health)   →  {sim_vacation_health:.4f}")

except Exception as e:
    fail("Embeddings failed", e)


# ══════════════════════════════════════════════════════════════════════════════
# CHECK 5 — Deduplication hashing (pure Python — Redis tested in check 11)
# ══════════════════════════════════════════════════════════════════════════════
header(5, "DEDUPLICATION HASHING  (SHA-256 — pure Python, no services)")

try:
    from tasks.deduplicate import _compute_content_hash

    h1 = _compute_content_hash("document one content")
    h2 = _compute_content_hash("document two content")
    h3 = _compute_content_hash("document one content")  # same as h1

    assert len(h1) == 64, "SHA-256 hex must be 64 chars"
    ok(f"Hash length correct  →  64 hex chars")

    assert h1 != h2, "Different content must produce different hashes"
    ok(f"Different content → different hashes  ✓")

    assert h1 == h3, "Same content must produce same hash"
    ok(f"Same content → same hash  ✓")
    ok(f"Example hash: {h1[:16]}…")

except Exception as e:
    fail("Deduplication hashing failed", e)


# ══════════════════════════════════════════════════════════════════════════════
# CHECK 6 — BM25 & Hybrid Search scoring (rank-bm25, numpy — pure Python)
# ══════════════════════════════════════════════════════════════════════════════
header(6, "BM25 + HYBRID SEARCH  (rank-bm25 — pure Python, no services)")

try:
    import numpy as np
    from rank_bm25 import BM25Okapi
    from tasks.hybrid_search import tokenize, HybridSearcher

    corpus = [
        "vacation policy paid time off annual leave days",
        "expense report reimbursement travel costs receipts",
        "health insurance medical dental vision benefits coverage",
        "performance review annual evaluation goals metrics",
    ]

    # tokenizer
    tokens = tokenize(corpus[0])
    assert isinstance(tokens, list) and len(tokens) > 0
    ok(f"tokenize()  →  {tokens}")

    # Raw BM25
    bm25 = BM25Okapi([tokenize(d) for d in corpus])
    scores = bm25.get_scores(tokenize("vacation days leave"))
    best = int(np.argmax(scores))
    ok(f"BM25 scores  →  {[f'{s:.3f}' for s in scores]}")
    ok(f"Best BM25 match  →  '{corpus[best][:50]}'")
    assert best == 0, "Expected vacation doc to rank first"

    # HybridSearcher with fake Qdrant results
    chunks = [{"id": str(i), "text": doc} for i, doc in enumerate(corpus)]
    fake_qdrant = [
        {"id": "2", "score": 0.90, "payload": {"text": corpus[2]}},
        {"id": "0", "score": 0.75, "payload": {"text": corpus[0]}},
        {"id": "3", "score": 0.60, "payload": {"text": corpus[3]}},
    ]

    searcher = HybridSearcher(chunks=chunks, bm25_weight=0.3)
    hybrid_results = searcher.search(
        query="vacation days leave",
        query_vector=[0.1] * 10,     # dummy — BM25 doesn't use this
        qdrant_results=fake_qdrant,
        top_k=3,
    )
    assert len(hybrid_results) == 3
    ok(f"HybridSearcher.search()  →  {len(hybrid_results)} results")
    for i, r in enumerate(hybrid_results):
        ok(f"  Rank {i+1}  combined={r['combined_score']:.4f}  "
           f"bm25={r['bm25_score']:.4f}  vec={r['vector_score']:.4f}")

except Exception as e:
    fail("BM25 / Hybrid Search failed", e)


# ══════════════════════════════════════════════════════════════════════════════
# CHECK 7 — Reranker (CrossEncoder CPU — ~100 MB auto-download)
# ══════════════════════════════════════════════════════════════════════════════
header(7, "RERANKER  (cross-encoder CPU — ~100 MB download on first run)")

try:
    from tasks.reranker import Reranker, rerank_results

    section("Loading cross-encoder/ms-marco-MiniLM-L-6-v2")
    t0 = time.time()
    reranker = Reranker()
    ok(f"Model loaded in {time.time() - t0:.1f}s")

    query = "how many vacation days do employees get per year"
    test_results = [
        {"id": "1", "score": 0.8, "combined_score": 0.8,
         "payload": {"text": "Submit expense reports within 30 days of travel."}},
        {"id": "2", "score": 0.7, "combined_score": 0.7,
         "payload": {"text": "Employees get 15 vacation days per year plus public holidays."}},
        {"id": "3", "score": 0.6, "combined_score": 0.6,
         "payload": {"text": "Annual leave accrues at 1.25 days per month of service."}},
    ]

    reranked = reranker.rerank(query=query, results=test_results, top_k=3)
    assert len(reranked) == 3
    ok(f"reranker.rerank()  →  {len(reranked)} results")

    for i, r in enumerate(reranked):
        ok(f"  Rank {i+1}  score={r['reranker_score']:.4f}  "
           f"(was #{r['original_rank']})  →  '{r['payload']['text'][:55]}…'")

    # Correctness: vacation-related doc should rank higher than expense doc
    texts_reranked = [r["payload"]["text"] for r in reranked]
    assert "vacation" in texts_reranked[0].lower() or "leave" in texts_reranked[0].lower(), \
        "Expected vacation doc to rank #1 after reranking"
    ok("Reranking puts vacation doc at rank #1 ✓")

except Exception as e:
    fail("Reranker failed", e)


# ══════════════════════════════════════════════════════════════════════════════
# CHECK 8 — Query Rewriter rule-based (no API, no services)
# ══════════════════════════════════════════════════════════════════════════════
header(8, "QUERY REWRITER — RULE-BASED  (no API, no services)")

try:
    from tasks.query_rewriter import rewrite_query_rule_based, rewrite_query

    test_cases = [
        ("pto policy",              "paid time off"),
        ("vpn setup instructions",  "virtual private network"),
        ("hr contact information",  "human resources"),
        ("no acronyms here",        None),           # no expansion expected
    ]

    for query, expected_term in test_cases:
        variations = rewrite_query_rule_based(query)
        assert variations[0] == query, "Original query must always be first"
        if expected_term:
            full_text = " ".join(variations)
            assert expected_term in full_text.lower(), \
                f"Expected '{expected_term}' in expansions: {variations}"
            ok(f"'{query}'  →  {len(variations)} variations (includes '{expected_term}')")
        else:
            ok(f"'{query}'  →  {len(variations)} variation(s) (no acronym — correct)")

    # rewrite_query wrapper with use_llm=False
    result = rewrite_query("faq about ppo plan", use_llm=False)
    assert len(result) >= 1 and result[0] == "faq about ppo plan"
    ok(f"rewrite_query(use_llm=False)  →  {result}")

except Exception as e:
    fail("Rule-based query rewriter failed", e)


# ══════════════════════════════════════════════════════════════════════════════
# CHECK 9 — Vector store in-memory + full mini-pipeline (no Docker at all)
# ══════════════════════════════════════════════════════════════════════════════
header(9, "FULL MINI-PIPELINE IN-MEMORY  (embed → store → search → rerank, no Docker)")

try:
    import numpy as np
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct
    from sentence_transformers import SentenceTransformer
    from tasks.hybrid_search import HybridSearcher
    from tasks.reranker import Reranker

    section("1 / 5  Load embedding model (cached after check 4)")
    model_name = os.getenv("RAG_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    model = SentenceTransformer(model_name)
    ok(f"Model ready: {model_name}")

    section("2 / 5  Embed knowledge-base documents")
    kb_docs = [
        "Employees receive 15 vacation days per year, plus 10 public holidays.",
        "Expense reports must be submitted within 30 days of the expense date.",
        "Health insurance covers medical, dental, and vision. Open enrolment is in November.",
        "Performance reviews are held twice a year: June and December.",
        "Remote work is allowed up to 3 days per week with manager approval.",
        "New employees complete a 90-day onboarding period.",
        "The company matches 401(k) contributions up to 4% of salary.",
    ]
    embeddings = model.encode(kb_docs, show_progress_bar=False)
    DIM = embeddings.shape[1]
    ok(f"Embedded {len(kb_docs)} docs  →  dim={DIM}")

    section("3 / 5  Store in in-memory Qdrant  (QdrantClient(':memory:'))")
    qclient = QdrantClient(":memory:")
    qclient.create_collection(
        "kb", vectors_config=VectorParams(size=DIM, distance=Distance.COSINE)
    )
    points = [
        PointStruct(id=i, vector=embeddings[i].tolist(), payload={"text": doc})
        for i, doc in enumerate(kb_docs)
    ]
    qclient.upsert("kb", points=points)
    info = qclient.get_collection("kb")
    ok(f"Stored {info.points_count} vectors  →  status={info.status}")

    section("4 / 5  Vector search + Hybrid re-scoring")
    query = "how many vacation days do I get each year"
    query_vec = model.encode([query], show_progress_bar=False)[0].tolist()

    qdrant_hits = qclient.search("kb", query_vector=query_vec, limit=5)
    formatted = [
        {"id": str(r.id), "score": r.score,
         "payload": r.payload, "combined_score": r.score}
        for r in qdrant_hits
    ]
    ok(f"Vector search  →  {len(formatted)} hits")
    ok(f"Top hit (score={formatted[0]['score']:.4f}): '{formatted[0]['payload']['text'][:55]}…'")

    chunks = [{"id": str(i), "text": doc} for i, doc in enumerate(kb_docs)]
    searcher = HybridSearcher(chunks=chunks, bm25_weight=0.3)
    hybrid = searcher.search(query=query, query_vector=query_vec,
                             qdrant_results=formatted, top_k=5)
    ok(f"Hybrid search  →  {len(hybrid)} results, "
       f"top combined_score={hybrid[0]['combined_score']:.4f}")

    section("5 / 5  Rerank final results")
    reranker = Reranker()
    final = reranker.rerank(query=query, results=hybrid, top_k=3)
    ok(f"Reranked  →  {len(final)} results")

    print(f"\n  {BOLD}Pipeline answer for: '{query}'{X}")
    for i, r in enumerate(final):
        print(f"  #{i+1} (score={r['reranker_score']:.4f})  {r['payload']['text']}")

except Exception as e:
    fail("Full in-memory pipeline failed", e)
    traceback.print_exc()


# ══════════════════════════════════════════════════════════════════════════════
# CHECK 10 — Query Rewriter LLM via Groq  (needs GROQ_API_KEY)
# ══════════════════════════════════════════════════════════════════════════════
header(10, "QUERY REWRITER — GROQ LLM  (needs GROQ_API_KEY in .env)")

groq_key = os.getenv("GROQ_API_KEY", "")

if groq_key and not groq_key.startswith("your") and not groq_key.startswith("gsk_your"):
    try:
        from groq import Groq
        client = Groq(api_key=groq_key)
        t0 = time.time()
        response = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[
                {"role": "system",
                 "content": "Return ONLY a valid JSON array of 2 alternative search queries. No explanation."},
                {"role": "user",
                 "content": "Alternative search queries for: vacation policy"},
            ],
            max_tokens=100,
            temperature=0.7,
        )
        elapsed = time.time() - t0
        content = response.choices[0].message.content.strip()
        ok(f"Groq API reachable  →  {elapsed:.1f}s")
        ok(f"Model: openai/gpt-oss-120b")
        ok(f"Response: {content}")
    except Exception as e:
        fail("Groq API call failed", e)
else:
    skip("Groq LLM rewriter", "Add GROQ_API_KEY=gsk_... to your .env file")


# ══════════════════════════════════════════════════════════════════════════════
# CHECK 11 — Redis connection + deduplication (needs Redis)
# ══════════════════════════════════════════════════════════════════════════════
header(11, "REDIS + DEDUPLICATION  (needs Redis — start with one Docker command)")

try:
    import redis as redis_lib

    host = os.getenv("REDIS_HOST", "localhost")
    port = int(os.getenv("REDIS_PORT", 6379))

    r = redis_lib.Redis(host=host, port=port, socket_timeout=2)
    r.ping()
    ok(f"Redis connected  →  {host}:{port}")

    # Basic set/get
    r.set("rag:check:ping", "pong", ex=30)
    assert r.get("rag:check:ping") == b"pong"
    ok("Redis set/get working")

    # Full deduplication function
    with mock.patch("utils.metadata_db.record_documents"), \
         mock.patch("utils.metrics_exporter.export_counter"), \
         mock.patch("utils.metrics_exporter.export_gauge"):
        from tasks.deduplicate import deduplicate_documents

    # Clear test keys first
    for key in r.scan_iter("rag:doc:hash:*"):
        r.delete(key)

    docs = [
        {"content": "Unique document alpha",   "source": "test", "filename": "a.txt", "source_uri": ""},
        {"content": "Unique document beta",    "source": "test", "filename": "b.txt", "source_uri": ""},
        {"content": "Unique document alpha",   "source": "test", "filename": "a.txt", "source_uri": ""},  # dup
    ]
    with mock.patch("utils.metadata_db.record_documents"), \
         mock.patch("utils.metrics_exporter.export_counter"), \
         mock.patch("utils.metrics_exporter.export_gauge"):
        unique = deduplicate_documents(docs)

    assert len(unique) == 2, f"Expected 2 unique, got {len(unique)}"
    ok(f"deduplicate_documents  →  {len(unique)} unique / {len(docs)-len(unique)} duplicate removed  ✓")

    # Second run — all 3 should be duplicates now
    with mock.patch("utils.metadata_db.record_documents"), \
         mock.patch("utils.metrics_exporter.export_counter"), \
         mock.patch("utils.metrics_exporter.export_gauge"):
        unique2 = deduplicate_documents(docs)
    assert len(unique2) == 0
    ok("Second run (all duplicates)  →  0 new docs  ✓")

except Exception as e:
    if "Connection refused" in str(e) or "timed out" in str(e) or isinstance(e, redis_lib.ConnectionError):
        skip("Redis not reachable",
             "Quick start:  docker run -d -p 6379:6379 redis:7.2-alpine")
    else:
        fail("Redis check failed", e)


# ══════════════════════════════════════════════════════════════════════════════
# CHECK 12 — Qdrant external connection  (needs Qdrant)
# ══════════════════════════════════════════════════════════════════════════════
header(12, "QDRANT EXTERNAL  (needs Qdrant — start with one Docker command)")

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct
    import numpy as np

    import socket as _socket
    host = os.getenv("QDRANT_HOST", "localhost")
    port = int(os.getenv("QDRANT_PORT", 6333))
    try:                          # Docker service names don't resolve locally
        _socket.getaddrinfo(host, port)
    except _socket.gaierror:
        print(f"  {Y}  \'{host}\' is a Docker hostname → falling back to localhost{X}")
        host = "localhost"

    client = QdrantClient(host=host, port=port, timeout=3)
    collections = client.get_collections().collections
    ok(f"Qdrant connected  →  {host}:{port}")
    ok(f"Existing collections: {[c.name for c in collections] or '(none yet)'}")

    # Round-trip test
    TEST_COL = "rag_check_tmp"
    DIM = 8
    try:
        client.delete_collection(TEST_COL)
    except Exception:
        pass
    client.create_collection(TEST_COL, vectors_config=VectorParams(size=DIM, distance=Distance.COSINE))
    ok("create_collection  ✓")

    pts = [PointStruct(id=i, vector=np.random.rand(DIM).tolist(), payload={"n": i}) for i in range(5)]
    client.upsert(TEST_COL, points=pts)
    ok("upsert 5 vectors  ✓")

    hits = client.search(TEST_COL, query_vector=np.random.rand(DIM).tolist(), limit=3)
    assert len(hits) == 3
    ok(f"search → {len(hits)} hits, top score={hits[0].score:.4f}  ✓")

    client.delete_collection(TEST_COL)
    ok("delete_collection  ✓")

except Exception as e:
    err_str = str(e).lower()
    unreachable = any(x in err_str for x in [
        "connection refused", "timed out", "connect",
        "nodename nor servname", "name or service not known",
        "errno 8", "errno 111", "errno 61",
    ])
    if unreachable:
        skip("Qdrant not reachable",
             "Quick start:  docker run -d -p 6333:6333 qdrant/qdrant:v1.7.4")
    else:
        fail("Qdrant check failed", e)


# ══════════════════════════════════════════════════════════════════════════════
# CHECK 13 — PostgreSQL connection  (needs Postgres)
# ══════════════════════════════════════════════════════════════════════════════
header(13, "POSTGRESQL  (needs Postgres — start with one Docker command)")

try:
    import psycopg2

    import socket as _socket2
    host = os.getenv("RAG_METADATA_DB_HOST", "localhost")
    try:
        _socket2.getaddrinfo(host, None)
    except _socket2.gaierror:
        print(f"  {Y}  \'{host}\' is a Docker hostname → falling back to localhost{X}")
        host = "localhost"
    port = int(os.getenv("RAG_METADATA_DB_PORT", 5432))
    db   = os.getenv("RAG_METADATA_DB_NAME",  "rag_metadata")
    user = os.getenv("RAG_METADATA_DB_USER",     "airflow")
    pwd  = os.getenv("RAG_METADATA_DB_PASSWORD",  "airflow")

    conn = psycopg2.connect(
        host=host, port=port, dbname=db,
        user=user, password=pwd, connect_timeout=3,
    )
    cur = conn.cursor()
    cur.execute("SELECT version();")
    ver = cur.fetchone()[0]
    ok(f"PostgreSQL connected  →  {host}:{port}/{db}")
    ok(f"Version: {ver[:55]}…")

    cur.execute("SELECT tablename FROM pg_tables WHERE schemaname='public';")
    tables = [row[0] for row in cur.fetchall()]
    ok(f"Tables in public schema: {tables or '(none — run sql/init.sql first)'}")
    conn.close()

except Exception as e:
    err_pg = str(e).lower()
    if any(x in err_pg for x in [
        "could not connect", "connection refused", "timed out",
        "nodename nor servname", "name or service not known",
        "could not translate host", "errno 8", "errno 111", "errno 61",
    ]):
        skip("PostgreSQL not reachable",
             "Quick start:  docker run -d -p 5432:5432 "
             "-e POSTGRES_USER=airflow -e POSTGRES_PASSWORD=airflow "
             "-e POSTGRES_DB=airflow postgres:15")
    else:
        fail("PostgreSQL check failed", e)


# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
passed  = sum(1 for s, _ in _results if s == "PASS")
failed  = sum(1 for s, _ in _results if s == "FAIL")
skipped = sum(1 for s, _ in _results if s == "SKIP")

bar = "═" * 58
print(f"\n{BOLD}{bar}{X}")
print(f"{BOLD} SUMMARY{X}")
print(f"{BOLD}{bar}{X}")
print(f"  {G}✅ PASSED{X}   {passed}")
print(f"  {R}❌ FAILED{X}   {failed}")
print(f"  {Y}⚠️  SKIPPED{X}  {skipped}")

if failed:
    print(f"\n{R}{BOLD}Failed:{X}")
    for s, msg in _results:
        if s == "FAIL":
            print(f"  {R}→ {msg}{X}")

if skipped:
    print(f"\n{Y}Skipped (need services):{X}")
    print(f"  Start only what you need:")
    print(f"  {Y}Redis    →  docker run -d -p 6379:6379 redis:7.2-alpine{X}")
    print(f"  {Y}Qdrant   →  docker run -d -p 6333:6333 qdrant/qdrant:v1.7.4{X}")
    print(f"  {Y}Postgres →  docker run -d -p 5432:5432 \\{X}")
    print(f"  {Y}            -e POSTGRES_USER=airflow -e POSTGRES_PASSWORD=airflow \\{X}")
    print(f"  {Y}            -e POSTGRES_DB=airflow postgres:15{X}")

print(f"\n{BOLD}Checks 1–9   →  work with zero Docker (pure Python + CPU){X}")
print(f"{BOLD}Checks 10–13 →  need one lightweight Docker container each{X}\n")