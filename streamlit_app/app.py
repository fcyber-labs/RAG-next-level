"""
Real-time RAG Knowledge Base Dashboard.
Allows users to search the knowledge base and see results visually.
"""

import streamlit as st
import os
import sys
import json
import time
from datetime import datetime
import markdown

# ── set_page_config MUST be the very first Streamlit call ───────────────────
# Moving it here fixes the crash when Groq isn't installed: the st.warning()
# below would fire before set_page_config and Streamlit would throw
# "set_page_config() can only be called once, and must be called as the
# first Streamlit command in your script."
st.set_page_config(
    page_title="RAG Knowledge Base",
    page_icon="🔍",
    layout="wide"
)

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dags.tasks.hybrid_search import perform_hybrid_search
import dags.tasks.query_rewriter as query_rewriter
from dags.utils.vector_store import get_qdrant_client, get_collection_info
from dags.utils.chunk_cache import get_cached_chunks, set_cached_chunks, get_bm25_index, set_bm25_index
from dags.utils.answer_cache import get_cached_answer, set_cached_answer

# Try to import Groq for LLM answer generation
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    st.warning("Groq library not installed. LLM answer generation disabled.")


# ── Module-level constants ───────────────────────────────────────────────────
# Defined here so both startup pre-warming AND the search block use the same
# value without re-evaluating os.getenv on every search click.
EMBEDDING_MODEL = os.getenv(
    "RAG_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


# ── Streamlit-cached model loaders ───────────────────────────────────────────
# @st.cache_resource loads once per app process and survives all reruns and
# all user sessions. The first call triggers the (slow) load; every subsequent
# call returns the cached object in microseconds.

@st.cache_resource(show_spinner=False)
def _load_embedding_model(model_name: str):
    """Load SentenceTransformer once; cache for the lifetime of the process."""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name)


@st.cache_resource(show_spinner=False)
def _load_reranker_model(model_name: str = RERANKER_MODEL):
    """Load CrossEncoder once; cache for the lifetime of the process."""
    from sentence_transformers import CrossEncoder
    return CrossEncoder(model_name)


# ── Startup model pre-warming ─────────────────────────────────────────────────
# Without this, the embedding model loads on the FIRST user search, causing
# 10-50 s of latency (or longer if the model needs to be downloaded from
# HuggingFace). Pre-warming here means:
#   • First app load:   user sees a brief "Loading model…" banner (~5-50 s),
#                       but ALL subsequent searches start fast.
#   • Process restart:  same one-time warm-up, then fast searches again.
#   • Second+ session:  model already cached → no delay at all.
#
# The session_state flag prevents re-showing the banner on every Streamlit
# rerun (e.g. when the user changes a sidebar toggle).
if "startup_done" not in st.session_state:
    _t_warm_start = time.time()
    with st.status("⏳ Loading embedding model on startup…", expanded=False) as _status:
        st.write(f"Model: `{EMBEDDING_MODEL}`")
        _load_embedding_model(EMBEDDING_MODEL)
        _model_load_s = round(time.time() - _t_warm_start, 2)
        st.write(f"✅ Ready in {_model_load_s}s")
        _status.update(label=f"✅ Model loaded ({_model_load_s}s)", state="complete")
    st.session_state.startup_done  = True
    st.session_state.model_load_s  = _model_load_s

# Title
st.title("🔍 RAG Knowledge Base Search")
st.markdown("**Real-time search with hybrid retrieval + reranking**")


def generate_answer_stream(query: str, context_chunks: list):
    """
    Generate an answer using Groq LLM with streaming.
    
    Args:
        query: User's question
        context_chunks: List of retrieved text chunks
        
    Yields:
        Chunks of the answer as they're generated
    """
    if not GROQ_AVAILABLE or not context_chunks:
        yield "LLM answer generation unavailable. Please check Groq API key or retrieved context."
        return
    
    try:
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        
        # Prepare context from retrieved chunks
        context_text = "\n\n---\n\n".join([
            f"Context {i+1}:\n{chunk.get('payload', {}).get('text', chunk.get('text', ''))}"
            for i, chunk in enumerate(context_chunks[:5])  # Use top 5 chunks for context
        ])
        
        # Estimate tokens for max_tokens
        query_length = len(query)
        context_length = len(context_text)
        estimated_tokens = (query_length + context_length) // 4
        max_tokens = 2048
        
        # Create streaming request
        stream = client.chat.completions.create(
            model='openai/gpt-oss-120b',
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful assistant that answers questions based on the provided context. "
                        "Use ONLY the context provided to answer the question. "
                        "If the context doesn't contain the answer, say 'I don't have enough information to answer this question.' "
                        "Be concise and direct in your answer."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Context:\n{context_text}\n\n"
                        f"Question: {query}\n\n"
                        f"Answer based ONLY on the context above:"
                    ),
                },
            ],
            max_tokens=max_tokens,
            temperature=0.3,
            stream=True,  # ✅ Enable streaming
        )
        
        # Yield chunks as they arrive
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content
    
    except Exception as e:
        yield f"Error generating answer: {str(e)}"


def generate_answer(query: str, context_chunks: list) -> str:
    """
    Generate an answer using Groq LLM based on retrieved context (non-streaming fallback).
    
    Args:
        query: User's question
        context_chunks: List of retrieved text chunks
        
    Returns:
        Generated answer string
    """
    if not GROQ_AVAILABLE or not context_chunks:
        return "LLM answer generation unavailable. Please check Groq API key or retrieved context."
    
    try:
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        
        # Prepare context from retrieved chunks
        context_text = "\n\n---\n\n".join([
            f"Context {i+1}:\n{chunk.get('payload', {}).get('text', chunk.get('text', ''))}"
            for i, chunk in enumerate(context_chunks[:5])  # Use top 5 chunks for context
        ])
        
        # Estimate tokens for max_tokens
        query_length = len(query)
        context_length = len(context_text)
        estimated_tokens = (query_length + context_length) // 4
        max_tokens = 2048
        
        response = client.chat.completions.create(
            model='openai/gpt-oss-120b',
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful assistant that answers questions based on the provided context. "
                        "Use ONLY the context provided to answer the question. "
                        "If the context doesn't contain the answer, say 'I don't have enough information to answer this question.' "
                        "Be concise and direct in your answer."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Context:\n{context_text}\n\n"
                        f"Question: {query}\n\n"
                        f"Answer based ONLY on the context above:"
                    ),
                },
            ],
            max_tokens=max_tokens,
            temperature=0.3,
        )
        
        return response.choices[0].message.content.strip()
    
    except Exception as e:
        return f"Error generating answer: {str(e)}"


# Sidebar - Configuration
st.sidebar.header("⚙️ Configuration")

collection_name = st.sidebar.text_input(
    "Collection Name",
    value="knowledge_base",
    help="Qdrant collection to search"
)

use_hybrid_search = st.sidebar.checkbox(
    "Use Hybrid Search (BM25 + Vector)",
    value=True,
    help="Combines keyword and semantic search"
)

use_query_rewriting = st.sidebar.checkbox(
    "Use Query Rewriting",
    value=False,
    help="Expand query with synonyms/acronyms"
)

use_reranking = st.sidebar.checkbox(
    "Use Reranking (slow on CPU ~5-8s)",
    value=False,
    help="Rerank results with cross-encoder. Accurate but adds 5-8s on CPU — disable for faster search."
)

use_llm_answer = st.sidebar.checkbox(
    "Generate LLM Answer",
    value=True,
    help="Use Groq LLM to generate a concise answer from retrieved context"
)

use_streaming = st.sidebar.checkbox(
    "Use Streaming",
    value=True,
    help="Stream the answer as it's generated (faster perceived response)"
)

use_answer_cache = st.sidebar.checkbox(
    "Cache LLM Answers (Redis)",
    value=True,
    help=(
        "Reuse a recent answer for the same question + retrieval settings "
        "instead of calling the LLM again. Cached answers are labeled "
        "in the UI and skip the LLM call entirely (no streaming)."
    )
)

top_k = st.sidebar.slider(
    "Number of Results",
    min_value=1,
    max_value=20,
    value=5,
    help="How many results to show"
)

# Sidebar - Collection Info
st.sidebar.header("📊 Collection Stats")

try:
    client = get_qdrant_client()
    info = get_collection_info(collection_name)
    
    if info:
        st.sidebar.metric("Total Vectors", f"{info.get('points_count', 0):,}")
        st.sidebar.metric("Collection Status", info.get('status', 'unknown'))
    else:
        st.sidebar.warning(f"Collection '{collection_name}' not found")
except Exception as e:
    st.sidebar.error(f"Qdrant error: {e}")

# Sidebar - Redis / Cache Status
st.sidebar.header("🗄️ Cache Status")
try:
    from dags.utils.hash_store import get_redis_client as _get_rc
    _rc = _get_rc()
    _rc.ping()
    st.sidebar.success("Redis ✅ connected")
    _bm25_key  = f"rag:bm25:{collection_name}"
    _bm25_ok   = bool(_rc.exists(_bm25_key))
    _ans_count = len(_rc.keys(f"rag:semantic_index:{collection_name}:*"))
    st.sidebar.caption(f"BM25 index: {'✅ cached' if _bm25_ok else '❌ run pipeline first'}")
    st.sidebar.caption(f"Cached answers: {_ans_count}"
                       + (" (semantic cache ready)" if _ans_count > 0 else " (no hits yet)"))
except Exception as _re:
    st.sidebar.error(f"Redis ❌  {str(_re)[:60]}")

# Sidebar - Search Timer
st.sidebar.header("⏱ Performance")
_timer_slot = st.sidebar.empty()

if 'last_elapsed' not in st.session_state:
    st.session_state.last_elapsed = None

if st.session_state.last_elapsed is not None:
    _timer_slot.metric("Last Search Latency", f"{st.session_state.last_elapsed:.2f}s")
else:
    _timer_slot.caption("Run a search to see latency")

# Show one-time model load cost so the developer knows what the 51s was
if st.session_state.get('model_load_s') is not None:
    st.sidebar.caption(
        f"Model loaded at startup in {st.session_state.model_load_s}s "
        f"(includes HuggingFace download on first run)"
    )

# Placeholder for per-stage breakdown — populated after each search
_stage_slot = st.sidebar.empty()


# Main search interface
st.markdown("---")

query = st.text_input(
    "🔎 Enter your query:",
    placeholder="e.g., What is the vacation policy?",
    help="Ask a question about your knowledge base"
)

search_button = st.button("Search", type="primary", use_container_width=True)


# Search logic
if search_button and query:
    _t0 = time.time()
    _timer_slot.info("⏳ Searching…")
    with st.spinner("Searching..."):
        try:
            # Step 1: Query rewriting (optional)
            if use_query_rewriting:
                st.info("🔄 Rewriting query...")
                queries = query_rewriter.rewrite_query(query, use_llm=False)
                st.success(f"Generated {len(queries)} query variations")
                
                with st.expander("View query variations"):
                    for i, q in enumerate(queries, 1):
                        st.write(f"{i}. {q}")
                
                search_query = queries[0]
            else:
                search_query = query
            
            # Step 2: Generate embedding
            st.info("🧠 Generating embedding...")
            _t_embed = time.time()
            _embed_model    = _load_embedding_model(EMBEDDING_MODEL)   # cached, ~0ms after startup
            query_embedding = _embed_model.encode([search_query], show_progress_bar=False).tolist()[0]
            _stage_embed_s  = time.time() - _t_embed

            # ══════════════════════════════════════════════════════════════════
            # Step 2.5: SEMANTIC CACHE CHECK — before ANY retrieval.
            #
            # Previously the cache check was in Step 5 (after hybrid search
            # + reranking), so a "cache hit" still paid ~3-4s of retrieval
            # cost. Moved here, a cache hit costs only:
            #   embedding (~200ms) + Redis cosine scan (~10ms) → ~0.3s total.
            # Steps 3 (retrieval), 4 (reranking), and 5 (LLM) are skipped.
            # ══════════════════════════════════════════════════════════════════
            _cache_hit = (
                get_cached_answer(
                    collection_name=collection_name,
                    query_embedding=query_embedding,
                    use_hybrid_search=use_hybrid_search,
                    use_query_rewriting=use_query_rewriting,
                    use_reranking=use_reranking,
                    top_k=top_k,
                )
                if use_answer_cache else None
            )

            _timer_stopped = False   # track whether we've already written latency

            if _cache_hit is not None:
                # ── FAST PATH ──────────────────────────────────────────────
                results     = _cache_hit.get('results', [])
                age_seconds = max(0, int(time.time() - _cache_hit.get('cached_at', time.time())))
                similarity  = _cache_hit.get('similarity', 1.0)
                _cache_label = (
                    f"⚡ Semantic cache hit — similarity {similarity:.3f} · "
                    f"generated {age_seconds}s ago · "
                    f"retrieval + reranking + LLM skipped"
                )

                if use_llm_answer:
                    st.markdown("---")
                    st.subheader("💡 Answer")
                    st.markdown(
                        f"""
                        <div style="
                            background-color: #d4edda;
                            padding: 20px;
                            border-radius: 10px;
                            border-left: 5px solid #28a745;
                            color: #155724;
                            font-size: 16px;
                        ">
                        {_cache_hit['answer']}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    # Badge sits directly under the answer where the user looks
                    st.caption(_cache_label)
                    if results:
                        with st.expander("📚 View Sources"):
                            for idx, result in enumerate(results[:3], 1):
                                payload = result.get("payload", {})
                                st.write(
                                    f"**Source {idx}:** "
                                    f"{payload.get('filename', 'unknown')} "
                                    f"(Score: {result.get('combined_score', result.get('score', 0)):.3f})"
                                )
                else:
                    # No answer box above — show the badge as a banner
                    st.success(_cache_label)

                # Timer stops here — fastest path
                _elapsed = time.time() - _t0
                st.session_state.last_elapsed = _elapsed
                _timer_slot.metric("Last Search Latency", f"{_elapsed:.2f}s")
                _timer_stopped = True

            else:
                # ── FULL PIPELINE PATH ─────────────────────────────────────
                # Step 3: Search
                if use_hybrid_search:
                    import math as _math
                    _candidates = top_k * 2 if use_reranking else top_k
                    _dense_k  = _math.ceil(_candidates / 2)
                    _sparse_k = _candidates - _dense_k
                    if use_reranking:
                        st.info(f"🔍 Hybrid search: {_dense_k} dense + {_sparse_k} sparse candidates → reranking → top {top_k}…")
                    else:
                        st.info(f"🔍 Hybrid search: {_dense_k} dense + {_sparse_k} sparse → top {top_k}…")

                    bm25_index, all_chunks = get_bm25_index(collection_name)

                    if bm25_index is not None:
                        st.caption(
                            f"⚡ Pre-built BM25 index loaded from Redis "
                            f"({len(all_chunks)} chunks)"
                        )
                    else:
                        all_chunks = []
                        offset = None
                        while True:
                            records, offset = client.scroll(
                                collection_name=collection_name,
                                limit=100,
                                offset=offset,
                                with_payload=True,
                                with_vectors=False,
                            )
                            if not records:
                                break
                            for record in records:
                                all_chunks.append({
                                    'id':      str(record.id),
                                    'text':    record.payload.get('text', ''),
                                    'payload': record.payload,
                                })
                            if offset is None:
                                break
                        set_cached_chunks(collection_name, all_chunks)

                    _t_search = time.time()
                    results = perform_hybrid_search(
                        query=search_query,
                        query_vector=query_embedding,
                        collection_name=collection_name,
                        chunks=all_chunks,
                        bm25_index=bm25_index,
                        top_k=_candidates,
                    )
                    _stage_search_s = time.time() - _t_search
                else:
                    st.info("🔍 Performing vector search...")
                    from dags.utils.vector_store import search_similar
                    _t_search = time.time()
                    results = search_similar(
                        collection_name=collection_name,
                        query_vector=query_embedding,
                        limit=top_k * 2 if use_reranking else top_k
                    )
                    _stage_search_s = time.time() - _t_search

                # Step 4: Reranking (optional — CrossEncoder on CPU adds ~5-8s)
                _stage_rerank_s = 0.0
                if use_reranking and results:
                    st.info("📊 Reranking results...")
                    _t_rerank = time.time()
                    _reranker = _load_reranker_model()   # cached by @st.cache_resource
                    _pairs    = [
                        [search_query, r.get('payload', {}).get('text', r.get('text', ''))]
                        for r in results
                    ]
                    _scores = _reranker.predict(_pairs)
                    for r, s in zip(results, _scores):
                        r['reranker_score'] = float(s)
                    results = sorted(results, key=lambda x: x.get('reranker_score', 0), reverse=True)[:top_k]
                    _stage_rerank_s = time.time() - _t_rerank
                else:
                    results = results[:top_k]

                # Step 5: Generate and Display LLM Answer (optional)
                _stage_llm_s = 0.0
                if use_llm_answer and results:
                    st.markdown("---")
                    st.subheader("💡 Answer")
                    answer_placeholder = st.empty()

                    if use_streaming:
                        full_answer = ""
                        _t_llm = time.time()
                        for chunk in generate_answer_stream(query, results):
                            full_answer += chunk
                            answer_placeholder.markdown(
                                f"""
                                <div style="
                                    background-color: #d4edda;
                                    padding: 20px;
                                    border-radius: 10px;
                                    border-left: 5px solid #28a745;
                                    color: #155724;
                                    font-size: 16px;
                                ">
                                {full_answer}
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                        _stage_llm_s = time.time() - _t_llm
                        # Timer stops after last streamed character
                        _elapsed = time.time() - _t0
                        st.session_state.last_elapsed = _elapsed
                        _timer_slot.metric("Last Search Latency", f"{_elapsed:.2f}s")
                        _timer_stopped = True
                        if use_answer_cache:
                            set_cached_answer(
                                collection_name=collection_name,
                                query_embedding=query_embedding,
                                use_hybrid_search=use_hybrid_search,
                                use_query_rewriting=use_query_rewriting,
                                use_reranking=use_reranking,
                                top_k=top_k,
                                answer=full_answer,
                                results=results,
                            )
                    else:
                        _t_llm = time.time()
                        with st.spinner("🤖 Generating answer..."):
                            answer = generate_answer(query, results)
                            answer_placeholder.markdown(
                                f"""
                                <div style="
                                    background-color: #d4edda;
                                    padding: 20px;
                                    border-radius: 10px;
                                    border-left: 5px solid #28a745;
                                    color: #155724;
                                    font-size: 16px;
                                ">
                                {answer}
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                        _stage_llm_s = time.time() - _t_llm
                        # Timer stops after answer fully rendered
                        _elapsed = time.time() - _t0
                        st.session_state.last_elapsed = _elapsed
                        _timer_slot.metric("Last Search Latency", f"{_elapsed:.2f}s")
                        _timer_stopped = True
                        if use_answer_cache:
                            set_cached_answer(
                                collection_name=collection_name,
                                query_embedding=query_embedding,
                                use_hybrid_search=use_hybrid_search,
                                use_query_rewriting=use_query_rewriting,
                                use_reranking=use_reranking,
                                top_k=top_k,
                                answer=answer,
                                results=results,
                            )

                    with st.expander("📚 View Sources"):
                        for idx, result in enumerate(results[:3], 1):
                            payload = result.get("payload", {})
                            st.write(
                                f"**Source {idx}:** "
                                f"{payload.get('filename', 'unknown')} "
                                f"(Score: {result.get('combined_score', result.get('score', 0)):.3f})"
                            )

            # ── Results display (both cache-hit and full-pipeline paths) ──
            st.markdown("---")
            st.subheader(f"📄 Retrieved {len(results)} Results")

            if not results:
                st.warning("No results found. Try a different query.")
            else:
                for idx, result in enumerate(results, 1):
                    with st.expander(
                        f"**Result #{idx}** - Score: {result.get('combined_score', result.get('score', 0)):.4f}",
                        expanded=(idx == 1 and not use_llm_answer)
                    ):
                        payload = result.get('payload', {})

                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.caption(f"**Source:** {payload.get('source', 'unknown')}")
                        with col2:
                            st.caption(f"**File:** {payload.get('filename', 'unknown')}")
                        with col3:
                            st.caption(f"**Chunk:** {payload.get('chunk_index', '?')} / {payload.get('total_chunks', '?')}")

                        st.markdown("**Text:**")
                        st.write(payload.get('text', result.get('text', 'No text available')))

                        st.markdown("**Scores:**")
                        score_cols = st.columns(4)
                        if 'vector_score' in result:
                            score_cols[0].metric("Vector", f"{result['vector_score']:.3f}")
                        if 'bm25_score' in result:
                            score_cols[1].metric("BM25", f"{result['bm25_score']:.3f}")
                        if 'combined_score' in result:
                            score_cols[2].metric("Combined", f"{result['combined_score']:.3f}")
                        if 'reranker_score' in result:
                            score_cols[3].metric("Reranker", f"{result['reranker_score']:.3f}")

                        st.divider()
                        st.markdown("### Full Metadata")
                        st.json(payload)

            # Final timer stop — no-LLM path or any path not yet stopped
            if not _timer_stopped:
                _elapsed = time.time() - _t0
                st.session_state.last_elapsed = _elapsed
                _timer_slot.metric("Last Search Latency", f"{_elapsed:.2f}s")

            # Per-stage breakdown in sidebar — the only way to know what
            # the 51s was. Shows embed / search / rerank / LLM separately.
            if _cache_hit is None:   # only meaningful for full pipeline runs
                _stage_slot.markdown(
                    f"**Stage breakdown**  \n"
                    f"🧠 Embed: `{_stage_embed_s:.2f}s`  \n"
                    f"🔍 Search: `{_stage_search_s:.2f}s`  \n"
                    + (f"📊 Rerank: `{_stage_rerank_s:.2f}s`  \n" if use_reranking else "")
                    + (f"🤖 LLM: `{_stage_llm_s:.2f}s`" if use_llm_answer else "")
                )

        except Exception as e:
            _elapsed = time.time() - _t0
            st.session_state.last_elapsed = _elapsed
            _timer_slot.metric("Last Search Latency", f"{_elapsed:.2f}s ❌")
            error_text = str(e)
            if "doesn't exist" in error_text or "Not found" in error_text or "404" in error_text:
                st.error(f"📪 Collection '{collection_name}' doesn't exist yet.")
                st.info(
                    "This usually means no pipeline run has successfully passed the "
                    "quality gate yet (recall@5 below threshold), so nothing has been "
                    "promoted from staging to production. Try searching "
                    "**knowledge_base_staging** instead (top-left field), or check the "
                    "`run_retrieval_eval` task logs in Airflow to see why retrieval "
                    "quality is low."
                )
            else:
                st.error(f"Error during search: {e}")
                st.exception(e)

elif search_button and not query:
    st.warning("Please enter a query to search.")


st.markdown("---")

# ── Cost & Budget section ─────────────────────────────────────────────────────
def _load_cost_summary():
    """Load cost summary written by the Airflow cost_analysis task."""
    try:
        from dags.utils.hash_store import get_redis_client
        rc  = get_redis_client()
        raw = rc.get("rag:cost_summary")
        if raw:
            return json.loads(raw)
    except Exception:
        pass
    return None

_cost = _load_cost_summary()

if _cost:
    st.subheader("💰 Embedding Cost & Budget")

    _sev_color = {"ok": "🟢", "warning": "🟡", "critical": "🔴"}.get(
        _cost.get("budget_severity", "ok"), "🔵"
    )
    _trend_icon = {"increasing": "📈", "decreasing": "📉", "stable": "➡️"}.get(
        _cost.get("trend", "stable"), "➡️"
    )

    _c1, _c2, _c3 = st.columns(3)
    _c1.metric(
        "Last Run Cost",
        f"${_cost.get('last_run_cost', 0):.4f}",
        help="Embedding cost for the most recent pipeline run"
    )
    _c2.metric(
        "Month-to-Date",
        f"${_cost.get('month_to_date', 0):.4f}",
        help="Cumulative embedding cost this calendar month"
    )
    _c3.metric(
        "Monthly Forecast",
        f"${_cost.get('monthly_forecast', 0):.2f}",
        delta=f"{_trend_icon} {_cost.get('trend', 'unknown')} "
              f"(confidence: {_cost.get('confidence', '?')})",
        help="scikit-learn LinearRegression forecast for the next 30 days"
    )

    # Budget utilisation bar
    _util  = min(_cost.get("budget_utilization", 0), 100)
    _limit = _cost.get("budget_limit", 50)
    st.caption(
        f"{_sev_color} {_cost.get('budget_message', '')}  ·  "
        f"Budget: ${_limit:.0f}/month  ·  "
        f"Forecast model R²: {_cost.get('r2_score', 0):.3f}  ·  "
        f"Based on {_cost.get('historical_points', 0)} data points"
    )
    st.progress(int(_util))

    st.caption(f"Last pipeline run: {_cost.get('updated_at', '—')}")
else:
    st.caption("💰 Cost data will appear here after the first Airflow pipeline run.")

st.caption("Powered by Qdrant + OpenAI + Airflow")