"""
Real-time RAG Knowledge Base Dashboard.
Allows users to search the knowledge base and see results visually.
"""

import streamlit as st
import os
import sys
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dags.tasks.embed import _get_local_embeddings
from dags.tasks.hybrid_search import perform_hybrid_search
from dags.tasks.query_rewriter import rewrite_query
from dags.tasks.reranker import rerank_results
from dags.utils.vector_store import get_qdrant_client, get_collection_info


# Page config
st.set_page_config(
    page_title="RAG Knowledge Base",
    page_icon="🔍",
    layout="wide"
)

# Title
st.title("🔍 RAG Knowledge Base Search")
st.markdown("**Real-time search with hybrid retrieval + reranking**")

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
    "Use Reranking",
    value=True,
    help="Rerank results with cross-encoder"
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
    st.sidebar.error(f"Error: {e}")


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
    with st.spinner("Searching..."):
        try:
            # Step 1: Query rewriting (optional)
            if use_query_rewriting:
                st.info("🔄 Rewriting query...")
                queries = rewrite_query(query, use_llm=False)  # Use rule-based for speed
                st.success(f"Generated {len(queries)} query variations")
                
                with st.expander("View query variations"):
                    for i, q in enumerate(queries, 1):
                        st.write(f"{i}. {q}")
                
                # Use first variation for search (or combine all)
                search_query = queries[0]
            else:
                search_query = query
            
            # Step 2: Generate embedding
            st.info("🧠 Generating embedding...")
            EMBEDDING_MODEL = os.getenv("RAG_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
            if EMBEDDING_MODEL.startswith("text-embedding-"):
                query_embedding = _get_local_embeddings([search_query], EMBEDDING_MODEL)[0]
            else:
                query_embedding = _get_local_embeddings([search_query], EMBEDDING_MODEL)[0]
                        
            # Step 3: Search
            if use_hybrid_search:
                st.info("🔍 Performing hybrid search (BM25 + Vector)...")
                
                # Get all chunks for BM25 (simplified - in production, cache this)
                all_chunks = []
                offset = None
                while True:
                    records, offset = client.scroll(
                        collection_name=collection_name,
                        limit=100,
                        offset=offset,
                        with_payload=True,
                        with_vectors=False
                    )
                    if not records:
                        break
                    for record in records:
                        all_chunks.append({
                            'id': record.id,
                            'text': record.payload.get('text', '')
                        })
                    if offset is None:
                        break
                
                results = perform_hybrid_search(
                    query=search_query,
                    query_vector=query_embedding,
                    collection_name=collection_name,
                    chunks=all_chunks,
                    top_k=top_k * 2  # Get more for reranking
                )
            else:
                st.info("🔍 Performing vector search...")
                from dags.utils.vector_store import search_similar
                results = search_similar(
                    collection_name=collection_name,
                    query_vector=query_embedding,
                    limit=top_k * 2
                )
            
            # Step 4: Reranking (optional)
            if use_reranking and results:
                st.info("📊 Reranking results...")
                results = rerank_results(
                    query=search_query,
                    results=results,
                    top_k=top_k
                )
            else:
                results = results[:top_k]
            
            # Display results
            st.markdown("---")
            st.subheader(f"📄 Found {len(results)} Results")
            
            if not results:
                st.warning("No results found. Try a different query.")
            else:
                for idx, result in enumerate(results, 1):
                    with st.expander(f"**Result #{idx}** - Score: {result.get('combined_score', result.get('score', 0)):.4f}", expanded=(idx == 1)):
                        # Metadata
                        payload = result.get('payload', {})
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.caption(f"**Source:** {payload.get('source', 'unknown')}")
                        with col2:
                            st.caption(f"**File:** {payload.get('filename', 'unknown')}")
                        with col3:
                            st.caption(f"**Chunk:** {payload.get('chunk_index', '?')} / {payload.get('total_chunks', '?')}")
                        
                        # Text content
                        st.markdown("**Text:**")
                        text = payload.get('text', result.get('text', 'No text available'))
                        st.write(text)
                        
                        # Scores breakdown
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
                        
                        # Additional metadata
                        with st.expander("View full metadata"):
                            st.json(payload)
        
        except Exception as e:
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


# Footer
st.markdown("---")
st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
st.caption("Powered by Qdrant + OpenAI + Airflow")