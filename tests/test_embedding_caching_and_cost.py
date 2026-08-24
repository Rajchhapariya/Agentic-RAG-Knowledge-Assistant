"""
Unit and Integration Tests for Persistent Embedding Caching, Safety Modes, and Cost Instrumentation.
"""

import os
import json
import pytest
import numpy as np
from pathlib import Path

from src.config import (
    OPENAI_EMBEDDING_MODEL,
    EMBEDDING_DIM,
    CHUNKS_JSON_PATH,
    DOC_EMBEDDINGS_CACHE_PATH,
    DOC_EMBEDDINGS_METADATA_PATH,
    QUERY_EMBEDDINGS_CACHE_PATH,
    EMBEDDINGS_CACHE_PATH,
)
from src.retrieval.embeddings import EmbeddingClient
from src.retrieval.vector_store import NumpyVectorStore
from src.models.document import Chunk, ChunkMetadata
from src.utils.cost_tracker import (
    CostTracker,
    get_cost_tracker,
    reset_cost_tracker,
    OpenAIAPIDisabledError,
    EMBEDDING_COST_PER_1M,
    GPT4O_MINI_PROMPT_COST_PER_1M,
    GPT4O_MINI_COMPLETION_COST_PER_1M,
)


def test_document_embeddings_persisted():
    """Test 1: Verify document embeddings are persisted on disk with 243 chunk vectors."""
    assert DOC_EMBEDDINGS_CACHE_PATH.exists() or EMBEDDINGS_CACHE_PATH.exists(), "Document embeddings NPZ file must exist."
    target_path = DOC_EMBEDDINGS_CACHE_PATH if DOC_EMBEDDINGS_CACHE_PATH.exists() else EMBEDDINGS_CACHE_PATH
    data = np.load(target_path, allow_pickle=True)
    assert len(data.files) == 243, f"Expected 243 cached chunk vectors, found {len(data.files)}"
    sample_key = data.files[0]
    vec = data[sample_key]
    assert vec.shape == (1536,), f"Expected vector shape (1536,), got {vec.shape}"
    assert np.isclose(np.linalg.norm(vec), 1.0, atol=1e-3), "Vectors must be L2-normalized"


def test_embedding_metadata_structure():
    """Test 2: Verify metadata file contains model, dimension, chunk count, and version."""
    assert DOC_EMBEDDINGS_METADATA_PATH.exists(), "Metadata JSON file must exist."
    with open(DOC_EMBEDDINGS_METADATA_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)

    assert meta["embedding_model"] == OPENAI_EMBEDDING_MODEL
    assert meta["dimensions"] == EMBEDDING_DIM
    assert meta["num_chunks"] == 243
    assert meta["corpus_version"] == "v1.0"
    assert "saved_at" in meta


def test_loading_retriever_zero_api_calls():
    """Test 3: Demonstrate that initializing the retriever and vector store makes 0 API calls."""
    reset_cost_tracker()
    tracker = get_cost_tracker()

    with open(CHUNKS_JSON_PATH, "r", encoding="utf-8") as f:
        chunks_data = json.load(f)
    chunks = [Chunk(**c) for c in chunks_data]

    client = EmbeddingClient()
    vec_store = NumpyVectorStore(embedding_client=client)
    vec_store.build_index(chunks)

    assert tracker.doc_embedding_api_calls == 0, "Loading cached embeddings must make 0 document embedding API calls."
    assert tracker.doc_chunks_cached == 243, "All 243 chunks must be identified as cache hits."
    assert tracker.doc_cost == 0.0, "Cached document loading must incur $0.00 cost."


@pytest.mark.integration
def test_query_embedding_cache_hit_and_miss():
    """Test 4 [INTEGRATION]: Verify query embedding caching, hits, misses, and disk persistence.
    
    Requires: live OpenAI API access (makes 1 real query embedding call).
    Run with: pytest -m integration
    """
    reset_cost_tracker()
    tracker = get_cost_tracker()

    test_q_cache_path = Path("data/processed_chunks/test_query_cache.npz")
    if test_q_cache_path.exists():
        test_q_cache_path.unlink()

    client = EmbeddingClient(query_cache_path=test_q_cache_path)

    # Invalidate memory cache for test query
    q_text = "What is the core contribution of Self-RAG?"
    q_key = client._hash_query(q_text)
    if q_key in client._query_cache:
        del client._query_cache[q_key]

    # First call -> Cache Miss (API call)
    vec1 = client.embed_query(q_text)
    assert vec1.shape == (1536,)
    assert tracker.query_cache_misses == 1
    assert tracker.query_cache_hits == 0
    assert tracker.query_embedding_api_calls == 1

    # Second call with same query -> 100% Cache Hit (0 API calls)
    vec2 = client.embed_query(q_text)
    assert np.allclose(vec1, vec2)
    assert tracker.query_cache_hits == 1
    assert tracker.query_embedding_api_calls == 1, "Second call must NOT make an API call."

    # Third call with case-insensitive whitespace variation -> Cache Hit
    vec3 = client.embed_query("  what is the core contribution of self-rag?  ")
    assert np.allclose(vec1, vec3)
    assert tracker.query_cache_hits == 2
    assert tracker.query_embedding_api_calls == 1

    # Clean up test file
    if test_q_cache_path.exists():
        test_q_cache_path.unlink()


@pytest.mark.integration
def test_safety_mode_openai_api_disabled():
    """Test 5 [INTEGRATION]: Verify OPENAI_API_ENABLED=False raises OpenAIAPIDisabledError on uncached queries.
    
    Requires: live OpenAI API access to verify boundary (tests the real client safety path).
    Run with: pytest -m integration
    """
    client = EmbeddingClient(api_enabled=False)

    # Cached query or cached doc chunks load fine
    # But an uncached query must raise OpenAIAPIDisabledError
    with pytest.raises(OpenAIAPIDisabledError) as exc_info:
        client.embed_query("Uncached novel query for safety mode verification 999")
    assert "OpenAI API is disabled" in str(exc_info.value)


def test_cache_only_mode():
    """Test 6: Verify CACHE_ONLY_MODE=True prevents external network API calls."""
    client = EmbeddingClient(cache_only=True)

    with pytest.raises(OpenAIAPIDisabledError) as exc_info:
        client.embed_query("Another completely unseen query string for cache-only check 888")
    assert "CACHE_ONLY_MODE=True" in str(exc_info.value) or "OpenAI API is disabled" in str(exc_info.value)


def test_local_retrieval_remains_local():
    """Test 7: Verify NumPy similarity search operates 100% locally on pre-computed vectors."""
    client = EmbeddingClient()
    vec_store = NumpyVectorStore(embedding_client=client)

    # Create dummy chunks
    dummy_chunks = [
        Chunk(
            chunk_id=f"dummy_{i}",
            content=f"Dummy chunk content {i}",
            metadata=ChunkMetadata(
                chunk_id=f"dummy_{i}",
                doc_id="Dummy_Doc",
                document_title="Dummy Doc",
                authors="Author A",
                year=2024,
                section_id="sec1",
                section_title="1 Intro",
                chunk_index_in_doc=i,
                chunk_index_in_section=i,
                token_count=10,
                char_start=0,
                char_end=20
            )
        )
        for i in range(5)
    ]
    # Set dummy pre-computed normalized matrix
    vec_store.chunks = dummy_chunks
    vec_store.chunk_ids = [c.chunk_id for c in dummy_chunks]
    vec_store.chunks_by_id = {c.chunk_id: c for c in dummy_chunks}
    vec_store.vectors = np.random.randn(5, 1536).astype(np.float32)
    vec_store.vectors /= np.linalg.norm(vec_store.vectors, axis=1, keepdims=True)

    # Query vector
    q_vec = np.random.randn(1536).astype(np.float32)
    q_vec /= np.linalg.norm(q_vec)

    results = vec_store.search_by_vector(q_vec, top_k=3)
    assert len(results) == 3
    assert all(isinstance(r[0], Chunk) for r in results)
    assert all(isinstance(r[1], float) for r in results)
    assert results[0][1] >= results[1][1] >= results[2][1], "Results must be sorted descending by cosine similarity"


def test_cost_tracker_instrumentation_math():
    """Test 8: Verify exact token and cost calculation across all components."""
    tracker = CostTracker()

    # Track doc embedding: 10,000 tokens
    tracker.track_doc_embedding(num_chunks=20, estimated_tokens=10_000, from_cache=False)
    # 10,000 / 1M * 0.02 = $0.000200
    assert np.isclose(tracker.doc_cost, 0.000200, atol=1e-6)

    # Track query embedding: 500 tokens (miss) + 1 hit
    tracker.track_query_embedding(estimated_tokens=500, is_cache_hit=False)
    tracker.track_query_embedding(estimated_tokens=500, is_cache_hit=True)
    # 500 / 1M * 0.02 = $0.000010
    assert np.isclose(tracker.query_cost, 0.000010, atol=1e-6)

    # Track LLM planner: 1,000 prompt tokens, 200 completion tokens
    # Prompt: 1000/1M * 0.15 = 0.000150; Completion: 200/1M * 0.60 = 0.000120; Total = 0.000270
    tracker.track_llm("planner", prompt_tokens=1000, completion_tokens=200)
    assert np.isclose(tracker.llm_usage["planner"]["cost"], 0.000270, atol=1e-6)

    # Check summary dict
    summary = tracker.get_summary_dict()
    assert summary["document_embeddings"]["api_calls"] == 1
    assert summary["document_embeddings"]["tokens"] == 10000
    assert summary["query_embeddings"]["cache_hits"] == 1
    assert summary["query_embeddings"]["cache_misses"] == 1
    assert summary["query_embeddings"]["cache_hit_rate"] == 0.5
    assert summary["llm_breakdown"]["planner"]["calls"] == 1

    # Check markdown table string
    table_str = tracker.get_summary_table()
    assert "Document Embeddings (Offline)" in table_str
    assert "Query Embeddings (Online)" in table_str
    assert "LLM: Planner" in table_str
    assert "TOTAL USAGE & COST" in table_str
