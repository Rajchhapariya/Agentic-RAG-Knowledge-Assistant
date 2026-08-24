"""
Hybrid Retriever: Unifies dense vector similarity and sparse BM25 search
using Reciprocal Rank Fusion (RRF) with metadata filtering.
"""

import time
from typing import List, Dict, Any, Optional, Literal, Tuple
import numpy as np
from src.config import DEFAULT_TOP_K, RRF_K
from src.models.document import Chunk
from src.models.retrieval import SearchResult, RetrievalQuery, RetrievalTrace
from src.retrieval.vector_store import NumpyVectorStore
from src.retrieval.bm25_engine import BM25SearchEngine


class HybridRetriever:
    """
    Unified retrieval engine supporting Dense, Sparse (BM25), and Hybrid (RRF) search.
    Provides a clean, modular interface for Baseline A, Baseline B, and Agentic RAG.
    """

    def __init__(
        self,
        vector_store: NumpyVectorStore,
        bm25_engine: BM25SearchEngine,
        rrf_k: int = RRF_K,
    ):
        self.vector_store = vector_store
        self.bm25_engine = bm25_engine
        self.rrf_k = rrf_k
        self.chunks_by_id: Dict[str, Chunk] = vector_store.chunks_by_id

    @classmethod
    def from_chunks(cls, chunks: List[Chunk], rrf_k: int = RRF_K) -> "HybridRetriever":
        """Factory method to initialize and index retriever directly from a chunk list."""
        vec_store = NumpyVectorStore()
        vec_store.build_index(chunks)
        
        bm25_eng = BM25SearchEngine()
        bm25_eng.build_index(chunks)
        
        return cls(vector_store=vec_store, bm25_engine=bm25_eng, rrf_k=rrf_k)

    @classmethod
    def from_chunks_and_vectors(
        cls,
        chunks: List[Chunk],
        vectors: np.ndarray,
        rrf_k: int = RRF_K
    ) -> "HybridRetriever":
        """Factory method to initialize retriever from pre-computed vectors and chunks (0 API calls)."""
        vec_store = NumpyVectorStore()
        vec_store.load_index(chunks, vectors)
        
        bm25_eng = BM25SearchEngine()
        bm25_eng.build_index(chunks)
        
        return cls(vector_store=vec_store, bm25_engine=bm25_eng, rrf_k=rrf_k)

    def retrieve(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        mode: Literal["dense", "sparse", "hybrid"] = "hybrid",
        filters: Optional[Dict[str, Any]] = None,
        dense_weight: float = 1.0,
        sparse_weight: float = 1.0,
    ) -> List[SearchResult]:
        """
        Main retrieval entry point.
        
        Args:
            query: Natural language or keyword query string.
            top_k: Number of ranked chunks to return.
            mode: 'dense' (cosine sim only), 'sparse' (BM25 only), or 'hybrid' (RRF).
            filters: Metadata attribute filters (e.g. {'doc_id': 'CRAG_Yan_2024'}).
            dense_weight: Multiplier weight for dense rank in fusion.
            sparse_weight: Multiplier weight for sparse rank in fusion.
            
        Returns:
            List of SearchResult objects sorted by final rank.
        """
        start_time = time.perf_counter()
        
        if mode == "dense":
            results = self._retrieve_dense(query, top_k, filters)
        elif mode == "sparse":
            results = self._retrieve_sparse(query, top_k, filters)
        elif mode == "hybrid":
            results = self._retrieve_hybrid(
                query=query,
                top_k=top_k,
                filters=filters,
                dense_weight=dense_weight,
                sparse_weight=sparse_weight
            )
        else:
            raise ValueError(f"Unknown retrieval mode: {mode}. Use 'dense', 'sparse', or 'hybrid'.")

        latency_ms = (time.perf_counter() - start_time) * 1000.0
        return results

    def _retrieve_dense(
        self, query: str, top_k: int, filters: Optional[Dict[str, Any]]
    ) -> List[SearchResult]:
        """Executes pure dense vector cosine retrieval."""
        dense_hits = self.vector_store.search(query, top_k=top_k, filters=filters)
        results = []
        for rank, (chunk, score, d_rank) in enumerate(dense_hits, start=1):
            results.append(SearchResult(
                chunk_id=chunk.chunk_id,
                doc_id=chunk.metadata.doc_id,
                document_title=chunk.metadata.document_title,
                section_id=chunk.metadata.section_id,
                section_title=chunk.metadata.section_title,
                content=chunk.content,
                dense_score=round(score, 4),
                dense_rank=d_rank,
                bm25_score=None,
                bm25_rank=None,
                rrf_score=None,
                final_rank=rank,
                metadata=chunk.metadata.model_dump()
            ))
        return results

    def _retrieve_sparse(
        self, query: str, top_k: int, filters: Optional[Dict[str, Any]]
    ) -> List[SearchResult]:
        """Executes pure sparse BM25 retrieval."""
        sparse_hits = self.bm25_engine.search(query, top_k=top_k, filters=filters)
        results = []
        for rank, (chunk, score, b_rank) in enumerate(sparse_hits, start=1):
            results.append(SearchResult(
                chunk_id=chunk.chunk_id,
                doc_id=chunk.metadata.doc_id,
                document_title=chunk.metadata.document_title,
                section_id=chunk.metadata.section_id,
                section_title=chunk.metadata.section_title,
                content=chunk.content,
                dense_score=None,
                dense_rank=None,
                bm25_score=round(score, 4),
                bm25_rank=b_rank,
                rrf_score=None,
                final_rank=rank,
                metadata=chunk.metadata.model_dump()
            ))
        return results

    def _retrieve_hybrid(
        self,
        query: str,
        top_k: int,
        filters: Optional[Dict[str, Any]],
        dense_weight: float,
        sparse_weight: float
    ) -> List[SearchResult]:
        """
        Executes Reciprocal Rank Fusion over candidate pools from Dense and BM25 retrievers.
        """
        # Fetch wider candidate pools (e.g. 2x top_k or at least 20 candidates) for robust fusion
        candidate_k = max(top_k * 3, 20)
        
        dense_hits = self.vector_store.search(query, top_k=candidate_k, filters=filters)
        sparse_hits = self.bm25_engine.search(query, top_k=candidate_k, filters=filters)

        # Lookup maps: chunk_id -> (score, rank)
        dense_map: Dict[str, Tuple[float, int]] = {c.chunk_id: (score, rank) for c, score, rank in dense_hits}
        sparse_map: Dict[str, Tuple[float, int]] = {c.chunk_id: (score, rank) for c, score, rank in sparse_hits}

        # Union of all candidate chunk IDs
        all_chunk_ids = set(dense_map.keys()) | set(sparse_map.keys())

        # Compute RRF score for every candidate
        scored_candidates: List[Tuple[str, float]] = []
        for cid in all_chunk_ids:
            rrf_score = 0.0
            if cid in dense_map:
                _, d_rank = dense_map[cid]
                rrf_score += dense_weight * (1.0 / (self.rrf_k + d_rank))
            if cid in sparse_map:
                _, s_rank = sparse_map[cid]
                rrf_score += sparse_weight * (1.0 / (self.rrf_k + s_rank))
            scored_candidates.append((cid, rrf_score))

        # Sort candidates by RRF score descending
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        top_candidates = scored_candidates[:top_k]

        # Build final SearchResult objects
        results: List[SearchResult] = []
        for final_rank, (cid, rrf_score) in enumerate(top_candidates, start=1):
            chunk = self.chunks_by_id[cid]
            d_score, d_rank = dense_map.get(cid, (None, None))
            s_score, s_rank = sparse_map.get(cid, (None, None))
            
            results.append(SearchResult(
                chunk_id=chunk.chunk_id,
                doc_id=chunk.metadata.doc_id,
                document_title=chunk.metadata.document_title,
                section_id=chunk.metadata.section_id,
                section_title=chunk.metadata.section_title,
                content=chunk.content,
                dense_score=round(d_score, 4) if d_score is not None else None,
                dense_rank=d_rank,
                bm25_score=round(s_score, 4) if s_score is not None else None,
                bm25_rank=s_rank,
                rrf_score=round(rrf_score, 6),
                final_rank=final_rank,
                metadata=chunk.metadata.model_dump()
            ))

        return results
