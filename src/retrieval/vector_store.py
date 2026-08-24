"""
NumPy-based in-memory vector store with vectorized cosine similarity and metadata filtering.
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from src.models.document import Chunk
from src.retrieval.embeddings import EmbeddingClient


class NumpyVectorStore:
    """Fast, zero-dependency in-memory vector index powered by NumPy matrix operations."""

    def __init__(self, embedding_client: Optional[EmbeddingClient] = None):
        self.embedding_client = embedding_client or EmbeddingClient()
        self.chunks: List[Chunk] = []
        self.chunk_ids: List[str] = []
        self.chunks_by_id: Dict[str, Chunk] = {}
        self.vectors: Optional[np.ndarray] = None  # Matrix of shape (N, D)

    def build_index(self, chunks: List[Chunk]) -> None:
        """Embeds and indexes all provided chunks."""
        self.chunks = chunks
        self.chunk_ids = [c.chunk_id for c in chunks]
        self.chunks_by_id = {c.chunk_id: c for c in chunks}

        texts = [c.content for c in chunks]
        keys = [f"chunk_{c.chunk_id}" for c in chunks]

        print(f"Building vector index for {len(chunks)} chunks...")
        self.vectors = self.embedding_client.embed_texts(texts, keys=keys)
        print(f"Vector index built. Shape: {self.vectors.shape}")

    def search(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[Chunk, float, int]]:
        """
        Searches the index with a text query.
        Returns list of (Chunk, cosine_similarity, dense_rank).
        """
        if self.vectors is None or len(self.chunks) == 0:
            raise ValueError("Vector store index is empty. Call build_index() first.")

        query_vec = self.embedding_client.embed_query(query)  # Shape (D,)
        return self.search_by_vector(query_vec, top_k=top_k, filters=filters)

    def search_by_vector(
        self,
        query_vec: np.ndarray,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[Chunk, float, int]]:
        """
        Searches by a pre-computed normalized query vector.
        """
        # Vectorized dot product cosine similarity (since vectors are L2-normalized)
        sim_scores = np.dot(self.vectors, query_vec)  # Shape: (N,)

        # Apply metadata filters if provided
        valid_mask = np.ones(len(self.chunks), dtype=bool)
        if filters:
            for idx, c in enumerate(self.chunks):
                m = c.metadata
                for f_key, f_val in filters.items():
                    if hasattr(m, f_key):
                        attr_val = getattr(m, f_key)
                    else:
                        attr_val = m.model_dump().get(f_key)
                        
                    if isinstance(f_val, list):
                        if attr_val not in f_val:
                            valid_mask[idx] = False
                            break
                    elif attr_val != f_val:
                        valid_mask[idx] = False
                        break

        # Filter indices
        candidate_indices = np.where(valid_mask)[0]
        if len(candidate_indices) == 0:
            return []

        candidate_scores = sim_scores[candidate_indices]
        
        # Sort top-k
        num_results = min(top_k, len(candidate_indices))
        top_sub_indices = np.argsort(-candidate_scores)[:num_results]
        
        results: List[Tuple[Chunk, float, int]] = []
        for rank, sub_idx in enumerate(top_sub_indices, start=1):
            real_idx = candidate_indices[sub_idx]
            chunk = self.chunks[real_idx]
            score = float(candidate_scores[sub_idx])
            results.append((chunk, score, rank))

        return results
