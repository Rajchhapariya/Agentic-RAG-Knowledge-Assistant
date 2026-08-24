"""
BM25 sparse keyword search engine for technical and acronym-heavy queries.
"""

import re
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from rank_bm25 import BM25Okapi
from src.config import BM25_K1, BM25_B
from src.models.document import Chunk


class BM25SearchEngine:
    """Sparse keyword retriever based on the BM25Okapi algorithm."""

    def __init__(self, k1: float = BM25_K1, b: float = BM25_B):
        self.k1 = k1
        self.b = b
        self.chunks: List[Chunk] = []
        self.bm25: Optional[BM25Okapi] = None

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenizes text preserving acronyms, technical terms, hyphenated words,
        and numbers while standardizing case.
        """
        # Replace special symbols that might separate tokens, but keep hyphens and underscores
        cleaned = re.sub(r'[^\w\s\-_]', ' ', text.lower())
        tokens = [t for t in cleaned.split() if len(t) > 1]
        return tokens

    def build_index(self, chunks: List[Chunk]) -> None:
        """Tokenizes and indexes all chunks for BM25 search."""
        self.chunks = chunks
        tokenized_corpus = [self.tokenize(c.content) for c in chunks]
        self.bm25 = BM25Okapi(tokenized_corpus, k1=self.k1, b=self.b)
        print(f"BM25 index built for {len(chunks)} chunks (k1={self.k1}, b={self.b}).")

    def search(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[Chunk, float, int]]:
        """
        Searches the BM25 index with a query.
        Returns list of (Chunk, bm25_score, bm25_rank).
        """
        if self.bm25 is None or len(self.chunks) == 0:
            raise ValueError("BM25 index is empty. Call build_index() first.")

        query_tokens = self.tokenize(query)
        if not query_tokens:
            return []

        scores = np.array(self.bm25.get_scores(query_tokens))

        # Apply metadata filters
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

        candidate_indices = np.where(valid_mask)[0]
        if len(candidate_indices) == 0:
            return []

        candidate_scores = scores[candidate_indices]

        # Only return items with score > 0
        positive_indices = np.where(candidate_scores > 0)[0]
        if len(positive_indices) == 0:
            # If no positive matches, still return top sorted up to top_k with 0 score
            sorted_sub_idxs = np.argsort(-candidate_scores)[:top_k]
        else:
            sorted_sub_idxs = np.argsort(-candidate_scores)[:top_k]

        results: List[Tuple[Chunk, float, int]] = []
        for rank, sub_idx in enumerate(sorted_sub_idxs, start=1):
            real_idx = candidate_indices[sub_idx]
            chunk = self.chunks[real_idx]
            score = float(candidate_scores[sub_idx])
            results.append((chunk, score, rank))

        return results
