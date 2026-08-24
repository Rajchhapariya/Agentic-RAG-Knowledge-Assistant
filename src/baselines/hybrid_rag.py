"""
Baseline B: Hybrid RAG.
Pipeline: user query -> dense + BM25 hybrid retrieval (RRF, top_k=5) -> single-pass generation.
No planner, no sufficiency checker, no retry loop, no refusal gate.
"""

import time
from typing import List, Optional
from src.retrieval.hybrid_retriever import HybridRetriever
from src.models.retrieval import SearchResult
from src.models.trace import AgentTrace
from src.baselines.common import BaselineGenerator, build_baseline_trace


class HybridRAG:
    """Baseline B: Single-Pass Hybrid RAG (Dense + BM25 via RRF)."""

    def __init__(self, retriever: HybridRetriever, top_k: int = 5):
        self.retriever = retriever
        self.top_k = top_k
        self.generator = BaselineGenerator()

    def run(self, query: str) -> AgentTrace:
        """Executes single-pass hybrid retrieval followed by generation."""
        start_time = time.perf_counter()

        # Hybrid retrieval (dense + BM25 via Reciprocal Rank Fusion)
        retrieved: List[SearchResult] = self.retriever.retrieve(
            query=query,
            top_k=self.top_k,
            mode="hybrid"
        )

        gen_result = self.generator.generate(query=query, retrieved_chunks=retrieved)
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        return build_baseline_trace(
            system_name="BaselineB_HybridRAG",
            query=query,
            retrieved_chunks=retrieved,
            gen_result=gen_result,
            latency_ms=elapsed_ms
        )
