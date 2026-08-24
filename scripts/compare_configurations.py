"""
Comparative execution script running Baseline A, Baseline B, Agentic C, and Ablations on key queries.
"""

import sys
import json

# Ensure UTF-8 output encoding on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.config import CHUNKS_JSON_PATH
from src.models.document import Chunk
from src.retrieval.hybrid_retriever import HybridRetriever
from src.baselines.naive_rag import NaiveRAG
from src.baselines.hybrid_rag import HybridRAG
from src.agent.orchestrator import AgentOrchestrator
from src.agent.ablations import (
    AblationNoPlanner,
    AblationDenseOnlyAgentic,
    AblationNoSufficiencyChecker,
    AblationNoRetryLoop,
    AblationNoDecomposition,
)


def run_comparison(query_label: str, query: str, retriever):
    print("\n" + "=" * 90)
    print(f"QUERY: {query_label}")
    print(f"Text:  \"{query}\"")
    print("=" * 90)

    systems = [
        ("Baseline A (Naive Dense RAG)", NaiveRAG(retriever=retriever, top_k=5)),
        ("Baseline B (Hybrid RRF RAG)", HybridRAG(retriever=retriever, top_k=5)),
        ("Ablation 3 (No Sufficiency Checker)", AblationNoSufficiencyChecker(retriever=retriever, top_k=5)),
        ("Ablation 4 (No Retry Loop)", AblationNoRetryLoop(retriever=retriever, top_k=5)),
        ("Agentic System C (Full Agentic)", AgentOrchestrator(retriever=retriever)),
    ]

    for name, sys_instance in systems:
        trace = sys_instance.run(query)
        print(f"\n--- {name} ---")
        print(f"Decision / Status: {trace.final_decision} / {trace.generation.status}")
        print(f"Passes: {len(trace.passes)} | Retries: {trace.retry_count} | Latency: {trace.total_latency_ms:.1f}ms")
        if trace.passes:
            print(f"Retrieved Chunks: {trace.passes[0].retrieved_chunk_ids[:3]}...")
        if trace.generation.citations:
            print(f"Citations ({len(trace.generation.citations)}): {[c.chunk_id for c in trace.generation.citations]}")
        response_preview = trace.generation.response_text.replace('\n', ' ')[:180]
        print(f"Response: {response_preview}...")


def main():
    with open(CHUNKS_JSON_PATH, "r", encoding="utf-8") as f:
        chunks = [Chunk(**item) for item in json.load(f)]

    retriever = HybridRetriever.from_chunks(chunks)

    # 1. Answerable Query
    run_comparison("Answerable Single-Hop", "What are the reflection critique tokens introduced in Self-RAG?", retriever)

    # 2. Unanswerable Adversarial Trap
    run_comparison("Unanswerable / Hallucination Trap", "What is the quantum teleportation fidelity achieved by Agent-Q in 2026?", retriever)


if __name__ == "__main__":
    main()
