"""
Verification script for Phase 2: Demonstrating Dense, BM25, and Hybrid RRF retrieval
across 5 representative query archetypes.
"""

import json
import time
import sys

# Ensure UTF-8 output encoding on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.config import CHUNKS_JSON_PATH
from src.models.document import Chunk
from src.retrieval.hybrid_retriever import HybridRetriever


def main():
    print("Loading chunks from index...")
    with open(CHUNKS_JSON_PATH, "r", encoding="utf-8") as f:
        chunks = [Chunk(**item) for item in json.load(f)]

    retriever = HybridRetriever.from_chunks(chunks)
    print(f"Retriever initialized over {len(chunks)} chunks.")

    test_queries = [
        {
            "category": "1. Semantic Conceptual Query",
            "query": "How does hierarchical context and external memory paging work in language agents?",
            "mode": "hybrid",
            "filters": None
        },
        {
            "category": "2. Exact Acronym / Technical Notation Query",
            "query": "[IsREL] [IsSUP] [Retrieve] [IsUSE] reflection critique tokens",
            "mode": "hybrid",
            "filters": None
        },
        {
            "category": "3. Numerical / Fine-Grained Parameter Query",
            "query": "CRAG confidence score threshold upper lower bound evaluation",
            "mode": "hybrid",
            "filters": None
        },
        {
            "category": "4. Cross-Paper Distractor Query",
            "query": "Active forward-looking retrieval versus static single-time passage retrieval",
            "mode": "hybrid",
            "filters": None
        },
        {
            "category": "5. Metadata-Filtered Query",
            "query": "How does the neural retriever training objective work?",
            "mode": "hybrid",
            "filters": {"doc_id": "REALM_Guu_2020"}
        }
    ]

    for item in test_queries:
        print(f"\n================================================================================")
        print(f"CATEGORY: {item['category']}")
        print(f"QUERY:    '{item['query']}'")
        print(f"MODE:     {item['mode']} | FILTERS: {item['filters']}")
        print(f"--------------------------------------------------------------------------------")

        start = time.perf_counter()
        results = retriever.retrieve(
            query=item["query"],
            top_k=2,
            mode=item["mode"],
            filters=item["filters"]
        )
        latency_ms = (time.perf_counter() - start) * 1000.0

        for r in results:
            print(f"Rank {r.final_rank}: Chunk {r.chunk_id}")
            print(f"  Document: {r.document_title} ({r.doc_id})")
            print(f"  Section:  {r.section_title}")
            print(f"  Scores:   Dense={r.dense_score} (Rank {r.dense_rank}) | BM25={r.bm25_score} (Rank {r.bm25_rank}) | RRF={r.rrf_score}")
            print(f"  Content:  \"{r.content[:180]}...\"\n")
        print(f"Latency:  {latency_ms:.2f} ms")


if __name__ == "__main__":
    main()
