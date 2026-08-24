"""
Demonstration script for Phase 3: Inspectable Agentic RAG Traces across 4 key scenarios.
"""

import sys
import json

# Ensure UTF-8 output encoding on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.config import CHUNKS_JSON_PATH
from src.models.document import Chunk
from src.retrieval.hybrid_retriever import HybridRetriever
from src.agent.orchestrator import AgentOrchestrator


def print_trace(title: str, trace):
    print("\n" + "=" * 85)
    print(f"SCENARIO: {title}")
    print("=" * 85)
    print(f"Trace ID:        {trace.trace_id}")
    print(f"Original Query:  \"{trace.query}\"")
    print(f"Planner Decision: Type={trace.planner.query_type} | Needs Retrieval={trace.planner.needs_retrieval}")
    print(f"Sub-Questions:   {trace.planner.sub_questions}")
    print(f"Initial Searches:{trace.planner.initial_search_queries}")
    print(f"Total Passes:    {len(trace.passes)} (Retries: {trace.retry_count})")
    
    for p in trace.passes:
        print(f"\n  --- Pass {p.pass_number} ---")
        print(f"  Search Queries:      {p.search_queries}")
        print(f"  Retrieved Chunk IDs: {p.retrieved_chunk_ids}")
        print(f"  Audit Verdict:       {p.audit_result.verdict}")
        print(f"  Missing Info:        {p.audit_result.missing_information}")
        if p.audit_result.contradiction.has_conflict:
            print(f"  Conflict Detected:   {p.audit_result.contradiction.conflict_summary}")
        if p.reformulated_query:
            print(f"  Reformulated Query:  \"{p.reformulated_query}\"")
        if p.audit_result.verified_supported_spans:
            print(f"  Verified Spans ({len(p.audit_result.verified_supported_spans)}):")
            for s in p.audit_result.verified_supported_spans:
                print(f"    * [{s.doc_id}: {s.section_title}] Quote: \"{s.exact_quote[:90]}...\" (Quote Verified: {s.is_quote_verified})")

    print(f"\nFinal Decision:  {trace.final_decision}")
    print(f"Generation Status:{trace.generation.status}")
    print(f"\nFinal Response:\n{trace.generation.response_text}\n")
    if trace.generation.citations:
        print(f"Citations ({len(trace.generation.citations)}):")
        for c in trace.generation.citations:
            print(f"  [{c.citation_id}] {c.doc_id} ({c.section_title}, Chunk {c.chunk_id}): \"{c.exact_quote[:80]}...\"")
    print(f"Latency:         {trace.total_latency_ms:.2f} ms")


def main():
    with open(CHUNKS_JSON_PATH, "r", encoding="utf-8") as f:
        chunks = [Chunk(**item) for item in json.load(f)]

    retriever = HybridRetriever.from_chunks(chunks)
    orchestrator = AgentOrchestrator(retriever=retriever)

    # 1. Simple Answerable
    t1 = orchestrator.run("What are the four reflection critique tokens introduced in Self-RAG?")
    print_trace("1. Answerable Single-Hop Query with Citations", t1)

    # 2. Multi-Hop Synthesis
    t2 = orchestrator.run("Compare how Self-RAG decides when to retrieve versus how FLARE triggers active retrieval.")
    print_trace("2. Multi-Hop Cross-Paper Comparative Query", t2)

    # 3. Unanswerable Hallucination Trap
    t3 = orchestrator.run("What is the benchmark accuracy of the Quantum-RAG model on MMLU in 2026?")
    print_trace("3. Genuinely Unanswerable / Hallucination Trap", t3)


if __name__ == "__main__":
    main()
