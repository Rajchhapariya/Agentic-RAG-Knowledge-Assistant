"""
Diagnostic script to inspect Planner, Retrieval, and Auditor behavior on Q02_dev and Q03_dev.
"""

import sys
import json
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.agent.planner import QueryPlanner
from src.retrieval.hybrid_retriever import HybridRetriever
from src.agent.evidence_auditor import EvidenceAuditor
from src.models.document import Chunk
from src.agent.orchestrator import AgentOrchestrator

with open("data/processed_chunks/chunks.json", "r", encoding="utf-8") as f:
    chunks = [Chunk(**c) for c in json.load(f)]
retriever = HybridRetriever.from_chunks(chunks)

print("=" * 80)
print("DIAGNOSING Q03_dev")
print("=" * 80)
q3 = "How does CRAG use upper and lower confidence thresholds to trigger Correct, Incorrect, and Ambiguous actions?"
planner = QueryPlanner()
plan3 = planner.plan(q3)
print("Plan 3 suggested_filters:", plan3.suggested_filters)
print("Plan 3 search queries:", plan3.initial_search_queries)
for sq in plan3.initial_search_queries:
    hits = retriever.retrieve(sq, top_k=5, filters=plan3.suggested_filters)
    print(f"Query '{sq}' -> Hits ({len(hits)}): {[h.chunk_id for h in hits]}")

print("\n" + "=" * 80)
print("DIAGNOSING Q02_dev")
print("=" * 80)
q2 = "What loss function and negative sampling strategy is used to train DPR dual encoders?"
plan2 = planner.plan(q2)
print("Plan 2 subquestions:", plan2.sub_questions)
print("Plan 2 search queries:", plan2.initial_search_queries)
for sq in plan2.initial_search_queries:
    hits = retriever.retrieve(sq, top_k=5, filters=plan2.suggested_filters)
    print(f"Query '{sq}' -> Hits ({len(hits)}): {[h.chunk_id for h in hits]}")

orch = AgentOrchestrator(retriever=retriever)
trace2 = orch.run(q2)
print("Trace 2 Decision:", trace2.final_decision)
for p in trace2.passes:
    print(f"Pass {p.pass_number}: queries={p.search_queries}")
    print(f"  Retrieved: {p.retrieved_chunk_ids}")
    print(f"  Audit verdict: {p.audit_result.verdict}")
    print(f"  Subquestion coverage: {p.audit_result.subquestion_coverage}")
    print(f"  Diagnosed gap: {p.audit_result.diagnosed_search_gap}")
