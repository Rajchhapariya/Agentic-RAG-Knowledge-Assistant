"""
Targeted Development Smoke Test for Evidence Auditing Refinements.

Executes a small, controlled live API smoke test across 5 representative development cases:
1. Normal answerable question (Q01_dev)
2. False-premise question (Q06_dev)
3. Exact numerical question (Q03_dev)
4. Genuinely unanswerable question (Q05_dev)
5. Multi-hop comparative question (Q04_dev)

Does NOT touch, overwrite, or rerun the held-out test split results.
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import json
from pathlib import Path
from src.config import CHUNKS_JSON_PATH
from src.models.document import Chunk
from src.retrieval.hybrid_retriever import HybridRetriever
from src.agent.orchestrator import AgentOrchestrator
from src.evaluation.benchmark_schema import BenchmarkQuestion
from src.evaluation.metrics import evaluate_question_trace
from src.utils.cost_tracker import get_cost_tracker, reset_cost_tracker


def run_dev_smoke_test():
    reset_cost_tracker()
    
    benchmark_path = Path("data/benchmark_dataset.json")
    with open(benchmark_path, "r", encoding="utf-8") as f:
        all_questions = [BenchmarkQuestion(**item) for item in json.load(f)]
        
    dev_target_ids = ["Q01_dev", "Q03_dev", "Q04_dev", "Q05_dev", "Q06_dev"]
    selected_questions = [q for q in all_questions if q.question_id in dev_target_ids]
    
    with open(CHUNKS_JSON_PATH, "r", encoding="utf-8") as f:
        chunks = [Chunk(**item) for item in json.load(f)]
        
    retriever = HybridRetriever.from_chunks(chunks)
    orchestrator = AgentOrchestrator(retriever=retriever)
    
    print("=" * 90)
    print("TARGETED DEVELOPMENT SMOKE TEST (5 Representative Dev Cases)")
    print("=" * 90)
    
    for i, q in enumerate(selected_questions, start=1):
        print(f"\n--- [{i}/5] Testing {q.question_id} ({q.category}) ---")
        print(f"Question: \"{q.question}\"")
        print(f"Expected Behavior: {q.expected_behavior} (Expected Refusal: {q.expected_refusal})")
        
        trace = orchestrator.run(q.question)
        eval_rec = evaluate_question_trace(question=q, trace=trace, system_name="Agentic_System_C_Full")
        
        status_flag = "✓" if eval_rec.correctness_score == 1.0 or eval_rec.is_true_refusal else ("~" if eval_rec.correctness_score == 0.5 else "✗")
        print(f"Outcome: {status_flag} {eval_rec.correctness_label} (Score: {eval_rec.correctness_score})")
        print(f"Final Decision: {trace.final_decision} | Generation Status: {trace.generation.status}")
        print(f"Retries: {trace.retry_count} | Total Latency: {trace.total_latency_ms:.0f} ms | LLM Calls: {trace.num_llm_calls}")
        print(f"Citations: {len(trace.generation.citations)} citations")
        print(f"Response:\n{trace.generation.response_text}")
        
    print("\n" + "=" * 90)
    print("SMOKE TEST COST & TOKEN SUMMARY")
    print("=" * 90)
    print(get_cost_tracker().get_summary_table())


if __name__ == "__main__":
    run_dev_smoke_test()
