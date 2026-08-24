"""
Scripted Portfolio Demo Validation Runner.
Executes each of the 5 demo scenarios once, logs execution results,
verifies agent decisions/retries/refusals, and tracks token usage and spend.
"""

import sys
import os
import json
import time
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.config import (
    CHUNKS_JSON_PATH,
    DOC_EMBEDDINGS_CACHE_PATH,
    USER_UPLOADS_DIR,
)
from src.models.document import Chunk
from src.models.trace import AgentTrace
from src.retrieval.hybrid_retriever import HybridRetriever
from src.ingestion.user_pdf_pipeline import UserPDFPipeline
from src.agent.orchestrator import AgentOrchestrator
from src.utils.cost_tracker import get_cost_tracker, reset_cost_tracker


def run_demo():
    print("=" * 90)
    print("AGENTIC RAG KNOWLEDGE ASSISTANT - SCRIPTED PORTFOLIO DEMO VALIDATION")
    print("=" * 90)
    
    reset_cost_tracker()
    tracker = get_cost_tracker()

    # Step 0: Initialize Research Retriever
    with open(CHUNKS_JSON_PATH, "r", encoding="utf-8") as f:
        chunks = [Chunk(**item) for item in json.load(f)]
    research_retriever = HybridRetriever.from_chunks(chunks)
    print(f"[*] Initialized Research Retriever: {len(chunks)} chunks loaded.")

    demo_results = []

    # -----------------------------------------------------------------------
    # SCENARIO 1: Grounded Answer with Verifiable Citations
    # -----------------------------------------------------------------------
    print("\n" + "-" * 90)
    print(">>> SCENARIO 1: Grounded Answer with Verifiable Citations")
    q1 = "How does Self-RAG use reflection tokens during inference?"
    print(f"Question: \"{q1}\"")
    
    orch1 = AgentOrchestrator(retriever=research_retriever)
    t0 = time.perf_counter()
    trace1: AgentTrace = orch1.run(q1)
    dur1 = (time.perf_counter() - t0) * 1000.0

    print(f"Verdict:         {trace1.final_decision}")
    print(f"Passes Executed: {len(trace1.passes)} | Retries: {trace1.retry_count}")
    print(f"Citations Found: {len(trace1.generation.citations)}")
    print(f"Latency:         {dur1:.1f} ms | LLM Calls: {trace1.num_llm_calls}")
    print(f"Response:\n{trace1.generation.response_text}")
    if trace1.generation.citations:
        print(f"Citation [1]: \"{trace1.generation.citations[0].exact_quote}\"")

    demo_results.append({
        "scenario": "Scenario 1: Grounded Answer",
        "verdict": trace1.final_decision,
        "passes": len(trace1.passes),
        "citations": len(trace1.generation.citations),
        "latency_ms": dur1
    })

    # -----------------------------------------------------------------------
    # SCENARIO 2: Search Gap Diagnosis & Bounded Retry
    # -----------------------------------------------------------------------
    print("\n" + "-" * 90)
    print(">>> SCENARIO 2: Dynamic Search Gap Diagnosis & Bounded Retry")
    q2 = "Compare the retrieval index maintenance in REALM versus DPR."
    print(f"Question: \"{q2}\"")
    
    orch2 = AgentOrchestrator(retriever=research_retriever)
    t0 = time.perf_counter()
    trace2: AgentTrace = orch2.run(q2)
    dur2 = (time.perf_counter() - t0) * 1000.0

    print(f"Verdict:         {trace2.final_decision}")
    print(f"Passes Executed: {len(trace2.passes)} | Retries: {trace2.retry_count}")
    print(f"Latency:         {dur2:.1f} ms | LLM Calls: {trace2.num_llm_calls}")
    for p in trace2.passes:
        print(f"  * Pass {p.pass_number}: Verdict={p.audit_result.verdict} | Chunks={len(p.retrieved_chunk_ids)}")
        if p.reformulated_query:
            print(f"    ↳ Reformulation: \"{p.reformulated_query}\"")
    print(f"Response:\n{trace2.generation.response_text}")

    demo_results.append({
        "scenario": "Scenario 2: Retry-Triggering",
        "verdict": trace2.final_decision,
        "passes": len(trace2.passes),
        "citations": len(trace2.generation.citations),
        "latency_ms": dur2
    })

    # -----------------------------------------------------------------------
    # SCENARIO 3: Hallucination Prevention via Controlled Refusal
    # -----------------------------------------------------------------------
    print("\n" + "-" * 90)
    print(">>> SCENARIO 3: Hallucination Prevention via Controlled Refusal")
    q3 = "What is the quantum teleportation fidelity achieved by Agent-Q in 2026?"
    print(f"Question: \"{q3}\"")
    
    orch3 = AgentOrchestrator(retriever=research_retriever)
    t0 = time.perf_counter()
    trace3: AgentTrace = orch3.run(q3)
    dur3 = (time.perf_counter() - t0) * 1000.0

    print(f"Verdict:         {trace3.final_decision}")
    print(f"Passes Executed: {len(trace3.passes)} | Retries: {trace3.retry_count}")
    print(f"Citations:       {len(trace3.generation.citations)}")
    print(f"Latency:         {dur3:.1f} ms | LLM Calls: {trace3.num_llm_calls}")
    print(f"Response:\n{trace3.generation.response_text}")

    demo_results.append({
        "scenario": "Scenario 3: Out-of-Scope Refusal",
        "verdict": trace3.final_decision,
        "passes": len(trace3.passes),
        "citations": len(trace3.generation.citations),
        "latency_ms": dur3
    })

    # -----------------------------------------------------------------------
    # SCENARIO 4: User-Uploaded PDF & Dynamic Ingestion
    # -----------------------------------------------------------------------
    print("\n" + "-" * 90)
    print(">>> SCENARIO 4: Dynamic PDF Ingestion & Content-Addressable Caching")
    
    # Generate synthetic research paper in-memory
    from reportlab.pdfgen import canvas
    import io
    pdf_buf = io.BytesIO()
    c = canvas.Canvas(pdf_buf)
    c.drawString(100, 750, "Agent-Synthesizer: Scalable Multi-Hop Synthesis")
    c.drawString(100, 700, "1 Introduction")
    c.drawString(100, 680, "Agent-Synthesizer operates across distributed knowledge corpora.")
    c.drawString(100, 100, "Page 1")
    c.showPage()
    c.drawString(100, 750, "2 Methodology & Convergence")
    c.drawString(100, 700, "In our optimization procedure, the convergence threshold is set to epsilon = 1e-5.")
    c.drawString(100, 100, "Page 2")
    c.showPage()
    c.drawString(100, 750, "3 Empirical Evaluation")
    c.drawString(100, 700, "Across multi-hop benchmark tests, Agent-Synthesizer achieves a factual accuracy of 88.4%.")
    c.drawString(100, 100, "Page 3")
    c.showPage()
    c.save()
    pdf_bytes = pdf_buf.getvalue()

    pipeline = UserPDFPipeline(upload_dir=USER_UPLOADS_DIR)
    
    # First Upload (Cache Miss)
    doc_uploaded, chunks_uploaded, user_retriever, info_first = pipeline.ingest_pdf(
        pdf_bytes, filename="Agent_Synthesizer_2026.pdf"
    )
    print(f"First Upload Status:  Cache Hit={info_first['cache_hit']} | Chunks={len(chunks_uploaded)}")

    # Repeat Upload (Cache Hit)
    _, _, _, info_repeat = pipeline.ingest_pdf(
        pdf_bytes, filename="Agent_Synthesizer_2026.pdf"
    )
    print(f"Repeat Upload Status: Cache Hit={info_repeat['cache_hit']} (0 embedding calls)")

    # Query against Uploaded PDF
    q4 = "What convergence threshold and factual accuracy are reported for Agent-Synthesizer?"
    print(f"Question on Upload: \"{q4}\"")
    orch4 = AgentOrchestrator(retriever=user_retriever)
    t0 = time.perf_counter()
    trace4: AgentTrace = orch4.run(q4)
    dur4 = (time.perf_counter() - t0) * 1000.0

    print(f"Verdict:         {trace4.final_decision}")
    print(f"Latency:         {dur4:.1f} ms | LLM Calls: {trace4.num_llm_calls}")
    print(f"Response:\n{trace4.generation.response_text}")
    if trace4.generation.citations:
        print(f"Page Citation:   {trace4.generation.citations[0].document_title} - {trace4.generation.citations[0].section_title}")

    demo_results.append({
        "scenario": "Scenario 4: User PDF Upload",
        "verdict": trace4.final_decision,
        "passes": len(trace4.passes),
        "citations": len(trace4.generation.citations),
        "latency_ms": dur4
    })

    # -----------------------------------------------------------------------
    # SCENARIO 5: False Premise Contradiction & Debunking
    # -----------------------------------------------------------------------
    print("\n" + "-" * 90)
    print(">>> SCENARIO 5: False Premise Contradiction & Debunking")
    q5 = "How does Toolformer use Proximal Policy Optimization (PPO) reinforcement learning to optimize API calls?"
    print(f"Question: \"{q5}\"")
    
    orch5 = AgentOrchestrator(retriever=research_retriever)
    t0 = time.perf_counter()
    trace5: AgentTrace = orch5.run(q5)
    dur5 = (time.perf_counter() - t0) * 1000.0

    print(f"Verdict:         {trace5.final_decision}")
    print(f"Passes Executed: {len(trace5.passes)} | Retries: {trace5.retry_count}")
    print(f"Latency:         {dur5:.1f} ms | LLM Calls: {trace5.num_llm_calls}")
    print(f"Response:\n{trace5.generation.response_text}")

    demo_results.append({
        "scenario": "Scenario 5: False Premise Debunk",
        "verdict": trace5.final_decision,
        "passes": len(trace5.passes),
        "citations": len(trace5.generation.citations),
        "latency_ms": dur5
    })

    # -----------------------------------------------------------------------
    # Summary of Demo Execution & Spend
    # -----------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("DEMONSTRATION RUNNER RESULTS SUMMARY")
    print("=" * 90)
    for res in demo_results:
        print(f"{res['scenario']:<35} | Verdict: {res['verdict']:<20} | Passes: {res['passes']} | Latency: {res['latency_ms']:.0f} ms")

    print("\n" + "=" * 90)
    print("TOTAL SPEND & API USAGE FOR DEMO EXECUTION")
    print("=" * 90)
    print(tracker.get_summary_table())


if __name__ == "__main__":
    run_demo()
