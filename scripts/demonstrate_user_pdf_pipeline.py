"""
Phase 6A Demonstration Script: Validates end-to-end User PDF Ingestion,
Persistent Caching, Cache-Hit Zero API calls, and Agentic RAG Answering/Refusal.
"""

import io
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from reportlab.pdfgen import canvas
from src.config import USER_UPLOADS_DIR
from src.ingestion.user_pdf_pipeline import UserPDFPipeline
from src.agent.orchestrator import AgentOrchestrator
from src.utils.cost_tracker import get_cost_tracker, reset_cost_tracker


def create_demo_pdf() -> bytes:
    """Creates a realistic 3-page research paper PDF."""
    buf = io.BytesIO()
    c = canvas.Canvas(buf)
    
    # Page 1: Abstract & Introduction
    c.setFont("Helvetica-Bold", 14)
    c.drawString(72, 750, "Self-Corrective Knowledge Agents for Long-Context Synthesis")
    c.setFont("Helvetica-Bold", 11)
    c.drawString(72, 710, "1 Abstract & Introduction")
    c.setFont("Helvetica", 10)
    c.drawString(72, 690, "We present Agent-Synthesizer, an autonomous system for verifiable knowledge synthesis.")
    c.drawString(72, 675, "The system employs an iterative evidence auditor operating with a strict 3-pass bounded loop.")
    c.drawString(72, 660, "Key hyperparameters include a target chunk size of 450 tokens with 75 token overlap.")
    c.drawString(72, 50, "Page 1")
    c.showPage()
    
    # Page 2: Methodology
    c.setFont("Helvetica-Bold", 11)
    c.drawString(72, 750, "2 Architecture and Verification Protocol")
    c.setFont("Helvetica", 10)
    c.drawString(72, 730, "The verification protocol enforces conservative mathematical normalization for scientific notation.")
    c.drawString(72, 715, "In our empirical evaluation, the convergence threshold is set to epsilon = 1e-5 (or 10^-5).")
    c.drawString(72, 700, "All claims must be supported by verbatim quotes verified against indexed document passages.")
    c.drawString(72, 50, "Page 2")
    c.showPage()
    
    # Page 3: Empirical Results
    c.setFont("Helvetica-Bold", 11)
    c.drawString(72, 750, "3 Empirical Evaluation")
    c.setFont("Helvetica", 10)
    c.drawString(72, 730, "On the multi-hop synthesis benchmark, Agent-Synthesizer achieves a factual accuracy of 88.4%.")
    c.drawString(72, 715, "Hallucination rate on unanswerable queries is reduced to 0.0% due to strict refusal routing.")
    c.drawString(72, 50, "Page 3")
    c.save()
    return buf.getvalue()


def run_demonstration():
    reset_cost_tracker()
    tracker = get_cost_tracker()
    
    pdf_bytes = create_demo_pdf()
    pdf_name = "Agent_Synthesizer_2026.pdf"
    
    pipeline = UserPDFPipeline(upload_dir=USER_UPLOADS_DIR)
    
    print("=" * 90)
    print("PHASE 6A DEMONSTRATION: USER PDF INGESTION & AGENTIC RAG PIPELINE")
    print("=" * 90)
    
    # -----------------------------------------------------------------------
    # Step 1: Initial Upload (Cache Miss -> Embed & Index)
    # -----------------------------------------------------------------------
    print("\n>>> STEP 1: First Upload of PDF (Cold Ingestion)")
    doc, chunks, retriever, info_1 = pipeline.ingest_pdf(pdf_bytes, filename=pdf_name)
    
    print(f"Extracted Document: \"{doc.title}\" (SHA-256: {info_1['file_hash'][:16]}...)")
    print(f"  - Extracted Pages:     {info_1['num_pages']} pages")
    print(f"  - Extracted Sections:  {len(doc.sections)} sections")
    print(f"  - Section-Aware Chunks:{len(chunks)} chunks")
    print(f"  - Cache Status:        {'CACHE HIT' if info_1['cache_hit'] else 'CACHE MISS (Embeddings Generated)'}")
    print(f"  - Stored Artifacts:    {info_1['doc_dir']}")
    
    # -----------------------------------------------------------------------
    # Step 2: Repeated Upload (Cache Hit -> Zero Doc Embedding Calls)
    # -----------------------------------------------------------------------
    print("\n>>> STEP 2: Repeated Upload of Same PDF (Warm Ingestion)")
    doc_2, chunks_2, retriever_2, info_2 = pipeline.ingest_pdf(pdf_bytes, filename=pdf_name)
    
    print(f"Re-ingested Document: \"{doc_2.title}\"")
    print(f"  - Cache Status:        {'CACHE HIT (0 API Calls)' if info_2['cache_hit'] else 'CACHE MISS'}")
    print(f"  - Reused Chunks:       {len(chunks_2)} chunks")
    print(f"  - Reused Embeddings:   {info_2['dimensions']} dims")
    
    # -----------------------------------------------------------------------
    # Step 3: Answerable Question against Uploaded PDF
    # -----------------------------------------------------------------------
    orchestrator = AgentOrchestrator(retriever=retriever_2)
    
    print("\n>>> STEP 3: Answerable Question via Full Agentic Pipeline")
    q_answerable = "What convergence threshold and factual accuracy are reported for Agent-Synthesizer?"
    print(f"Question: \"{q_answerable}\"")
    
    trace_1 = orchestrator.run(q_answerable)
    print(f"Final Decision:    {trace_1.final_decision} | Status: {trace_1.generation.status}")
    print(f"Total Latency:     {trace_1.total_latency_ms:.0f} ms | LLM Calls: {trace_1.num_llm_calls}")
    print(f"Citations ({len(trace_1.generation.citations)} verified):")
    for cit in trace_1.generation.citations:
        print(f"  * [{cit.document_title} | {cit.section_title}]: \"{cit.exact_quote}\"")
    print(f"Response:\n{trace_1.generation.response_text}")
    
    # -----------------------------------------------------------------------
    # Step 4: Unanswerable Question against Uploaded PDF (Refusal Check)
    # -----------------------------------------------------------------------
    print("\n>>> STEP 4: Unanswerable Question (Hallucination Safeguard Check)")
    q_unanswerable = "What is the quantum superposition error rate measured on cryogenic hardware?"
    print(f"Question: \"{q_unanswerable}\"")
    
    trace_2 = orchestrator.run(q_unanswerable)
    print(f"Final Decision:    {trace_2.final_decision} | Status: {trace_2.generation.status}")
    print(f"Total Latency:     {trace_2.total_latency_ms:.0f} ms | Retries: {trace_2.retry_count}")
    print(f"Citations:         {len(trace_2.generation.citations)} citations")
    print(f"Response:\n{trace_2.generation.response_text}")
    
    print("\n" + "=" * 90)
    print("DEMONSTRATION COST & API USAGE SUMMARY")
    print("=" * 90)
    print(tracker.get_summary_table())


if __name__ == "__main__":
    run_demonstration()
