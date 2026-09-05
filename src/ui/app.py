"""
Agentic RAG Knowledge Assistant - Streamlit Application Interface.
Exposes the multi-stage Agentic RAG pipeline (Planning -> Hybrid Retrieval ->
Evidence Auditing -> Bounded Retry -> Grounded Generation / Refusal)
with full inspectability for both the Research Corpus and dynamic User PDF uploads.
"""

import sys
import os
import re
import json
import time
import hashlib
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

import streamlit as st
import numpy as np

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    CHUNKS_JSON_PATH,
    OPENAI_MODEL,
    OPENAI_EMBEDDING_MODEL,
    EMBEDDING_DIM,
    USER_UPLOADS_DIR,
)
from src.models.document import Chunk, Document
from src.models.trace import AgentTrace, PassRecord
from src.retrieval.hybrid_retriever import HybridRetriever
from src.ingestion.user_pdf_pipeline import UserPDFPipeline
from src.ingestion.pdf_loader import UnextractablePDFError
from src.agent.orchestrator import AgentOrchestrator
from src.utils.cost_tracker import get_cost_tracker, reset_cost_tracker, OpenAIAPIDisabledError


# ---------------------------------------------------------------------------
# Preset Query Mappings
# ---------------------------------------------------------------------------

preset_query_map: Dict[str, str] = {
    "1. Answerable: Self-RAG reflection tokens": "How does Self-RAG use reflection tokens during inference?",
    "2. Retry: REALM vs DPR index maintenance": "Compare the retrieval index maintenance in REALM versus DPR.",
    "3. Refusal: Quantum fidelity (Out-of-Scope)": "What is the quantum teleportation fidelity achieved by Agent-Q in 2026?",
    "4. Debunk: Toolformer PPO reinforcement learning": "How does Toolformer use Proximal Policy Optimization (PPO) reinforcement learning to optimize API calls?"
}


# ---------------------------------------------------------------------------
# Cached Resources
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading Research Corpus (10 papers, 243 chunks)...")
def load_research_retriever() -> HybridRetriever:
    """Loads the frozen 10-paper research corpus into an in-memory HybridRetriever."""
    if not CHUNKS_JSON_PATH.exists():
        st.error(f"Research corpus artifacts not found in {CHUNKS_JSON_PATH.parent}. Run ingestion first.")
        st.stop()

    with open(CHUNKS_JSON_PATH, "r", encoding="utf-8") as f:
        chunks = [Chunk(**item) for item in json.load(f)]

    return HybridRetriever.from_chunks(chunks)


def clean_display_text(text: str, doc_title: Optional[str] = None) -> str:
    """Replaces internal user hashes with clean human-readable titles in answer text."""
    if doc_title:
        text = re.sub(r"user_[0-9a-f]{12}", doc_title, text)
    return text


# ---------------------------------------------------------------------------
# Main Streamlit Application Renderer
# ---------------------------------------------------------------------------

def run_app():
    # Streamlit Page Setup & Custom CSS
    st.set_page_config(
        page_title="Agentic RAG Knowledge Assistant",
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown("""
    <style>
        /* Metric & Badge Styling */
        .status-badge {
            display: inline-block;
            padding: 6px 14px;
            border-radius: 4px;
            font-size: 0.88rem;
            font-weight: 600;
            margin-bottom: 12px;
            letter-spacing: 0.02em;
        }
        .badge-sufficient { background-color: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
        .badge-partial { background-color: #fff3cd; color: #856404; border: 1px solid #ffeeba; }
        .badge-debunk { background-color: #e2d9f3; color: #381e72; border: 1px solid #d0c4e8; }
        .badge-refuse { background-color: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
        .badge-direct { background-color: #e2e3e5; color: #383d41; border: 1px solid #d6d8db; }

        .cache-badge-hit { 
            background-color: #d1ecf1; 
            color: #0c5460; 
            font-weight: 600; 
            padding: 4px 10px; 
            border-radius: 4px; 
            border: 1px solid #bee5eb;
            display: inline-block;
            margin-bottom: 8px;
        }
        .cache-badge-indexed { 
            background-color: #e2f0d9; 
            color: #2b542c; 
            font-weight: 600; 
            padding: 4px 10px; 
            border-radius: 4px; 
            border: 1px solid #cce2c2;
            display: inline-block;
            margin-bottom: 8px;
        }

        /* High-Contrast Readable Quote Box */
        .quote-box {
            background-color: rgba(3, 102, 214, 0.07);
            color: #1a202c;
            border-left: 4px solid #0366d6;
            padding: 10px 14px;
            font-style: normal;
            font-size: 0.95rem;
            line-height: 1.5;
            margin: 8px 0 12px 0;
            border-radius: 0 4px 4px 0;
        }

        /* Source Provenance Label */
        .provenance-tag {
            color: #4a5568;
            font-size: 0.85rem;
            margin-bottom: 4px;
        }
    </style>
    """, unsafe_allow_html=True)

    # -----------------------------------------------------------------------
    # Sidebar: Corpus Mode & Configuration
    # -----------------------------------------------------------------------
    st.sidebar.title("🔬 Agentic RAG")
    st.sidebar.markdown("**Evidence-Aware Self-Corrective Knowledge Assistant**")

    corpus_mode = st.sidebar.radio(
        "Corpus Mode",
        options=["Research Corpus (10 Papers)", "Upload PDF"],
        index=0,
        help="Switch between the frozen 10-paper research corpus and dynamic user PDF uploads."
    )

    active_retriever: Optional[HybridRetriever] = None
    active_corpus_name: str = ""
    active_doc_title: Optional[str] = None

    if corpus_mode == "Research Corpus (10 Papers)":
        active_retriever = load_research_retriever()
        active_corpus_name = "Research Corpus (10 Papers)"
        
        st.sidebar.success("✓ Active: Frozen Research Corpus")
        st.sidebar.markdown("""
        - **Papers**: 10 Foundation Papers (RAG, Self-RAG, CRAG, FLARE, ReAct, Toolformer, MemGPT, Reflexion, DPR, REALM)
        - **Chunks**: 243 section-aware passages
        - **Search**: Hybrid RRF (Dense + BM25)
        """)
        
        # Preset Demo Questions
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 🎯 Quick Demo Presets")
        demo_preset = st.sidebar.selectbox(
            "Select a demo query:",
            options=[
                "Custom Query",
                "1. Answerable: Self-RAG reflection tokens",
                "2. Retry: REALM vs DPR index maintenance",
                "3. Refusal: Quantum fidelity (Out-of-Scope)",
                "4. Debunk: Toolformer PPO reinforcement learning"
            ]
        )

    else:
        active_corpus_name = "Uploaded PDF Document"
        st.sidebar.markdown("### 📄 User PDF Ingestion")
        
        uploaded_file = st.sidebar.file_uploader(
            "Upload a research paper or PDF document",
            type=["pdf"],
            help="Upload any PDF to parse, chunk, embed, and query through the Agentic RAG pipeline."
        )
        
        demo_preset = "Custom Query"

        if uploaded_file is not None:
            file_bytes = uploaded_file.getvalue()
            file_hash = hashlib.sha256(file_bytes).hexdigest()
            active_doc_title = uploaded_file.name
            
            # Check if already processed in session state
            if st.session_state.get("uploaded_pdf_hash") != file_hash:
                try:
                    with st.sidebar.status("Processing uploaded PDF...", expanded=True) as status:
                        st.write("1. Extracting text & sections...")
                        pipeline = UserPDFPipeline(upload_dir=USER_UPLOADS_DIR)
                        doc, chunks, retriever, info = pipeline.ingest_pdf(
                            file_bytes,
                            filename=uploaded_file.name
                        )
                        
                        st.session_state["uploaded_pdf_hash"] = file_hash
                        st.session_state["uploaded_doc"] = doc
                        st.session_state["uploaded_chunks"] = chunks
                        st.session_state["uploaded_retriever"] = retriever
                        st.session_state["uploaded_info"] = info
                        status.update(label="PDF Ingestion Complete", state="complete", expanded=False)
                except UnextractablePDFError as e:
                    st.sidebar.error(f"❌ Extraction Failed: {e}")
                except Exception as e:
                    err_str = str(e)
                    if "401" in err_str or "invalid_api_key" in err_str:
                        st.sidebar.error("🔑 **Invalid OpenAI API Key (401)**: The key provided is invalid or a placeholder. In Streamlit Cloud, go to **Manage app (lower right) → Settings (⋮) → Secrets** and set `OPENAI_API_KEY` to your valid key from https://platform.openai.com/api-keys.")
                    else:
                        st.sidebar.error(f"❌ Ingestion Error: {e}")
            
            # Display Upload Metadata from session state
            if "uploaded_info" in st.session_state:
                info = st.session_state["uploaded_info"]
                st.sidebar.markdown("---")
                if info.get("cache_hit"):
                    st.sidebar.markdown('<span class="cache-badge-hit">✓ Cache Hit — Embeddings Reused</span>', unsafe_allow_html=True)
                else:
                    st.sidebar.markdown('<span class="cache-badge-indexed">✓ PDF Indexed — Embeddings Generated</span>', unsafe_allow_html=True)

                st.sidebar.markdown(f"**Document**: `{info['filename']}`")
                st.sidebar.markdown(f"**Pages**: `{info['num_pages']}` | **Chunks**: `{info['num_chunks']}`")
                active_retriever = st.session_state.get("uploaded_retriever")
                active_doc_title = info['filename']
        else:
            st.sidebar.info("Please upload a PDF file to begin querying.")
            st.session_state.pop("uploaded_pdf_hash", None)
            st.session_state.pop("uploaded_doc", None)
            st.session_state.pop("uploaded_chunks", None)
            st.session_state.pop("uploaded_retriever", None)
            st.session_state.pop("uploaded_info", None)

    # System details footer
    st.sidebar.markdown("---")
    st.sidebar.caption(f"**Generator Model**: `{OPENAI_MODEL}`\n\n**Embedding**: `{OPENAI_EMBEDDING_MODEL}` ({EMBEDDING_DIM}d)\n\n**Auditing Max Retries**: `2` (3 passes max)")

    # -----------------------------------------------------------------------
    # Main Query Area
    # -----------------------------------------------------------------------
    st.title("Agentic RAG Knowledge Assistant")
    st.caption("Plan → Retrieve → Audit → Retry → Grounded Answer")

    default_query = preset_query_map.get(demo_preset, "")

    with st.form("query_form", clear_on_submit=False):
        col_input, col_btn = st.columns([5, 1])
        with col_input:
            user_query = st.text_input(
                "Enter research question:",
                value=default_query,
                placeholder="Ask a factual question across the active corpus...",
                label_visibility="collapsed"
            )
        with col_btn:
            submit_button = st.form_submit_button("Ask Assistant", use_container_width=True)

    # -----------------------------------------------------------------------
    # Execution & Trace Visualization
    # -----------------------------------------------------------------------
    if submit_button and user_query.strip():
        if active_retriever is None:
            st.warning("⚠️ No active retriever available. Please select a valid corpus or upload a PDF.")
        else:
            reset_cost_tracker()
            orchestrator = AgentOrchestrator(retriever=active_retriever)
            
            with st.spinner("Executing Agentic Planning, Retrieval, and Evidence Auditing..."):
                try:
                    trace: AgentTrace = orchestrator.run(user_query.strip())
                    st.session_state["last_trace"] = trace
                except OpenAIAPIDisabledError as e:
                    st.error(f"🔒 API Execution Guard: {e}")
                    st.stop()
                except Exception as e:
                    err_str = str(e)
                    if "401" in err_str or "invalid_api_key" in err_str:
                        st.error("🔑 **Invalid OpenAI API Key (401)**: The key provided is invalid or a placeholder. In Streamlit Cloud, go to **Manage app (lower right) → Settings (⋮) → Secrets** and set `OPENAI_API_KEY` to your valid key from https://platform.openai.com/api-keys.")
                    else:
                        st.error(f"❌ Execution Error: {e}")
                    st.stop()

    # Render Last Trace if available
    if "last_trace" in st.session_state:
        trace: AgentTrace = st.session_state["last_trace"]
        
        st.markdown("---")
        
        # 1. Decision Status Badge
        badge_class_map = {
            "SUFFICIENT": ("badge-sufficient", "✓ ANSWER: FULLY GROUNDED & VERIFIED"),
            "PARTIALLY_SUFFICIENT": ("badge-partial", "⚠ PARTIAL ANSWER (MISSING INFORMATION CAVEATS)"),
            "DEBUNK_FALSE_PREMISE": ("badge-debunk", "🛡️ FALSE PREMISE DEBUNKED & CORRECTED"),
            "REFUSE": ("badge-refuse", "🛑 REFUSED: INSUFFICIENT EVIDENCE (HALLUCINATION PREVENTION)"),
            "DIRECT_ANSWER": ("badge-direct", "💬 DIRECT CONVERSATIONAL RESPONSE")
        }
        badge_cls, badge_text = badge_class_map.get(trace.final_decision, ("badge-direct", trace.final_decision))
        
        st.markdown(f'<div class="status-badge {badge_cls}">{badge_text}</div>', unsafe_allow_html=True)
        
        # 2. Main Response Box (Human-readable without internal IDs)
        display_response = clean_display_text(trace.generation.response_text, active_doc_title)
        st.markdown("### Answer")
        st.markdown(display_response)
        
        # 3. Verified Citations
        if trace.generation.citations:
            st.markdown("#### 📚 Verified Source Citations")
            for i, cit in enumerate(trace.generation.citations, start=1):
                # Format human-readable title and location
                clean_title = active_doc_title if (active_doc_title and cit.doc_id.startswith("user_")) else cit.document_title
                source_label = f"{clean_title} — {cit.section_title}" if cit.section_title else f"{clean_title}"
                
                with st.expander(f"📌 Citation [{i}] Source: {source_label}", expanded=False):
                    st.markdown("**Verified Supporting Evidence:**")
                    st.markdown(f'<div class="quote-box">"{cit.exact_quote}"</div>', unsafe_allow_html=True)
                    
                    st.markdown(f"- **Document**: `{clean_title}`")
                    if cit.section_title:
                        st.markdown(f"- **Section / Location**: `{cit.section_title}`")
                    
                    # Technical IDs in small caption area
                    with st.container():
                        st.caption(f"Technical Trace ID: `{cit.doc_id}` | Chunk: `{cit.chunk_id}` | Verified Claim: {cit.claim_text}")
        elif trace.final_decision == "REFUSE":
            st.info("ℹ️ No citations are provided because the indexed corpus lacks verified supporting evidence.")

        # 4. Agentic Retry Flow (Visualized if retries occurred)
        if trace.retry_count > 0:
            st.markdown("#### 🔄 Iterative Self-Correction Trace")
            cols = st.columns(len(trace.passes))
            for p_idx, p_rec in enumerate(trace.passes):
                with cols[p_idx]:
                    st.markdown(f"**Pass {p_rec.pass_number}**")
                    st.markdown(f"- Verdict: `{p_rec.audit_result.verdict}`")
                    st.markdown(f"- Chunks: `{len(p_rec.retrieved_chunk_ids)}`")
                    if p_rec.reformulated_query:
                        st.caption(f"↳ Reformulated: *\"{p_rec.reformulated_query}\"*")
                    elif p_rec.audit_result.verdict == "SUFFICIENT":
                        st.caption("↳ Verified sufficient evidence.")

        # 5. Expandable Full Agent Trace (Deep Inspect)
        with st.expander("🔍 Deep Inspect: Complete Agent Execution Trace", expanded=False):
            tab_plan, tab_retrieval, tab_audit, tab_perf = st.tabs([
                "1. Query Plan",
                "2. Retrieval Passes",
                "3. Evidence Auditor",
                "4. Latency & Spend"
            ])
            
            with tab_plan:
                st.markdown(f"**Query Type**: `{trace.planner.query_type}` | **Retrieval Needed**: `{trace.planner.needs_retrieval}`")
                st.markdown(f"**Target Concepts**: {', '.join([f'`{c}`' for c in trace.planner.target_concepts])}")
                st.markdown("**Sub-Questions Decomposed:**")
                for idx, sq in enumerate(trace.planner.sub_questions, start=1):
                    st.markdown(f"  {idx}. {sq}")
                st.markdown("**Initial Search Queries:**")
                for q in trace.planner.initial_search_queries:
                    st.code(q, language="text")

            with tab_retrieval:
                for p_rec in trace.passes:
                    st.markdown(f"##### Pass {p_rec.pass_number}")
                    st.markdown(f"**Executed Queries:** {', '.join([f'`{q}`' for q in p_rec.search_queries])}")
                    st.markdown(f"**Candidate Chunks Retrieved ({len(p_rec.retrieved_results)}):**")
                    for r in p_rec.retrieved_results:
                        score_val = r.rrf_score if r.rrf_score is not None else (r.dense_score if r.dense_score is not None else 0.0)
                        st.markdown(f"- `{r.chunk_id}` [{r.doc_id} | {r.section_title}] - Score: `{score_val:.4f}` (Dense Rank: {r.dense_rank}, BM25 Rank: {r.bm25_rank})")

            with tab_audit:
                last_pass = trace.passes[-1]
                last_audit = last_pass.audit_result
                st.markdown(f"**Final Audit Verdict**: `{last_audit.verdict}`")
                st.markdown(f"**Verified Quotes Count**: `{len(last_audit.verified_supported_spans)}`")
                
                if last_audit.contradiction.has_conflict:
                    st.warning(f"Contradiction / False Premise Detected: {last_audit.contradiction.conflict_summary}")
                
                if last_audit.missing_information:
                    st.markdown("**Missing Information Diagnosed:**")
                    for m in last_audit.missing_information:
                        st.markdown(f"- {m}")
                
                st.markdown("**Evidence Relationships:**")
                for rel in last_audit.evidence_relationships:
                    status_icon = "✓" if rel.is_quote_verified and rel.relationship == "SUPPORTED" else ("✗" if rel.relationship == "CONTRADICTED" else "○")
                    st.markdown(f"- {status_icon} **Sub-Q {rel.subquestion_idx + 1}** [{rel.chunk_id}]: `{rel.relationship}` (Quote Verified: `{rel.is_quote_verified}`)")
                    if rel.exact_quote:
                        st.caption(f"  Quote: \"{rel.exact_quote}\"")

            with tab_perf:
                col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                col_m1.metric("Total Latency", f"{trace.total_latency_ms:.0f} ms")
                col_m2.metric("Auditor Latency", f"{trace.auditor_latency_ms:.0f} ms")
                col_m3.metric("LLM API Calls", trace.num_llm_calls)
                col_m4.metric("Retries Executed", trace.retry_count)
                
                st.markdown("##### Latency Breakdown")
                perf_data = {
                    "Planning": f"{trace.planner_latency_ms:.1f} ms",
                    "Retrieval": f"{trace.retrieval_latency_ms:.1f} ms",
                    "Evidence Auditing": f"{trace.auditor_latency_ms:.1f} ms",
                    "Generation / Refusal": f"{trace.generator_latency_ms:.1f} ms",
                    "Total Execution": f"{trace.total_latency_ms:.1f} ms"
                }
                st.json(perf_data)

                # Cost Summary Table
                cost_tracker = get_cost_tracker()
                st.markdown("##### API Token & Spend Instrumentation")
                st.text(cost_tracker.get_summary_table())


if __name__ == "__main__":
    run_app()
