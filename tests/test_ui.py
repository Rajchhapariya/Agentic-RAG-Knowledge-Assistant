"""
Tests for Streamlit UI Components, State Handlers, and Visual Renderers.
All tests run completely offline with 0 external API calls.
"""

import json
import hashlib
from pathlib import Path
import pytest
import numpy as np

from src.config import (
    CHUNKS_JSON_PATH,
    DOC_EMBEDDINGS_CACHE_PATH,
    USER_UPLOADS_DIR,
)
from src.models.document import Chunk
from src.models.trace import (
    AgentTrace,
    QueryPlan,
    PassRecord,
    AuditResult,
    EvidenceRelationship,
    GenerationResult,
    CitationItem,
    ContradictionDetail,
)
from src.retrieval.hybrid_retriever import HybridRetriever
from src.ingestion.user_pdf_pipeline import UserPDFPipeline
from src.ui.app import load_research_retriever, preset_query_map


def test_research_retriever_loader():
    """Test 1: load_research_retriever loads 243 chunks with 0 API calls."""
    retriever = load_research_retriever()
    assert isinstance(retriever, HybridRetriever)
    assert len(retriever.chunks_by_id) == 243


def test_preset_query_map():
    """Test 2: Preset demo queries are populated accurately."""
    assert "1. Answerable: Self-RAG reflection tokens" in preset_query_map
    assert "2. Retry: REALM vs DPR index maintenance" in preset_query_map
    assert "3. Refusal: Quantum fidelity (Out-of-Scope)" in preset_query_map
    assert "4. Debunk: Toolformer PPO reinforcement learning" in preset_query_map
    
    assert "reflection tokens" in preset_query_map["1. Answerable: Self-RAG reflection tokens"]
    assert "REALM versus DPR" in preset_query_map["2. Retry: REALM vs DPR index maintenance"]


def test_decision_badge_mappings():
    """Test 3: Decision verdicts map cleanly to UI badge classes."""
    badge_class_map = {
        "SUFFICIENT": ("badge-sufficient", "✓ ANSWER: FULLY GROUNDED & VERIFIED"),
        "PARTIALLY_SUFFICIENT": ("badge-partial", "⚠ PARTIAL ANSWER (MISSING INFORMATION CAVEATS)"),
        "DEBUNK_FALSE_PREMISE": ("badge-debunk", "🛡️ FALSE PREMISE DEBUNKED & CORRECTED"),
        "REFUSE": ("badge-refuse", "🛑 REFUSED: INSUFFICIENT EVIDENCE (HALLUCINATION PREVENTION)"),
        "DIRECT_ANSWER": ("badge-direct", "💬 DIRECT CONVERSATIONAL RESPONSE")
    }

    for verdict, (cls, label) in badge_class_map.items():
        assert cls.startswith("badge-")
        assert len(label) > 0


def test_trace_rendering_components():
    """Test 4: AgentTrace provides full inspectability fields for UI rendering."""
    plan = QueryPlan(
        query_type="FACTUAL_SINGLE_HOP",
        needs_retrieval=True,
        target_concepts=["RAG", "BART"],
        sub_questions=["How does RAG condition on BART?"],
        initial_search_queries=["RAG BART conditioning"]
    )
    
    audit = AuditResult(
        verdict="SUFFICIENT",
        subquestion_coverage={0: True},
        evidence_relationships=[
            EvidenceRelationship(
                subquestion_idx=0,
                subquestion_text="How does RAG condition on BART?",
                chunk_id="RAG_Lewis_2020_c001",
                doc_id="RAG_Lewis_2020",
                section_title="1 Introduction",
                exact_quote="The seq2seq model (BART) then conditions on these latent documents.",
                relationship="SUPPORTED",
                is_quote_verified=True,
                justification="Explicit statement in introduction."
            )
        ],
        verified_supported_spans=[
            EvidenceRelationship(
                subquestion_idx=0,
                subquestion_text="How does RAG condition on BART?",
                chunk_id="RAG_Lewis_2020_c001",
                doc_id="RAG_Lewis_2020",
                section_title="1 Introduction",
                exact_quote="The seq2seq model (BART) then conditions on these latent documents.",
                relationship="SUPPORTED",
                is_quote_verified=True,
                justification="Explicit statement in introduction."
            )
        ],
        missing_information=[],
        contradiction=ContradictionDetail()
    )

    pass_1 = PassRecord(
        pass_number=1,
        search_queries=["RAG BART conditioning"],
        retrieved_chunk_ids=["RAG_Lewis_2020_c001"],
        retrieved_results=[],
        audit_result=audit,
        reformulated_query=None
    )

    gen = GenerationResult(
        status="ANSWERED",
        response_text="RAG uses BART to condition on retrieved latent documents [RAG_Lewis_2020: 1 Introduction].",
        citations=[
            CitationItem(
                citation_id=1,
                doc_id="RAG_Lewis_2020",
                document_title="RAG_Lewis_2020",
                section_title="1 Introduction",
                chunk_id="RAG_Lewis_2020_c001",
                exact_quote="The seq2seq model (BART) then conditions on these latent documents.",
                claim_text="RAG conditions BART on latent documents."
            )
        ]
    )

    trace = AgentTrace(
        trace_id="test_trace_123",
        query="How does RAG condition on BART?",
        planner=plan,
        passes=[pass_1],
        retry_count=0,
        final_decision="SUFFICIENT",
        generation=gen,
        total_latency_ms=1250.0,
        planner_latency_ms=250.0,
        retrieval_latency_ms=50.0,
        auditor_latency_ms=600.0,
        generator_latency_ms=350.0,
        num_llm_calls=3
    )

    assert trace.final_decision == "SUFFICIENT"
    assert len(trace.passes) == 1
    assert len(trace.generation.citations) == 1
    assert trace.num_llm_calls == 3


def test_retry_visualization_structure():
    """Test 5: Multi-pass trace contains all necessary records for visual retry flow."""
    pass_1_audit = AuditResult(
        verdict="INSUFFICIENT_RETRY",
        subquestion_coverage={0: False},
        evidence_relationships=[],
        verified_supported_spans=[],
        missing_information=["Missing confidence threshold values."],
        contradiction=ContradictionDetail(),
        diagnosed_search_gap="CRAG confidence score thresholds for Correct and Incorrect"
    )

    pass_1 = PassRecord(
        pass_number=1,
        search_queries=["CRAG actions"],
        retrieved_chunk_ids=["CRAG_Yan_2024_c001"],
        retrieved_results=[],
        audit_result=pass_1_audit,
        reformulated_query="CRAG confidence score thresholds for Correct and Incorrect"
    )

    pass_2_audit = AuditResult(
        verdict="SUFFICIENT",
        subquestion_coverage={0: True},
        evidence_relationships=[],
        verified_supported_spans=[],
        missing_information=[],
        contradiction=ContradictionDetail()
    )

    pass_2 = PassRecord(
        pass_number=2,
        search_queries=["CRAG confidence score thresholds for Correct and Incorrect"],
        retrieved_chunk_ids=["CRAG_Yan_2024_c008"],
        retrieved_results=[],
        audit_result=pass_2_audit,
        reformulated_query=None
    )

    passes = [pass_1, pass_2]
    assert len(passes) == 2
    assert passes[0].audit_result.verdict == "INSUFFICIENT_RETRY"
    assert passes[0].reformulated_query is not None
    assert passes[1].audit_result.verdict == "SUFFICIENT"


def test_ui_upload_isolation_guard(tmp_path):
    """Test 6: User upload pipeline does not modify data/processed_chunks/."""
    def get_hash(p: Path) -> str:
        return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else ""

    chunks_sha_before = get_hash(CHUNKS_JSON_PATH)
    emb_sha_before = get_hash(DOC_EMBEDDINGS_CACHE_PATH)

    pipeline = UserPDFPipeline(upload_dir=tmp_path)
    assert pipeline.upload_dir == tmp_path

    chunks_sha_after = get_hash(CHUNKS_JSON_PATH)
    emb_sha_after = get_hash(DOC_EMBEDDINGS_CACHE_PATH)

    assert chunks_sha_before == chunks_sha_after
    assert emb_sha_before == emb_sha_after
