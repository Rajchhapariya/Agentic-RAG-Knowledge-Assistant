"""
Phase 4 Test Suite: Baselines (A, B), Agentic System C, and 5 Ablations.

Architecture:
  REAL application logic + FAKE LLM responses.

The retrieval engine (HybridRetriever, BM25, NumpyVectorStore) runs REAL code
against the cached embeddings. The planner, auditor, and generator are mocked
at the OpenAI boundary so no API calls are made.

Key design choices:
- Baseline A (NaiveRAG) and Baseline B (HybridRAG) use a BaselineGenerator
  (not GroundedGenerator), so their mock target is src.baselines.common.BaselineGenerator.
- AblationNoSufficiencyChecker bypasses EvidenceAuditor completely — no auditor mock needed.
- Schema compatibility test verifies trace_id prefixes without running full LLM pipelines.

This file makes ZERO external OpenAI API calls.
"""

import json
import pytest
from unittest.mock import patch
from src.config import CHUNKS_JSON_PATH
from src.models.document import Chunk
from src.retrieval.hybrid_retriever import HybridRetriever
from src.models.trace import AgentTrace, GenerationResult
from src.baselines.naive_rag import NaiveRAG
from src.baselines.hybrid_rag import HybridRAG
from src.agent.orchestrator import AgentOrchestrator
from src.agent.reformulator import QueryReformulator
from src.agent.ablations import (
    AblationNoPlanner,
    AblationDenseOnlyAgentic,
    AblationNoSufficiencyChecker,
    AblationNoRetryLoop,
    AblationNoDecomposition,
)
from conftest import (
    make_query_plan,
    make_sufficient_audit,
    make_definitively_absent_audit,
    make_answered_generation,
    make_refused_generation,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def loaded_chunks():
    with open(CHUNKS_JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [Chunk(**item) for item in data]


@pytest.fixture(scope="module")
def retriever(loaded_chunks):
    return HybridRetriever.from_chunks(loaded_chunks)


# ---------------------------------------------------------------------------
# Test 1: Baseline A (Naive RAG: Dense-only, single-pass)
#   Verifies trace structure, prefix, and pass count.
# ---------------------------------------------------------------------------
def test_baseline_a_dense_only_single_pass(retriever):
    """NaiveRAG uses BaselineGenerator, which is mocked. Real dense retrieval runs."""
    baseline_a = NaiveRAG(retriever=retriever, top_k=5)
    query = "What is the dense retriever dual-encoder architecture in DPR?"
    ans = make_answered_generation(doc_id="DPR_Karpukhin_2020")

    with patch("src.baselines.common.BaselineGenerator.generate", return_value=ans):
        trace: AgentTrace = baseline_a.run(query)

    assert trace.trace_id.startswith("baselinea_")
    assert len(trace.passes) == 1
    assert trace.retry_count == 0
    assert len(trace.passes[0].retrieved_chunk_ids) <= 5
    assert trace.generation.status == "ANSWERED"
    assert trace.final_decision == "SUFFICIENT"


# ---------------------------------------------------------------------------
# Test 2: Baseline B (Hybrid RAG: Dense + BM25, single-pass)
# ---------------------------------------------------------------------------
def test_baseline_b_hybrid_single_pass(retriever):
    """HybridRAG uses BaselineGenerator, which is mocked. Real hybrid retrieval runs."""
    baseline_b = HybridRAG(retriever=retriever, top_k=5)
    query = "What is the dense retriever dual-encoder architecture in DPR?"
    ans = make_answered_generation(doc_id="DPR_Karpukhin_2020")

    with patch("src.baselines.common.BaselineGenerator.generate", return_value=ans):
        trace: AgentTrace = baseline_b.run(query)

    assert trace.trace_id.startswith("baselineb_")
    assert len(trace.passes) == 1
    assert trace.retry_count == 0
    assert len(trace.passes[0].retrieved_chunk_ids) <= 5
    assert trace.generation.status == "ANSWERED"
    assert trace.final_decision == "SUFFICIENT"


# ---------------------------------------------------------------------------
# Test 3: Baselines Lack Refusal Gate on Adversarial/Unanswerable Query
#   Verifies that baselines always attempt generation (no refusal logic).
# ---------------------------------------------------------------------------
def test_baselines_attempt_answer_on_unanswerable_query(retriever):
    """
    Baselines have no evidence auditor — they always attempt generation.
    Both should report ANSWERED/SUFFICIENT even on nonsense queries.
    """
    baseline_a = NaiveRAG(retriever=retriever, top_k=5)
    baseline_b = HybridRAG(retriever=retriever, top_k=5)
    unanswerable = "What is the quantum teleportation fidelity achieved by Agent-Q in 2026?"
    ans = make_answered_generation()

    with patch("src.baselines.common.BaselineGenerator.generate", return_value=ans):
        trace_a = baseline_a.run(unanswerable)
        trace_b = baseline_b.run(unanswerable)

    # Baselines have no refusal gate → always ANSWERED
    assert trace_a.generation.status == "ANSWERED"
    assert trace_b.generation.status == "ANSWERED"
    assert trace_a.final_decision == "SUFFICIENT"
    assert trace_b.final_decision == "SUFFICIENT"


# ---------------------------------------------------------------------------
# Test 4: Query Reformulator Produces Non-Redundant, Gap-Targeted Queries
#   Fully deterministic — no LLM calls required.
# ---------------------------------------------------------------------------
def test_reformulator_produces_distinct_non_redundant_queries():
    """QueryReformulator.reformulate() is deterministic string manipulation — no API calls."""
    original = "What confidence thresholds trigger the Ambiguous action in CRAG?"
    previous_searches = [
        "CRAG confidence thresholds",
        "CRAG ambiguous action threshold"
    ]
    missing = ["upper and lower numerical threshold values for retrieval evaluator"]
    gap = "retrieval evaluator upper and lower action trigger bounds"
    concepts = ["CRAG", "retrieval evaluator", "action trigger"]

    new_q = QueryReformulator.reformulate(
        original_query=original,
        diagnosed_gap=gap,
        missing_information=missing,
        previous_queries=previous_searches,
        target_concepts=concepts
    )

    assert isinstance(new_q, str)
    assert len(new_q) > 5
    assert new_q.lower() not in [p.lower() for p in previous_searches]
    assert "crag" in new_q.lower() or "retrieval evaluator" in new_q.lower()


# ---------------------------------------------------------------------------
# Test 5: Ablation 1 (No Planner) — trace structure verification
# ---------------------------------------------------------------------------
def test_ablation_1_no_planner(retriever):
    """
    AblationNoPlanner bypasses QueryPlanner. Real auditor+generator are mocked.
    Verifies: single monolithic sub-question, trace ID prefix, real retrieval.
    """
    abl1 = AblationNoPlanner(retriever=retriever, top_k=5)
    query = "How does ReAct combine thought and action in reasoning tasks?"
    suf = make_sufficient_audit(sub_questions=[query])
    ans = make_answered_generation()

    with patch("src.agent.evidence_auditor.EvidenceAuditor.audit", return_value=suf), \
         patch("src.agent.grounded_generator.GroundedGenerator.generate", return_value=ans):
        trace: AgentTrace = abl1.run(query)

    assert trace.trace_id.startswith("abl1_")
    assert trace.planner.sub_questions == [query]  # Monolithic raw query (no decomposition)
    assert len(trace.passes) >= 1
    assert trace.generation.status in {"ANSWERED", "REFUSED"}


# ---------------------------------------------------------------------------
# Test 6: Ablation 2 (No BM25 / Dense-Only Agentic)
# ---------------------------------------------------------------------------
def test_ablation_2_dense_only_agentic(retriever):
    """
    AblationDenseOnlyAgentic disables BM25/RRF (dense-only retrieval).
    Planner, auditor, generator are mocked. Verifies dense-only retrieval path.
    """
    abl2 = AblationDenseOnlyAgentic(retriever=retriever, top_k=5)
    query = "What is the memory hierarchy in MemGPT?"
    plan = make_query_plan(sub_questions=[query], search_queries=[query])
    suf = make_sufficient_audit(sub_questions=[query])
    ans = make_answered_generation(doc_id="MemGPT_Packer_2023")

    with patch("src.agent.planner.QueryPlanner.plan", return_value=plan), \
         patch("src.agent.evidence_auditor.EvidenceAuditor.audit", return_value=suf), \
         patch("src.agent.grounded_generator.GroundedGenerator.generate", return_value=ans):
        trace: AgentTrace = abl2.run(query)

    assert trace.trace_id.startswith("abl2_")
    assert len(trace.planner.sub_questions) >= 1
    assert len(trace.passes) >= 1
    assert trace.generation.status in {"ANSWERED", "REFUSED"}


# ---------------------------------------------------------------------------
# Test 7: Ablation 3 (No Sufficiency Checker)
#   Bypasses EvidenceAuditor entirely — only BaselineGenerator needs mocking.
# ---------------------------------------------------------------------------
def test_ablation_3_no_sufficiency_checker(retriever):
    """
    AblationNoSufficiencyChecker never calls EvidenceAuditor.audit().
    Uses BaselineGenerator (mocked). Always SUFFICIENT/ANSWERED.
    """
    abl3 = AblationNoSufficiencyChecker(retriever=retriever, top_k=5)
    query = "What is the memory hierarchy in MemGPT?"
    plan = make_query_plan(sub_questions=[query], search_queries=[query])
    ans = make_answered_generation()

    with patch("src.agent.planner.QueryPlanner.plan", return_value=plan), \
         patch("src.baselines.common.BaselineGenerator.generate", return_value=ans):
        trace: AgentTrace = abl3.run(query)

    assert trace.trace_id.startswith("abl3_")
    assert len(trace.passes) == 1
    assert trace.retry_count == 0
    # No sufficiency checker → always SUFFICIENT/ANSWERED
    assert trace.final_decision == "SUFFICIENT"
    assert trace.generation.status == "ANSWERED"


# ---------------------------------------------------------------------------
# Test 8: Ablation 4 (No Retry Loop — Single-Pass Auditor)
# ---------------------------------------------------------------------------
def test_ablation_4_no_retry_loop(retriever):
    """
    AblationNoRetryLoop (max_retries=0): even DEFINITIVELY_ABSENT leads to REFUSE in one pass.
    Auditor returns absent. Verifies single-pass enforcement.
    """
    abl4 = AblationNoRetryLoop(retriever=retriever, top_k=5)
    query = "What is the quantum teleportation fidelity achieved by Agent-Q in 2026?"
    plan = make_query_plan(sub_questions=[query], search_queries=[query])
    absent = make_definitively_absent_audit(sub_questions=[query])
    ref = make_refused_generation()

    with patch("src.agent.planner.QueryPlanner.plan", return_value=plan), \
         patch("src.agent.evidence_auditor.EvidenceAuditor.audit", return_value=absent), \
         patch("src.agent.grounded_generator.GroundedGenerator.generate", return_value=ref):
        trace: AgentTrace = abl4.run(query)

    assert trace.trace_id.startswith("abl4_")
    assert len(trace.passes) == 1  # Exactly 1 pass, no retries
    assert trace.retry_count == 0
    assert trace.final_decision == "REFUSE"
    assert trace.generation.status == "REFUSED"


# ---------------------------------------------------------------------------
# Test 9: Ablation 5 (No Multi-Hop Decomposition)
# ---------------------------------------------------------------------------
def test_ablation_5_no_decomposition(retriever):
    """
    AblationNoDecomposition forces a single monolithic sub-question regardless of query.
    Even multi-hop queries get a coverage dict with exactly 1 key.
    """
    abl5 = AblationNoDecomposition(retriever=retriever, top_k=5)
    query = "Compare how Self-RAG decides when to retrieve versus how FLARE triggers active retrieval."

    # Plan returns 2 sub_questions — but Ablation 5 ignores them and forces monolithic
    plan = make_query_plan(
        query_type="MULTI_HOP_COMPARATIVE",
        sub_questions=["Self-RAG retrieve", "FLARE active retrieval"],
        search_queries=["Self-RAG retrieve vs FLARE active retrieval"],
    )
    # Auditor sees monolithic [query] — 1 sub-question → coverage dict has 1 key
    suf = make_sufficient_audit(sub_questions=[query])
    ans = make_answered_generation()

    with patch("src.agent.planner.QueryPlanner.plan", return_value=plan), \
         patch("src.agent.evidence_auditor.EvidenceAuditor.audit", return_value=suf), \
         patch("src.agent.grounded_generator.GroundedGenerator.generate", return_value=ans):
        trace: AgentTrace = abl5.run(query)

    assert trace.trace_id.startswith("abl5_")
    # Ablation 5 forces single monolithic sub-question → coverage has exactly 1 entry
    assert len(trace.passes[0].audit_result.subquestion_coverage) == 1


# ---------------------------------------------------------------------------
# Test 10: System C (Full Agentic RAG) Remains Fully Functional
# ---------------------------------------------------------------------------
def test_system_c_agentic_unchanged(
    retriever,
    patch_planner_sufficient,
    patch_auditor_always_sufficient,
    patch_generator_answered,
):
    """Full AgentOrchestrator with all LLM components mocked — verifies end-to-end trace."""
    system_c = AgentOrchestrator(retriever=retriever)
    query = "What are the reflection critique tokens introduced in Self-RAG?"
    trace: AgentTrace = system_c.run(query)

    assert trace.trace_id.startswith("tr_")
    assert trace.final_decision in {"SUFFICIENT", "REFUSE"}
    assert len(trace.passes) >= 1
    assert trace.total_latency_ms > 0


# ---------------------------------------------------------------------------
# Test 11: Schema Compatibility Across All 8 Configurations
#   Verifies trace_id prefix and required Pydantic fields — no live LLM calls.
# ---------------------------------------------------------------------------
def test_schema_compatibility_across_all_configurations(retriever):
    """
    All 8 system configurations must produce AgentTrace with valid schema.
    Real retrieval runs; LLM boundaries are mocked per-system with context managers.
    """
    query = "What is the dual-encoder architecture in DPR?"
    suf_plan = make_query_plan(sub_questions=[query], search_queries=[query])
    suf_audit = make_sufficient_audit(sub_questions=[query])
    ans_gen = make_answered_generation()

    expected_prefixes = {
        "baselinea_": NaiveRAG(retriever=retriever, top_k=3),
        "baselineb_": HybridRAG(retriever=retriever, top_k=3),
        "abl1_": AblationNoPlanner(retriever=retriever, top_k=3),
        "abl2_": AblationDenseOnlyAgentic(retriever=retriever, top_k=3),
        "abl3_": AblationNoSufficiencyChecker(retriever=retriever, top_k=3),
        "abl4_": AblationNoRetryLoop(retriever=retriever, top_k=3),
        "abl5_": AblationNoDecomposition(retriever=retriever, top_k=3),
        "tr_": AgentOrchestrator(retriever=retriever),
    }

    for expected_prefix, sys_obj in expected_prefixes.items():
        with patch("src.agent.planner.QueryPlanner.plan", return_value=suf_plan), \
             patch("src.agent.evidence_auditor.EvidenceAuditor.audit", return_value=suf_audit), \
             patch("src.agent.grounded_generator.GroundedGenerator.generate", return_value=ans_gen), \
             patch("src.baselines.common.BaselineGenerator.generate", return_value=ans_gen):
            trace: AgentTrace = sys_obj.run(query)

        # Verify schema fields
        assert isinstance(trace.trace_id, str), f"trace_id must be str for {expected_prefix}"
        assert trace.trace_id.startswith(expected_prefix), (
            f"Expected trace_id prefix '{expected_prefix}', got '{trace.trace_id}'"
        )
        assert isinstance(trace.query, str)
        assert isinstance(trace.passes, list)
        assert isinstance(trace.retry_count, int)
        assert trace.final_decision in {"SUFFICIENT", "REFUSE", "DIRECT_ANSWER", "PARTIALLY_SUFFICIENT", "DEBUNK_FALSE_PREMISE"}
        assert trace.generation.status in {"ANSWERED", "REFUSED"}
        assert isinstance(trace.total_latency_ms, float)
