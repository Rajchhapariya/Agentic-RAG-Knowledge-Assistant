"""
tests/conftest.py — Shared Pytest Fixtures and Mock Boundaries

Design principle:
  REAL application logic + FAKE LLM responses

The mocks in this file replace ONLY the external OpenAI API boundary:
  - QueryPlanner.plan()        → deterministic QueryPlan (no LLM call)
  - EvidenceAuditor.audit()    → deterministic AuditResult (no LLM call)
  - GroundedGenerator.generate() → deterministic GenerationResult (no LLM call)

Everything else (orchestrator loop, state machine, retry counter, retriever,
BM25, RRF, quote verification, schema validation) runs REAL code.

Integration tests that need real API calls are marked @pytest.mark.integration
and excluded from the default pytest run (see pytest.ini).
"""

import json
import pytest
from unittest.mock import patch, MagicMock
from typing import Iterator, List, Optional

from src.models.trace import (
    QueryPlan,
    AuditResult,
    AuditResult,
    EvidenceRelationship,
    ContradictionDetail,
    GenerationResult,
    CitationItem,
)
from src.config import CHUNKS_JSON_PATH
from src.models.document import Chunk
from src.retrieval.hybrid_retriever import HybridRetriever


@pytest.fixture(scope="session")
def loaded_chunks() -> List[Chunk]:
    """Loads indexed chunks from chunks.json."""
    with open(CHUNKS_JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [Chunk(**item) for item in data]


@pytest.fixture(scope="session")
def retriever(loaded_chunks) -> HybridRetriever:
    """Builds a real HybridRetriever backed by local embeddings cache (0 API calls)."""
    return HybridRetriever.from_chunks(loaded_chunks)


# ---------------------------------------------------------------------------
# Canonical Fixture Factories — realistic Pydantic domain objects
# ---------------------------------------------------------------------------

def make_query_plan(
    query_type: str = "FACTUAL_SINGLE_HOP",
    sub_questions: Optional[List[str]] = None,
    search_queries: Optional[List[str]] = None,
    needs_retrieval: bool = True,
    filters: Optional[dict] = None,
) -> QueryPlan:
    """Build a realistic QueryPlan without calling OpenAI."""
    sq = sub_questions or ["What are the reflection tokens in Self-RAG?"]
    return QueryPlan(
        query_type=query_type,
        needs_retrieval=needs_retrieval,
        direct_response=None if needs_retrieval else "Hello! I am the Agentic RAG Assistant.",
        target_concepts=["Self-RAG", "reflection tokens"],
        sub_questions=sq,
        initial_search_queries=search_queries or sq,
        suggested_filters=filters,
    )


def make_sufficient_audit(
    sub_questions: Optional[List[str]] = None,
    chunk_id: str = "SelfRAG_Asai_2023_c001",
    doc_id: str = "SelfRAG_Asai_2023",
    quote: str = "Self-RAG introduces special reflection tokens: [Retrieve], [IsREL], [IsSUP], [IsUSE].",
) -> AuditResult:
    """Build a SUFFICIENT AuditResult with one verified supported span."""
    sq = sub_questions or ["What are the reflection tokens in Self-RAG?"]
    rel = EvidenceRelationship(
        subquestion_idx=0,
        subquestion_text=sq[0],
        chunk_id=chunk_id,
        doc_id=doc_id,
        section_title="3 Self-RAG Framework",
        exact_quote=quote,
        relationship="SUPPORTED",
        is_quote_verified=True,
        justification="Direct match found in chunk content.",
    )
    return AuditResult(
        verdict="SUFFICIENT",
        subquestion_coverage={i: True for i in range(len(sq))},
        evidence_relationships=[rel],
        verified_supported_spans=[rel],
        missing_information=[],
        contradiction=ContradictionDetail(),
        diagnosed_search_gap=None,
    )


def make_insufficient_retry_audit(
    sub_questions: Optional[List[str]] = None,
    diagnosed_gap: str = "specific threshold values",
) -> AuditResult:
    """Build an INSUFFICIENT_RETRY AuditResult (triggers reformulation)."""
    sq = sub_questions or ["What are the reflection tokens in Self-RAG?"]
    return AuditResult(
        verdict="INSUFFICIENT_RETRY",
        subquestion_coverage={i: False for i in range(len(sq))},
        evidence_relationships=[],
        verified_supported_spans=[],
        missing_information=[f"Sub-question 1 ('{sq[0]}') has no verified supporting evidence."],
        contradiction=ContradictionDetail(),
        diagnosed_search_gap=diagnosed_gap,
    )


def make_definitively_absent_audit(
    sub_questions: Optional[List[str]] = None,
) -> AuditResult:
    """Build a DEFINITIVELY_ABSENT AuditResult (triggers REFUSE)."""
    sq = sub_questions or ["What is quantum teleportation fidelity in 2026?"]
    return AuditResult(
        verdict="DEFINITIVELY_ABSENT",
        subquestion_coverage={i: False for i in range(len(sq))},
        evidence_relationships=[],
        verified_supported_spans=[],
        missing_information=["No relevant passages were retrieved from the 10-paper corpus."],
        contradiction=ContradictionDetail(),
        diagnosed_search_gap=None,
    )


def make_debunk_audit(
    sub_questions: Optional[List[str]] = None,
    premise_claim: str = "RAG-Sequence performs backpropagation through BM25 indexes",
    refutation_claim: str = "RAG-Sequence uses dense retrieval via DPR, not BM25 indexing.",
    quote: str = "RAG architectures use dense retrieval with DPR to retrieve documents.",
    doc_id: str = "RAG_Lewis_2020",
    chunk_id: str = "RAG_Lewis_2020_c001",
) -> AuditResult:
    """Build a DEBUNK_FALSE_PREMISE AuditResult with verified refuting span."""
    sq = sub_questions or ["How does RAG-Sequence backpropagate through BM25?"]
    ref_rel = EvidenceRelationship(
        subquestion_idx=0,
        subquestion_text=sq[0],
        chunk_id=chunk_id,
        doc_id=doc_id,
        section_title="2 Methods",
        exact_quote=quote,
        relationship="CONTRADICTED",
        is_quote_verified=True,
        justification=f"Evidence contradicts premise: {refutation_claim}",
    )
    return AuditResult(
        verdict="DEBUNK_FALSE_PREMISE",
        subquestion_coverage={0: False},
        evidence_relationships=[ref_rel],
        verified_supported_spans=[ref_rel],
        missing_information=[],
        contradiction=ContradictionDetail(
            has_conflict=True,
            is_false_premise=True,
            claim_a=premise_claim,
            source_a="User Question",
            claim_b=refutation_claim,
            source_b=doc_id,
            conflict_summary=f"The question premise ('{premise_claim}') is contradicted by {doc_id}, which shows {refutation_claim}"
        ),
        diagnosed_search_gap=None,
    )


def make_contradiction_audit(
    sub_questions: Optional[List[str]] = None,
    claim_a: str = "Dense embeddings underperform BM25 in zero-shot settings.",
    source_a: str = "RAG_Lewis_2020",
    claim_b: str = "DPR dual-encoders consistently outperform BM25.",
    source_b: str = "DPR_Karpukhin_2020",
) -> AuditResult:
    """Build an inter-document CONTRADICTED AuditResult."""
    sq = sub_questions or ["How does dense retrieval compare to BM25?"]
    span_a = EvidenceRelationship(
        subquestion_idx=0,
        subquestion_text=sq[0],
        chunk_id="RAG_Lewis_2020_c005",
        doc_id=source_a,
        section_title="4 Results",
        exact_quote=claim_a,
        relationship="SUPPORTED",
        is_quote_verified=True,
        justification="RAG finding on zero-shot retrieval"
    )
    span_b = EvidenceRelationship(
        subquestion_idx=0,
        subquestion_text=sq[0],
        chunk_id="DPR_Karpukhin_2020_c003",
        doc_id=source_b,
        section_title="5 Results",
        exact_quote=claim_b,
        relationship="SUPPORTED",
        is_quote_verified=True,
        justification="DPR finding on open-domain retrieval"
    )
    return AuditResult(
        verdict="CONTRADICTED",
        subquestion_coverage={0: True},
        evidence_relationships=[span_a, span_b],
        verified_supported_spans=[span_a, span_b],
        missing_information=[],
        contradiction=ContradictionDetail(
            has_conflict=True,
            is_false_premise=False,
            claim_a=claim_a,
            source_a=source_a,
            claim_b=claim_b,
            source_b=source_b,
            conflict_summary="Empirical divergence between RAG and DPR on zero-shot domain transfer."
        ),
        diagnosed_search_gap=None,
    )


def make_answered_generation(
    query: str = "What are the reflection tokens in Self-RAG?",
    doc_id: str = "SelfRAG_Asai_2023",
    quote: str = "Self-RAG introduces special reflection tokens: [Retrieve], [IsREL], [IsSUP], [IsUSE].",
) -> GenerationResult:
    """Build a realistic ANSWERED GenerationResult."""
    return GenerationResult(
        status="ANSWERED",
        response_text=(
            f"Self-RAG introduces four special reflection tokens: [Retrieve], [IsREL], [IsSUP], and [IsUSE]. "
            f"These tokens guide selective retrieval and self-critique [{doc_id}]."
        ),
        citations=[
            CitationItem(
                citation_id=1,
                doc_id=doc_id,
                document_title="Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection",
                section_title="3 Self-RAG Framework",
                chunk_id="SelfRAG_Asai_2023_c001",
                exact_quote=quote,
                claim_text="Self-RAG uses four reflection tokens to guide selective retrieval.",
            )
        ],
        refusal_reason=None,
        missing_caveats=None,
        has_conflict_acknowledged=False,
    )


def make_refused_generation(
    missing_info: Optional[List[str]] = None,
) -> GenerationResult:
    """Build a realistic REFUSED GenerationResult."""
    reasons = missing_info or ["No relevant passages were retrieved from the 10-paper corpus."]
    bullet_list = "\n".join(f"- {r}" for r in reasons)
    return GenerationResult(
        status="REFUSED",
        response_text=(
            "Based on the 10 research papers in the index, there is insufficient evidence to answer this question.\n\n"
            f"**Missing Evidence Details:**\n{bullet_list}\n\n"
            "*The system refused to generate an unverified answer to prevent hallucination.*"
        ),
        citations=[],
        refusal_reason=bullet_list,
        missing_caveats=None,
        has_conflict_acknowledged=False,
    )


# ---------------------------------------------------------------------------
# Shared Component-Level Mock Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_plan_sufficient() -> QueryPlan:
    """A standard FACTUAL_SINGLE_HOP plan for answerable queries."""
    return make_query_plan()


@pytest.fixture
def mock_plan_multi_hop() -> QueryPlan:
    """A MULTI_HOP_COMPARATIVE plan with two sub-questions."""
    return make_query_plan(
        query_type="MULTI_HOP_COMPARATIVE",
        sub_questions=[
            "How does Self-RAG decide when to retrieve?",
            "How does MemGPT manage virtual context?",
        ],
        search_queries=[
            "Self-RAG selective retrieval decision tokens",
            "MemGPT virtual memory OS paging",
        ],
    )


@pytest.fixture
def mock_plan_direct() -> QueryPlan:
    """A DIRECT_CONVERSATIONAL plan (no retrieval needed)."""
    return make_query_plan(
        query_type="DIRECT_CONVERSATIONAL",
        needs_retrieval=False,
        sub_questions=[],
        search_queries=[],
    )


@pytest.fixture
def mock_audit_sufficient() -> AuditResult:
    """A SUFFICIENT audit result with one verified supported span."""
    return make_sufficient_audit()


@pytest.fixture
def mock_audit_insufficient() -> AuditResult:
    """An INSUFFICIENT_RETRY audit result."""
    return make_insufficient_retry_audit()


@pytest.fixture
def mock_audit_absent() -> AuditResult:
    """A DEFINITIVELY_ABSENT audit result."""
    return make_definitively_absent_audit()


@pytest.fixture
def mock_generation_answered() -> GenerationResult:
    """A realistic ANSWERED GenerationResult."""
    return make_answered_generation()


@pytest.fixture
def mock_generation_refused() -> GenerationResult:
    """A realistic REFUSED GenerationResult."""
    return make_refused_generation()


# ---------------------------------------------------------------------------
# Planner Patch Fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def patch_planner_sufficient(mock_plan_sufficient):
    """Patches QueryPlanner.plan() to return a SUFFICIENT (factual single-hop) plan."""
    with patch("src.agent.planner.QueryPlanner.plan", return_value=mock_plan_sufficient) as m:
        yield m


@pytest.fixture
def patch_planner_multi_hop(mock_plan_multi_hop):
    """Patches QueryPlanner.plan() to return a MULTI_HOP plan."""
    with patch("src.agent.planner.QueryPlanner.plan", return_value=mock_plan_multi_hop) as m:
        yield m


@pytest.fixture
def patch_planner_direct(mock_plan_direct):
    """Patches QueryPlanner.plan() to return a DIRECT_CONVERSATIONAL plan."""
    with patch("src.agent.planner.QueryPlanner.plan", return_value=mock_plan_direct) as m:
        yield m


# ---------------------------------------------------------------------------
# Auditor Patch Fixtures — including retry sequence control
# ---------------------------------------------------------------------------

@pytest.fixture
def patch_auditor_always_sufficient(mock_audit_sufficient):
    """Patches EvidenceAuditor.audit() to always return SUFFICIENT."""
    with patch("src.agent.evidence_auditor.EvidenceAuditor.audit", return_value=mock_audit_sufficient) as m:
        yield m


@pytest.fixture
def patch_auditor_always_absent(mock_audit_absent):
    """Patches EvidenceAuditor.audit() to always return DEFINITIVELY_ABSENT."""
    with patch("src.agent.evidence_auditor.EvidenceAuditor.audit", return_value=mock_audit_absent) as m:
        yield m


@pytest.fixture
def patch_auditor_retry_then_sufficient(mock_audit_insufficient, mock_audit_sufficient):
    """
    Patches EvidenceAuditor.audit() to return:
      Pass 1 → INSUFFICIENT_RETRY
      Pass 2 → SUFFICIENT
    Tests that the orchestrator correctly retries and succeeds.
    """
    side_effects = [mock_audit_insufficient, mock_audit_sufficient]
    with patch("src.agent.evidence_auditor.EvidenceAuditor.audit", side_effect=side_effects) as m:
        yield m


@pytest.fixture
def patch_auditor_retry_retry_absent(mock_audit_insufficient, mock_audit_absent):
    """
    Patches EvidenceAuditor.audit() to return:
      Pass 1 → INSUFFICIENT_RETRY
      Pass 2 → INSUFFICIENT_RETRY
      Pass 3 → DEFINITIVELY_ABSENT
    Tests that the orchestrator exhausts retries and refuses.
    """
    side_effects = [mock_audit_insufficient, mock_audit_insufficient, mock_audit_absent]
    with patch("src.agent.evidence_auditor.EvidenceAuditor.audit", side_effect=side_effects) as m:
        yield m


# ---------------------------------------------------------------------------
# Generator Patch Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def patch_generator_answered(mock_generation_answered):
    """Patches GroundedGenerator.generate() to always return ANSWERED."""
    with patch("src.agent.grounded_generator.GroundedGenerator.generate", return_value=mock_generation_answered) as m:
        yield m


@pytest.fixture
def patch_generator_refused(mock_generation_refused):
    """Patches GroundedGenerator.generate() to always return REFUSED."""
    with patch("src.agent.grounded_generator.GroundedGenerator.generate", return_value=mock_generation_refused) as m:
        yield m


# ---------------------------------------------------------------------------
# Baseline Generator Patch (for NaiveRAG / HybridRAG which use BaselineGenerator)
# ---------------------------------------------------------------------------

@pytest.fixture
def patch_baseline_generator_answered(mock_generation_answered):
    """Patches baselines.common.BaselineGenerator.generate() to return ANSWERED."""
    with patch("src.baselines.common.BaselineGenerator.generate", return_value=mock_generation_answered) as m:
        yield m


# ---------------------------------------------------------------------------
# Reformulator Patch — deterministic, no LLM (already is deterministic, but
# patch to be sure and to enable controlled query reformulation verification)
# ---------------------------------------------------------------------------

@pytest.fixture
def patch_reformulator():
    """Patches QueryReformulator.reformulate() to return a fixed reformulated query string."""
    reformulated = "CRAG retrieval evaluator confidence threshold upper lower bound ambiguous action"
    with patch("src.agent.reformulator.QueryReformulator.reformulate", return_value=reformulated) as m:
        yield m
