"""
Unit tests for Evidence Auditing Refinements:
1. Conservative multi-stage quote normalization (scientific notation, whitespace, Unicode, LaTeX)
2. False-premise detection and debunking routing (DEBUNK_FALSE_PREMISE)
3. Inter-document contradiction handling (CONTRADICTED)
4. Hallucination and alteration rejection
5. Deterministic orchestrator routing for all 4 policy outcomes (ANSWER, REFUSE, DEBUNK, CONTRADICT)
"""

import pytest
from unittest.mock import patch
from src.agent.evidence_auditor import normalize_text_for_matching, verify_quote_in_text, EvidenceAuditor
from src.models.retrieval import SearchResult
from src.models.trace import EvidenceRelationship, ContradictionDetail, AuditResult, AgentTrace
from src.agent.orchestrator import AgentOrchestrator
from conftest import (
    make_query_plan,
    make_sufficient_audit,
    make_definitively_absent_audit,
    make_debunk_audit,
    make_contradiction_audit,
    make_answered_generation,
    make_refused_generation,
)


# ===========================================================================
# Section 1: Normalization and Quote Verification Unit Tests (Pure Python)
# ===========================================================================

def test_scientific_notation_equivalent_formats_accepted():
    """
    Verifies that equivalent scientific notation representations (10^-5, 1e-5, 10^{-5}, 10⁻⁵, 1*10^-5)
    are correctly matched and accepted without altering the underlying content.
    """
    content = "We train the encoders using Adam with a learning rate of 10^-5 and linear warmup."

    # Equivalent representations extracted by LLM or search
    assert verify_quote_in_text("learning rate of 10^-5", content) is True
    assert verify_quote_in_text("learning rate of 1e-5", content) is True
    assert verify_quote_in_text("learning rate of 10^{-5}", content) is True
    assert verify_quote_in_text("learning rate of 10⁻⁵", content) is True
    assert verify_quote_in_text("learning rate of 1 * 10^-5", content) is True
    assert verify_quote_in_text("learning rate of 1.0 * 10^-5", content) is True


def test_materially_different_numbers_rejected():
    """
    Verifies that materially different numbers (e.g. 1e-4 vs 10^-5, 0.05 vs 0.01)
    are strictly rejected by the normalization layer.
    """
    content = "We train the encoders using Adam with a learning rate of 10^-5."

    assert verify_quote_in_text("learning rate of 1e-4", content) is False
    assert verify_quote_in_text("learning rate of 10^-6", content) is False
    assert verify_quote_in_text("learning rate of 0.001", content) is False
    assert verify_quote_in_text("learning rate of 2e-5", content) is False


def test_altered_wording_rejected():
    """
    Verifies that altered, paraphrased, or hallucinated wording is rejected
    even if it shares semantic keywords with the chunk.
    """
    content = "Self-RAG introduces special reflection tokens: [Retrieve], [IsREL], [IsSUP], [IsUSE]."

    # Paraphrased / altered quotes
    assert verify_quote_in_text("Self-RAG utilizes unique critique tags", content) is False
    assert verify_quote_in_text("Self-RAG introduces special evaluation tokens", content) is False
    assert verify_quote_in_text("special reflection tokens including [Search] and [Evaluate]", content) is False


def test_fabricated_quote_rejected():
    """
    Verifies that a completely fabricated quote is strictly rejected.
    """
    content = "RAG models combine parametric memory with non-parametric retrieval."

    assert verify_quote_in_text("Quantum teleportation achieves 99% fidelity in RAG", content) is False
    assert verify_quote_in_text("Backpropagation through BM25 is accomplished using Gumbel-Softmax", content) is False


def test_whitespace_unicode_linebreak_accepted():
    """
    Verifies that formatting variations such as linebreaks, non-breaking spaces,
    and smart quotes are normalized and accepted.
    """
    content = "According to Lewis et al. (2020), “RAG-Sequence uses the same document\nfor all target tokens.”"

    # Smart quote vs straight quote, line-break vs single space
    quote_with_straight_quotes = 'According to Lewis et al. (2020), "RAG-Sequence uses the same document for all target tokens."'
    assert verify_quote_in_text(quote_with_straight_quotes, content) is True

    # Multi-space / newline variation
    quote_with_newlines = "RAG-Sequence uses the   same document\nfor all target tokens."
    assert verify_quote_in_text(quote_with_newlines, content) is True


# ===========================================================================
# Section 2: Policy & Decision Level Routing Tests (State Machine)
# ===========================================================================

def test_false_premise_routes_to_debunk_decision(retriever):
    """
    Verifies that when evidence contradicts a question premise, the orchestrator
    routes to DEBUNK_FALSE_PREMISE and includes refuting citations.
    """
    orchestrator = AgentOrchestrator(retriever=retriever)
    query = "How does RAG-Sequence perform backpropagation through non-differentiable BM25 indexes?"

    plan = make_query_plan(
        query_type="FACTUAL_SINGLE_HOP",
        sub_questions=["How does RAG-Sequence backpropagate through BM25?"]
    )
    debunk_audit = make_debunk_audit(
        sub_questions=["How does RAG-Sequence backpropagate through BM25?"],
        premise_claim="RAG-Sequence performs backpropagation through BM25 indexes",
        refutation_claim="RAG-Sequence uses dense DPR retrieval, not BM25 indexing.",
        quote="RAG architectures use dense retrieval with DPR to retrieve documents.",
        doc_id="RAG_Lewis_2020",
    )
    
    gen_result = make_answered_generation()
    gen_result.response_text = (
        "The premise in the question is incorrect: RAG-Sequence uses dense retrieval (DPR) "
        "rather than BM25, so it does not backpropagate through a BM25 index [RAG_Lewis_2020]."
    )

    with patch("src.agent.planner.QueryPlanner.plan", return_value=plan), \
         patch("src.agent.evidence_auditor.EvidenceAuditor.audit", return_value=debunk_audit), \
         patch("src.agent.grounded_generator.GroundedGenerator.generate", return_value=gen_result):
        trace: AgentTrace = orchestrator.run(query)

    assert trace.final_decision == "DEBUNK_FALSE_PREMISE"
    assert trace.generation.status == "ANSWERED"
    assert "incorrect" in trace.generation.response_text.lower() or "premise" in trace.generation.response_text.lower()
    assert len(trace.passes) == 1
    assert trace.passes[0].audit_result.verdict == "DEBUNK_FALSE_PREMISE"


def test_genuinely_unanswerable_routes_to_refuse(retriever):
    """
    Verifies that a genuinely out-of-scope query routes to REFUSE on DEFINITIVELY_ABSENT.
    """
    orchestrator = AgentOrchestrator(retriever=retriever)
    query = "What is the quantum teleportation fidelity achieved by Agent-Q in 2026?"

    plan = make_query_plan(
        query_type="OUT_OF_SCOPE_SUSPECT",
        sub_questions=[query]
    )
    absent_audit = make_definitively_absent_audit(sub_questions=[query])
    refuse_gen = make_refused_generation()

    with patch("src.agent.planner.QueryPlanner.plan", return_value=plan), \
         patch("src.agent.evidence_auditor.EvidenceAuditor.audit", return_value=absent_audit), \
         patch("src.agent.grounded_generator.GroundedGenerator.generate", return_value=refuse_gen):
        trace: AgentTrace = orchestrator.run(query)

    assert trace.final_decision == "REFUSE"
    assert trace.generation.status == "REFUSED"
    assert len(trace.generation.citations) == 0


def test_ordinary_supported_question_routes_to_answer(retriever):
    """
    Verifies that an ordinary answerable question with supported evidence routes to SUFFICIENT / ANSWERED.
    """
    orchestrator = AgentOrchestrator(retriever=retriever)
    query = "What are the reflection tokens in Self-RAG?"

    plan = make_query_plan(sub_questions=[query])
    suf_audit = make_sufficient_audit(sub_questions=[query])
    ans_gen = make_answered_generation()

    with patch("src.agent.planner.QueryPlanner.plan", return_value=plan), \
         patch("src.agent.evidence_auditor.EvidenceAuditor.audit", return_value=suf_audit), \
         patch("src.agent.grounded_generator.GroundedGenerator.generate", return_value=ans_gen):
        trace: AgentTrace = orchestrator.run(query)

    assert trace.final_decision == "SUFFICIENT"
    assert trace.generation.status == "ANSWERED"
    assert len(trace.generation.citations) >= 1


def test_inter_document_contradiction_routes_to_contradicted(retriever):
    """
    Verifies that inter-document empirical conflict routes to CONTRADICTED / cited synthesis.
    """
    orchestrator = AgentOrchestrator(retriever=retriever)
    query = "How does dense retrieval compare to BM25 on out-of-domain datasets?"

    plan = make_query_plan(sub_questions=[query])
    conflict_audit = make_contradiction_audit(sub_questions=[query])
    ans_gen = make_answered_generation()
    ans_gen.has_conflict_acknowledged = True

    with patch("src.agent.planner.QueryPlanner.plan", return_value=plan), \
         patch("src.agent.evidence_auditor.EvidenceAuditor.audit", return_value=conflict_audit), \
         patch("src.agent.grounded_generator.GroundedGenerator.generate", return_value=ans_gen):
        trace: AgentTrace = orchestrator.run(query)

    assert trace.passes[0].audit_result.verdict == "CONTRADICTED"
    assert trace.passes[0].audit_result.contradiction.has_conflict is True
    assert trace.passes[0].audit_result.contradiction.is_false_premise is False
