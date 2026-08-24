"""
Phase 5 Automated Test Suite: Benchmark Integrity, Split Separation, Exact Metric Calculations, and Runner Harness.
"""

import os
import json
import pytest
from src.config import CHUNKS_JSON_PATH
from src.evaluation.benchmark_schema import BenchmarkQuestion, QuestionEvaluationRecord
from src.evaluation.metrics import (
    compute_precision_at_k,
    compute_recall_at_k,
    compute_mrr,
    compute_aggregate_metrics,
    evaluate_question_trace,
)
from src.evaluation.runner import ExperimentHarness, BENCHMARK_PATH
from src.models.trace import AgentTrace, QueryPlan, PassRecord, AuditResult, GenerationResult, CitationItem


# ---------------------------------------------------------------------------
# Test 1: Benchmark Dataset Integrity & Schema Validation
# ---------------------------------------------------------------------------
def test_benchmark_dataset_integrity():
    assert os.path.exists(BENCHMARK_PATH), f"Benchmark dataset file missing at {BENCHMARK_PATH}"
    with open(BENCHMARK_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert len(data) == 40, f"Expected 40 benchmark questions, found {len(data)}"
    
    questions = [BenchmarkQuestion(**item) for item in data]
    dev_qs = [q for q in questions if q.split == "dev"]
    test_qs = [q for q in questions if q.split == "test"]

    assert len(dev_qs) == 8, f"Expected 8 dev questions, found {len(dev_qs)}"
    assert len(test_qs) == 32, f"Expected 32 test questions, found {len(test_qs)}"

    for q in questions:
        assert q.question_id.startswith("Q")
        assert len(q.question) > 10
        assert q.acceptable_answer_criteria is not None
        if not q.answerable and q.expected_behavior == "REFUSE_OUT_OF_SCOPE":
            assert q.expected_refusal is True


# ---------------------------------------------------------------------------
# Test 2: Ground Truth Chunk Fidelity (All Expected Chunks Must Exist)
# ---------------------------------------------------------------------------
def test_benchmark_ground_truth_chunk_fidelity():
    with open(CHUNKS_JSON_PATH, "r", encoding="utf-8") as f:
        chunk_ids = {item["chunk_id"] for item in json.load(f)}

    with open(BENCHMARK_PATH, "r", encoding="utf-8") as f:
        questions = [BenchmarkQuestion(**item) for item in json.load(f)]

    for q in questions:
        for cid in q.expected_chunk_ids:
            assert cid in chunk_ids, f"Question {q.question_id} references non-existent chunk {cid}"


# ---------------------------------------------------------------------------
# Test 3: Dev/Test Split Separation & ID Disjointness
# ---------------------------------------------------------------------------
def test_train_dev_test_separation():
    with open(BENCHMARK_PATH, "r", encoding="utf-8") as f:
        questions = [BenchmarkQuestion(**item) for item in json.load(f)]

    dev_ids = {q.question_id for q in questions if q.split == "dev"}
    test_ids = {q.question_id for q in questions if q.split == "test"}

    assert len(dev_ids & test_ids) == 0, "Dev and Test splits contain overlapping question IDs!"


# ---------------------------------------------------------------------------
# Test 4: Toy Retrieval Metrics Validation
# ---------------------------------------------------------------------------
def test_retrieval_metrics_toy_examples():
    expected = ["chunk_A", "chunk_B"]
    
    # Case 1: Perfect retrieval top 2
    retrieved_1 = ["chunk_A", "chunk_B", "chunk_C", "chunk_D", "chunk_E"]
    assert compute_precision_at_k(retrieved_1, expected, k=5) == 2 / 5.0
    assert compute_recall_at_k(retrieved_1, expected, k=5) == 1.0
    assert compute_mrr(retrieved_1, expected) == 1.0

    # Case 2: Hit at rank 3
    retrieved_2 = ["chunk_X", "chunk_Y", "chunk_A", "chunk_Z", "chunk_W"]
    assert compute_precision_at_k(retrieved_2, expected, k=5) == 1 / 5.0
    assert compute_recall_at_k(retrieved_2, expected, k=5) == 0.5
    assert compute_mrr(retrieved_2, expected) == 1 / 3.0

    # Case 3: Complete miss
    retrieved_3 = ["chunk_X", "chunk_Y", "chunk_Z", "chunk_W", "chunk_V"]
    assert compute_precision_at_k(retrieved_3, expected, k=5) == 0.0
    assert compute_recall_at_k(retrieved_3, expected, k=5) == 0.0
    assert compute_mrr(retrieved_3, expected) == 0.0


# ---------------------------------------------------------------------------
# Test 5: Hallucination & Refusal Metrics Mathematical Correctness
# ---------------------------------------------------------------------------
def test_hallucination_and_refusal_metrics_toy_examples():
    # Construct 4 toy records:
    # 2 Answerable: 1 correct, 1 false refusal
    # 2 Unanswerable: 1 true refusal, 1 hallucination
    records = [
        QuestionEvaluationRecord(
            question_id="Q1", system_name="ToySys", split="dev", category="answerable_single_hop",
            answerable=True, expected_behavior="ANSWER", expected_refusal=False, final_decision="SUFFICIENT", generation_status="ANSWERED",
            response_text="Correct answer", correctness_score=1.0, correctness_label="Correct",
            is_hallucination=False, is_true_refusal=False, is_false_refusal=False
        ),
        QuestionEvaluationRecord(
            question_id="Q2", system_name="ToySys", split="dev", category="answerable_single_hop",
            answerable=True, expected_behavior="ANSWER", expected_refusal=False, final_decision="REFUSE", generation_status="REFUSED",
            response_text="I refuse", correctness_score=0.0, correctness_label="Wrong",
            is_hallucination=False, is_true_refusal=False, is_false_refusal=True
        ),
        QuestionEvaluationRecord(
            question_id="Q3", system_name="ToySys", split="dev", category="out_of_scope_unanswerable",
            answerable=False, expected_behavior="REFUSE_OUT_OF_SCOPE", expected_refusal=True, final_decision="REFUSE", generation_status="REFUSED",
            response_text="I refuse", correctness_score=1.0, correctness_label="Appropriately Refused",
            is_hallucination=False, is_true_refusal=True, is_false_refusal=False
        ),
        QuestionEvaluationRecord(
            question_id="Q4", system_name="ToySys", split="dev", category="out_of_scope_unanswerable",
            answerable=False, expected_behavior="REFUSE_OUT_OF_SCOPE", expected_refusal=True, final_decision="SUFFICIENT", generation_status="ANSWERED",
            response_text="Hallucinated answer", correctness_score=0.0, correctness_label="Wrong",
            is_hallucination=True, is_true_refusal=False, is_false_refusal=False
        ),
    ]

    agg = compute_aggregate_metrics(records, system_name="ToySys", split="dev")
    
    assert agg.num_questions == 4
    assert agg.num_answerable == 2
    assert agg.num_unanswerable == 2
    assert agg.answer_accuracy == 0.5  # (1.0 + 0.0) / 2
    assert agg.hallucination_rate == 0.5  # 1 hallucination out of 2 unanswerable
    assert agg.true_refusal_rate == 0.5  # 1 true refusal out of 2 unanswerable
    assert agg.false_refusal_rate == 0.5  # 1 false refusal out of 2 answerable


# ---------------------------------------------------------------------------
# Test 6: Experiment Harness Initialization & System Registry
# ---------------------------------------------------------------------------
def test_experiment_runner_initialization():
    harness = ExperimentHarness()
    assert len(harness.systems) == 8
    expected_systems = {
        "Baseline_A_NaiveDense",
        "Baseline_B_HybridRAG",
        "Ablation_1_NoPlanner",
        "Ablation_2_DenseOnlyAgentic",
        "Ablation_3_NoSufficiencyChecker",
        "Ablation_4_NoRetryLoop",
        "Ablation_5_NoDecomposition",
        "Agentic_System_C_Full",
    }
    assert set(harness.systems.keys()) == expected_systems


# ---------------------------------------------------------------------------
# Test 7: Canonical Document ID Normalization
# ---------------------------------------------------------------------------
def test_canonical_doc_id_normalization():
    from src.agent.planner import normalize_canonical_doc_id, CANONICAL_DOC_IDS

    assert len(CANONICAL_DOC_IDS) == 10
    assert normalize_canonical_doc_id("CRAG") == "CRAG_Yan_2024"
    assert normalize_canonical_doc_id("crag") == "CRAG_Yan_2024"
    assert normalize_canonical_doc_id("Self-RAG") == "SelfRAG_Asai_2023"
    assert normalize_canonical_doc_id("Toolformer") == "Toolformer_Schick_2023"
    assert normalize_canonical_doc_id("DPR_Karpukhin_2020") == "DPR_Karpukhin_2020"
    assert normalize_canonical_doc_id("Invalid_Doc_XYZ") is None
    assert normalize_canonical_doc_id("") is None
    assert normalize_canonical_doc_id(None) is None


# ---------------------------------------------------------------------------
# Test 8: Metadata Filter Zero-Result Fallback Behavior
# ---------------------------------------------------------------------------
def test_filter_zero_result_fallback():
    from src.retrieval.hybrid_retriever import HybridRetriever
    from src.models.document import Chunk, ChunkMetadata

    dummy_chunks = [
        Chunk(
            chunk_id="RAG_Lewis_2020_c001",
            content="RAG paper text on retrieval augmented generation.",
            metadata=ChunkMetadata(
                chunk_id="RAG_Lewis_2020_c001",
                doc_id="RAG_Lewis_2020",
                document_title="RAG Paper",
                authors="Lewis et al.",
                year=2020,
                section_id="intro",
                section_title="1 Introduction",
                chunk_index_in_doc=0,
                chunk_index_in_section=0,
                token_count=10,
                char_start=0,
                char_end=50
            )
        )
    ]
    retriever = HybridRetriever.from_chunks(dummy_chunks)
    
    hits_filtered = retriever.retrieve("retrieval", top_k=5, filters={"doc_id": "NonExistentDoc"})
    assert len(hits_filtered) == 0

    hits_fallback = retriever.retrieve("retrieval", top_k=5, filters=None)
    assert len(hits_fallback) == 1
    assert hits_fallback[0].chunk_id == "RAG_Lewis_2020_c001"


# ---------------------------------------------------------------------------
# Test 9: False Premise Debunking vs Unanswerable Refusal
# ---------------------------------------------------------------------------
def test_false_premise_debunk_vs_unanswerable_evaluation():
    q_debunk = BenchmarkQuestion(
        question_id="Q06_dev",
        question="How does Toolformer use PPO?",
        category="false_premise",
        split="dev",
        answerable=False,
        expected_behavior="DEBUNK_FALSE_PREMISE",
        expected_refusal=False,
        acceptable_answer_criteria="Debunk PPO",
        required_evidence_spans=["does not rely on reinforcement"]
    )
    
    trace_debunk = AgentTrace(
        trace_id="tr_debunk",
        query="How does Toolformer use PPO?",
        planner=QueryPlan(query_type="FACTUAL_SINGLE_HOP", needs_retrieval=True, target_concepts=["Toolformer"], sub_questions=["Toolformer PPO"], initial_search_queries=["Toolformer PPO"]),
        passes=[],
        retry_count=0,
        final_decision="DEBUNK_FALSE_PREMISE",
        generation=GenerationResult(
            status="ANSWERED",
            response_text="Toolformer does not rely on reinforcement learning or PPO [Toolformer_Schick_2023].",
            citations=[]
        )
    )
    
    rec = evaluate_question_trace(q_debunk, trace_debunk, system_name="Agentic_System_C_Full")
    assert rec.is_correct_debunk is True
    assert rec.correctness_score == 1.0
    assert rec.correctness_label == "Correctly Debunked"
    assert rec.is_hallucination is False
