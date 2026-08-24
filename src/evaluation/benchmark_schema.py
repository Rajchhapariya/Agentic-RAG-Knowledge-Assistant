"""
Benchmark Data Models: Defines structured schemas for manually verified benchmark questions
and question-level evaluation records.
"""

from typing import List, Optional, Literal, Dict, Any
from pydantic import BaseModel, Field


BenchmarkCategory = Literal[
    "answerable_single_hop",
    "answerable_multi_hop",
    "retrieval_sensitive",
    "exact_numerical",
    "ambiguous",
    "false_premise",
    "out_of_scope_unanswerable",
    "entity_near_neighbor_confusion",
    "contradiction_conflicting_evidence"
]

BenchmarkSplit = Literal["dev", "test"]


class BenchmarkQuestion(BaseModel):
    """A single manually verified evaluation benchmark item."""
    question_id: str = Field(..., description="Unique question identifier (e.g. Q01_dev)")
    question: str = Field(..., description="The user query text")
    category: BenchmarkCategory = Field(..., description="Adversarial or factual question category")
    split: BenchmarkSplit = Field(..., description="'dev' for threshold selection, 'test' for final evaluation")
    answerable: bool = Field(..., description="Whether the question can be factually answered from the corpus")
    expected_behavior: Literal["ANSWER", "DEBUNK_FALSE_PREMISE", "REFUSE_OUT_OF_SCOPE"] = Field(
        "ANSWER", description="Expected ideal behavior: direct answer, premise debunking, or out-of-scope refusal"
    )
    expected_doc_ids: List[str] = Field(default_factory=list, description="Target document IDs")
    expected_chunk_ids: List[str] = Field(default_factory=list, description="Target chunk IDs containing evidence")
    expected_section_titles: List[str] = Field(default_factory=list, description="Target section titles")
    required_evidence_spans: List[str] = Field(default_factory=list, description="Key factual evidence substrings")
    acceptable_answer_criteria: str = Field(..., description="Human rubric for grading answer correctness")
    expected_refusal: bool = Field(..., description="Whether a faithful system MUST refuse to answer")
    expected_citations: List[str] = Field(default_factory=list, description="Expected document IDs in citations")


class QuestionEvaluationRecord(BaseModel):
    """Detailed question-level evaluation record preserving all intermediate metrics and judgments."""
    question_id: str
    system_name: str
    split: str
    category: str
    answerable: bool
    expected_behavior: str = "ANSWER"
    expected_refusal: bool
    
    # Retrieval metrics (multi-view breakdown)
    retrieved_chunk_ids: List[str] = Field(default_factory=list)
    retrieved_doc_ids: List[str] = Field(default_factory=list)
    pass1_chunk_recall_at_k: float = 0.0
    cumulative_pool_chunk_recall: float = 0.0
    best_pass_chunk_recall_at_k: float = 0.0
    per_subquery_chunk_recall: float = 0.0
    chunk_precision_at_k: float = 0.0
    chunk_recall_at_k: float = 0.0
    doc_precision_at_k: float = 0.0
    doc_recall_at_k: float = 0.0
    mrr: float = 0.0
    
    # Generation & decision metrics
    final_decision: str
    generation_status: str
    response_text: str
    correctness_score: float = 0.0  # 1.0 = Correct, 0.5 = Partial, 0.0 = Wrong
    correctness_label: Literal["Correct", "Partially Correct", "Wrong", "Appropriately Refused", "Correctly Debunked"]
    citation_precision: float = 0.0
    citation_coverage: float = 0.0
    is_hallucination: bool = False
    is_true_refusal: bool = False
    is_false_refusal: bool = False
    is_correct_debunk: bool = False
    
    # Latency and engineering metrics
    total_latency_ms: float = 0.0
    planner_latency_ms: float = 0.0
    retrieval_latency_ms: float = 0.0
    auditor_latency_ms: float = 0.0
    generator_latency_ms: float = 0.0
    num_llm_calls: int = 0
    retry_count: int = 0
    
    # Trace ID for debugging
    trace_id: str = ""
    candidate_failure_category: Optional[str] = None
