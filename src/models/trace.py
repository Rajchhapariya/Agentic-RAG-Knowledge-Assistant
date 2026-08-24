"""
Structured Pydantic schemas for Query Planning, Evidence Auditing, Provenance, and Execution Traces.
"""

from typing import Optional, List, Dict, Any, Literal
from pydantic import BaseModel, Field
from src.models.retrieval import SearchResult


class QueryPlan(BaseModel):
    """Structured plan produced by the Query Planner."""
    query_type: Literal[
        "DIRECT_CONVERSATIONAL",
        "FACTUAL_SINGLE_HOP",
        "MULTI_HOP_COMPARATIVE",
        "SEARCH_SYNTHESIS",
        "OUT_OF_SCOPE_SUSPECT"
    ] = Field(..., description="Classification of the query type")
    needs_retrieval: bool = Field(..., description="Whether external document retrieval is needed")
    direct_response: Optional[str] = Field(None, description="Direct response for greetings/meta queries")
    target_concepts: List[str] = Field(default_factory=list, description="Key technical entities or concepts")
    sub_questions: List[str] = Field(default_factory=list, description="Decomposed atomic sub-questions (max 3)")
    initial_search_queries: List[str] = Field(default_factory=list, description="Search strings for each sub-question")
    suggested_filters: Optional[Dict[str, Any]] = Field(None, description="Optional metadata filters (e.g. doc_id)")


class EvidenceRelationship(BaseModel):
    """Detailed relationship between a retrieved chunk and a required sub-question/claim."""
    subquestion_idx: int = Field(..., description="Index of the sub-question being audited")
    subquestion_text: str = Field(..., description="Text of the sub-question")
    chunk_id: str = Field(..., description="ID of candidate chunk")
    doc_id: str = Field(..., description="Document ID of candidate chunk")
    section_title: str = Field(..., description="Section title of chunk")
    exact_quote: str = Field(..., description="Exact verbatim excerpt from the chunk text")
    relationship: Literal["SUPPORTED", "UNSUPPORTED", "CONTRADICTED", "IRRELEVANT"] = Field(
        ..., description="Evidence support status"
    )
    is_quote_verified: bool = Field(False, description="Whether Python verified the quote is a verbatim substring")
    justification: str = Field(..., description="Brief factual explanation of the relationship assignment")


class ContradictionDetail(BaseModel):
    """Details of a detected empirical contradiction between chunks or a false premise in the query."""
    has_conflict: bool = Field(False, description="Whether a material contradiction or false premise was detected")
    is_false_premise: bool = Field(False, description="Whether the question asserts a false factual premise contradicted by evidence")
    claim_a: str = Field("", description="First conflicting claim or question premise")
    source_a: str = Field("", description="Source doc_id/chunk_id or 'User Question'")
    claim_b: str = Field("", description="Second conflicting claim or refuting evidence")
    source_b: str = Field("", description="Source doc_id/chunk_id for refuting evidence")
    conflict_summary: str = Field("", description="Summary of the empirical divergence or premise refutation")


class AuditResult(BaseModel):
    """Complete output of the Evidence Auditor for a retrieval pass."""
    verdict: Literal[
        "SUFFICIENT",
        "PARTIALLY_SUFFICIENT",
        "CONTRADICTED",
        "DEBUNK_FALSE_PREMISE",
        "INSUFFICIENT_RETRY",
        "DEFINITIVELY_ABSENT"
    ] = Field(..., description="Overall sufficiency verdict")
    subquestion_coverage: Dict[int, bool] = Field(
        default_factory=dict, description="Whether each sub-question has at least one verified SUPPORTED span"
    )
    evidence_relationships: List[EvidenceRelationship] = Field(
        default_factory=list, description="Audited evidence relationships"
    )
    verified_supported_spans: List[EvidenceRelationship] = Field(
        default_factory=list, description="Filtered list of verified SUPPORTED spans"
    )
    missing_information: List[str] = Field(
        default_factory=list, description="List of missing facts/entities required to answer"
    )
    contradiction: ContradictionDetail = Field(
        default_factory=ContradictionDetail, description="Contradiction audit results"
    )
    diagnosed_search_gap: Optional[str] = Field(
        None, description="Specific diagnosed search gap for query reformulation"
    )


class CitationItem(BaseModel):
    """A granular in-text citation linking a claim to its source chunk."""
    citation_id: int = Field(..., description="1-indexed citation number [1]")
    doc_id: str = Field(..., description="Source document ID")
    document_title: str = Field(..., description="Source paper title")
    section_title: str = Field(..., description="Section title")
    chunk_id: str = Field(..., description="Source chunk ID")
    exact_quote: str = Field(..., description="Verbatim evidence quote supporting the claim")
    claim_text: str = Field(..., description="The synthesized claim supported by this citation")


class GenerationResult(BaseModel):
    """Final generation payload."""
    status: Literal["ANSWERED", "REFUSED", "DIRECT_CONVERSATIONAL"]
    response_text: str = Field(..., description="Final response with citations or explicit refusal")
    citations: List[CitationItem] = Field(default_factory=list, description="List of verified citations")
    refusal_reason: Optional[str] = Field(None, description="Detailed explanation if refused")
    missing_caveats: Optional[List[str]] = Field(None, description="Unconfirmed/missing details in partially sufficient answers")
    has_conflict_acknowledged: bool = Field(False, description="Whether conflicting sources were cited")


class PassRecord(BaseModel):
    """Record of a single Retrieve-Audit iteration."""
    pass_number: int = Field(..., description="1-indexed pass number")
    search_queries: List[str] = Field(..., description="Search queries executed in this pass")
    retrieved_chunk_ids: List[str] = Field(..., description="List of chunk IDs returned")
    retrieved_results: List[SearchResult] = Field(default_factory=list, description="Full search results")
    audit_result: AuditResult = Field(..., description="Audit output for this pass")
    reformulated_query: Optional[str] = Field(None, description="Reformulated query if retry triggered")
    filter_fallback_event: Optional[str] = Field(None, description="Event record if metadata filter yielded 0 hits and triggered fallback")


class AgentTrace(BaseModel):
    """End-to-end inspectable execution trace."""
    trace_id: str = Field(..., description="Unique trace UUID")
    query: str = Field(..., description="Original user query")
    planner: QueryPlan = Field(..., description="Query planning decision")
    passes: List[PassRecord] = Field(default_factory=list, description="History of retrieval passes")
    retry_count: int = Field(0, description="Total retries executed (0, 1, or 2)")
    final_decision: Literal["SUFFICIENT", "PARTIALLY_SUFFICIENT", "DEBUNK_FALSE_PREMISE", "REFUSE", "DIRECT_ANSWER"] = Field(..., description="Final routing decision")
    generation: GenerationResult = Field(..., description="Final generated answer or refusal")
    total_latency_ms: float = Field(0.0, description="Total execution latency in milliseconds")
    planner_latency_ms: float = Field(0.0, description="Planner execution latency in milliseconds")
    retrieval_latency_ms: float = Field(0.0, description="Total retrieval latency across all passes in milliseconds")
    auditor_latency_ms: float = Field(0.0, description="Total auditor latency across all passes in milliseconds")
    generator_latency_ms: float = Field(0.0, description="Generator latency in milliseconds")
    num_llm_calls: int = Field(0, description="Total LLM completions requested across all components")
    token_usage: Dict[str, int] = Field(default_factory=dict, description="LLM prompt and completion token counts")
