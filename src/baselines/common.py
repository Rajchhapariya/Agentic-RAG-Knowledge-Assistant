"""
Shared prompt templates and standard result wrapper for baseline RAG systems.
"""

import time
import uuid
from typing import List, Dict, Any, Optional
from openai import OpenAI
from src.config import (
    OPENAI_API_KEY,
    OPENAI_MODEL,
    OPENAI_API_ENABLED,
    CACHE_ONLY_MODE,
)
from src.models.retrieval import SearchResult
from src.models.trace import (
    AgentTrace,
    QueryPlan,
    PassRecord,
    AuditResult,
    GenerationResult,
    CitationItem,
)
from src.utils.cost_tracker import get_cost_tracker, OpenAIAPIDisabledError


class BaselineGenerator:
    """Generates standard single-pass RAG completions without evidence gating or refusal checks."""

    STANDARD_RAG_PROMPT = """You are an AI assistant answering questions based on the provided research paper passages.
Answer the user's question directly using the provided context passages. Do your best to provide a helpful answer."""

    def __init__(self, api_key: Optional[str] = None, model: str = OPENAI_MODEL):
        self.api_key = api_key or OPENAI_API_KEY
        self.client = OpenAI(api_key=self.api_key)
        self.model = model

    def generate(self, query: str, retrieved_chunks: List[SearchResult]) -> GenerationResult:
        """Generates a standard single-pass completion over retrieved passages."""
        context_str = "\n\n".join([
            f"--- Passage {i+1} [{r.doc_id}: {r.section_title}, Chunk {r.chunk_id}] ---\n{r.content}"
            for i, r in enumerate(retrieved_chunks)
        ])

        user_message = f"Context Passages:\n{context_str}\n\nUser Question: {query}\n\nAnswer:"

        if not OPENAI_API_ENABLED or CACHE_ONLY_MODE:
            raise OpenAIAPIDisabledError(
                f"OpenAI API is disabled (OPENAI_API_ENABLED={OPENAI_API_ENABLED}, CACHE_ONLY_MODE={CACHE_ONLY_MODE}) "
                f"cannot execute BaselineGenerator."
            )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.STANDARD_RAG_PROMPT},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.0
            )

            # Track usage
            if hasattr(response, "usage") and response.usage:
                get_cost_tracker().track_llm(
                    "baseline_generator",
                    prompt_tokens=response.usage.prompt_tokens,
                    completion_tokens=response.usage.completion_tokens
                )

            response_text = response.choices[0].message.content or ""
        except OpenAIAPIDisabledError:
            raise
        except Exception as e:
            response_text = f"Baseline generation error: {e}"

        # Standard RAG does not verify citations, but we record chunks present in prompt
        citations = [
            CitationItem(
                citation_id=i+1,
                doc_id=r.doc_id,
                document_title=r.document_title,
                section_title=r.section_title,
                chunk_id=r.chunk_id,
                exact_quote="",
                claim_text=r.section_title
            )
            for i, r in enumerate(retrieved_chunks[:2])
        ]

        return GenerationResult(
            status="ANSWERED",
            response_text=response_text,
            citations=citations,
            refusal_reason=None,
            has_conflict_acknowledged=False
        )


def build_baseline_trace(
    system_name: str,
    query: str,
    retrieved_chunks: List[SearchResult],
    gen_result: GenerationResult,
    latency_ms: float,
    retrieval_latency_ms: float = 0.0
) -> AgentTrace:
    """Wraps baseline execution into a standard AgentTrace object for evaluation compatibility."""
    dummy_plan = QueryPlan(
        query_type="FACTUAL_SINGLE_HOP",
        needs_retrieval=True,
        direct_response=None,
        target_concepts=[query],
        sub_questions=[query],
        initial_search_queries=[query],
        suggested_filters=None
    )

    dummy_audit = AuditResult(
        verdict="SUFFICIENT",  # Baseline implicitly assumes retrieved chunks are always sufficient
        subquestion_coverage={0: True},
        evidence_relationships=[],
        verified_supported_spans=[],
        missing_information=[],
        diagnosed_search_gap=None
    )

    pass_rec = PassRecord(
        pass_number=1,
        search_queries=[query],
        retrieved_chunk_ids=[r.chunk_id for r in retrieved_chunks],
        retrieved_results=retrieved_chunks,
        audit_result=dummy_audit,
        reformulated_query=None
    )

    gen_ms = max(0.0, latency_ms - retrieval_latency_ms)

    return AgentTrace(
        trace_id=f"{system_name.lower()}_{uuid.uuid4().hex[:8]}",
        query=query,
        planner=dummy_plan,
        passes=[pass_rec],
        retry_count=0,
        final_decision="SUFFICIENT",
        generation=gen_result,
        total_latency_ms=round(latency_ms, 2),
        planner_latency_ms=0.0,
        retrieval_latency_ms=round(retrieval_latency_ms, 2),
        auditor_latency_ms=0.0,
        generator_latency_ms=round(gen_ms, 2),
        num_llm_calls=1
    )
