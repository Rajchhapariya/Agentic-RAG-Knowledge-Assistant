"""
Agent Orchestrator: Coordinates the full agentic state machine:
PLAN -> MULTI-HOP RETRIEVE -> EVIDENCE AUDIT -> BOUNDED RETRY -> GROUNDED GENERATE / REFUSE.
"""

import time
import uuid
from typing import Optional, List, Dict, Any
from src.config import MAX_RETRIES, DEFAULT_TOP_K
from src.models.retrieval import SearchResult
from src.models.trace import (
    AgentTrace,
    QueryPlan,
    PassRecord,
    AuditResult,
    GenerationResult,
)
from src.retrieval.hybrid_retriever import HybridRetriever
from src.agent.planner import QueryPlanner
from src.agent.evidence_auditor import EvidenceAuditor
from src.agent.grounded_generator import GroundedGenerator


class AgentOrchestrator:
    """
    Main state machine orchestrating the iterative Agentic RAG loop.
    Enforces a strict maximum of 2 retries, deterministic evidence verification,
    and full trace observability.
    """

    def __init__(
        self,
        retriever: HybridRetriever,
        planner: Optional[QueryPlanner] = None,
        auditor: Optional[EvidenceAuditor] = None,
        generator: Optional[GroundedGenerator] = None,
        max_retries: int = MAX_RETRIES,
        top_k: int = DEFAULT_TOP_K,
    ):
        self.retriever = retriever
        self.planner = planner or QueryPlanner()
        self.auditor = auditor or EvidenceAuditor(max_retries=max_retries)
        self.generator = generator or GroundedGenerator()
        self.max_retries = max_retries
        self.top_k = top_k

    def run(self, query: str, force_mode: str = "hybrid") -> AgentTrace:
        """
        Executes the end-to-end Agentic RAG pipeline for a user question.
        Returns a complete, inspectable AgentTrace object.
        """
        start_time = time.perf_counter()
        trace_id = f"tr_{uuid.uuid4().hex[:8]}"

        # =========================================================================
        # Stage 1: Query Planning & Decomposition
        # =========================================================================
        t_plan_start = time.perf_counter()
        plan: QueryPlan = self.planner.plan(query)
        planner_ms = (time.perf_counter() - t_plan_start) * 1000.0
        num_llm_calls = 1

        # Handle direct conversational queries immediately
        if not plan.needs_retrieval:
            t_gen_start = time.perf_counter()
            gen_res = self.generator.generate(
                query=query,
                audit_result=AuditResult(verdict="SUFFICIENT"),
                decision="DIRECT_ANSWER",
                direct_response=plan.direct_response
            )
            generator_ms = (time.perf_counter() - t_gen_start) * 1000.0
            num_llm_calls += 1
            total_latency = (time.perf_counter() - start_time) * 1000.0
            return AgentTrace(
                trace_id=trace_id,
                query=query,
                planner=plan,
                passes=[],
                retry_count=0,
                final_decision="DIRECT_ANSWER",
                generation=gen_res,
                total_latency_ms=round(total_latency, 2),
                planner_latency_ms=round(planner_ms, 2),
                retrieval_latency_ms=0.0,
                auditor_latency_ms=0.0,
                generator_latency_ms=round(generator_ms, 2),
                num_llm_calls=num_llm_calls
            )

        # =========================================================================
        # Stage 2 & 3: Bounded Retrieve-Audit Loop
        # =========================================================================
        passes_history: List[PassRecord] = []
        current_search_queries = list(plan.initial_search_queries)
        accumulated_chunks: Dict[str, SearchResult] = {}
        last_audit: Optional[AuditResult] = None
        final_decision = "REFUSE"
        retry_count = 0
        total_retrieval_ms = 0.0
        total_auditor_ms = 0.0

        while retry_count <= self.max_retries:
            pass_num = retry_count + 1
            pass_retrieved_results: List[SearchResult] = []
            pass_filter_fallback_event: Optional[str] = None

            # Execute retrieval for each search query (supporting multi-hop sub-queries)
            t_ret_start = time.perf_counter()
            for sq in current_search_queries:
                hits = self.retriever.retrieve(
                    query=sq,
                    top_k=self.top_k,
                    mode=force_mode,
                    filters=plan.suggested_filters
                )
                
                # Check for zero-result filtered retrieval and execute controlled fallback
                if not hits and plan.suggested_filters:
                    pass_filter_fallback_event = (
                        f"Suggested filter {plan.suggested_filters} returned 0 results for query '{sq}'. "
                        "Executing controlled fallback to unfiltered hybrid retrieval."
                    )
                    hits = self.retriever.retrieve(
                        query=sq,
                        top_k=self.top_k,
                        mode=force_mode,
                        filters=None
                    )

                for h in hits:
                    if h.chunk_id not in accumulated_chunks:
                        accumulated_chunks[h.chunk_id] = h
                    pass_retrieved_results.append(h)
            total_retrieval_ms += (time.perf_counter() - t_ret_start) * 1000.0

            # Audit accumulated evidence against all sub-questions
            t_aud_start = time.perf_counter()
            candidate_pool = list(accumulated_chunks.values())
            audit_res: AuditResult = self.auditor.audit(
                sub_questions=plan.sub_questions,
                retrieved_results=candidate_pool,
                retry_count=retry_count
            )
            total_auditor_ms += (time.perf_counter() - t_aud_start) * 1000.0
            num_llm_calls += 1
            last_audit = audit_res

            # Check audit verdict
            if audit_res.verdict == "SUFFICIENT":
                final_decision = "SUFFICIENT"
                passes_history.append(PassRecord(
                    pass_number=pass_num,
                    search_queries=current_search_queries,
                    retrieved_chunk_ids=[r.chunk_id for r in pass_retrieved_results],
                    retrieved_results=pass_retrieved_results,
                    audit_result=audit_res,
                    reformulated_query=None,
                    filter_fallback_event=pass_filter_fallback_event
                ))
                break

            elif audit_res.verdict == "DEBUNK_FALSE_PREMISE":
                final_decision = "DEBUNK_FALSE_PREMISE"
                passes_history.append(PassRecord(
                    pass_number=pass_num,
                    search_queries=current_search_queries,
                    retrieved_chunk_ids=[r.chunk_id for r in pass_retrieved_results],
                    retrieved_results=pass_retrieved_results,
                    audit_result=audit_res,
                    reformulated_query=None,
                    filter_fallback_event=pass_filter_fallback_event
                ))
                break

            elif audit_res.verdict == "CONTRADICTED":
                final_decision = "DEBUNK_FALSE_PREMISE" if audit_res.contradiction.is_false_premise else "SUFFICIENT"
                passes_history.append(PassRecord(
                    pass_number=pass_num,
                    search_queries=current_search_queries,
                    retrieved_chunk_ids=[r.chunk_id for r in pass_retrieved_results],
                    retrieved_results=pass_retrieved_results,
                    audit_result=audit_res,
                    reformulated_query=None,
                    filter_fallback_event=pass_filter_fallback_event
                ))
                break

            elif audit_res.verdict == "PARTIALLY_SUFFICIENT":
                final_decision = "PARTIALLY_SUFFICIENT"
                passes_history.append(PassRecord(
                    pass_number=pass_num,
                    search_queries=current_search_queries,
                    retrieved_chunk_ids=[r.chunk_id for r in pass_retrieved_results],
                    retrieved_results=pass_retrieved_results,
                    audit_result=audit_res,
                    reformulated_query=None,
                    filter_fallback_event=pass_filter_fallback_event
                ))
                break

            elif audit_res.verdict == "INSUFFICIENT_RETRY" and retry_count < self.max_retries:
                # Trigger retry: reformulate query targeting the diagnosed search gap using QueryReformulator
                from src.agent.reformulator import QueryReformulator
                
                gap_query = audit_res.diagnosed_search_gap or query
                previous_searches = [sq for p in passes_history for sq in p.search_queries] + current_search_queries
                
                reformulated_q = QueryReformulator.reformulate(
                    original_query=query,
                    diagnosed_gap=gap_query,
                    missing_information=audit_res.missing_information,
                    previous_queries=previous_searches,
                    target_concepts=plan.target_concepts
                )
                num_llm_calls += 1
                
                passes_history.append(PassRecord(
                    pass_number=pass_num,
                    search_queries=current_search_queries,
                    retrieved_chunk_ids=[r.chunk_id for r in pass_retrieved_results],
                    retrieved_results=pass_retrieved_results,
                    audit_result=audit_res,
                    reformulated_query=reformulated_q,
                    filter_fallback_event=pass_filter_fallback_event
                ))
                
                # Update queries for next iteration
                current_search_queries = [reformulated_q]
                retry_count += 1

            else:
                # Retries exhausted or definitive absence -> check partial sufficiency or REFUSE
                if audit_res.verdict == "PARTIALLY_SUFFICIENT":
                    final_decision = "PARTIALLY_SUFFICIENT"
                else:
                    final_decision = "REFUSE"

                passes_history.append(PassRecord(
                    pass_number=pass_num,
                    search_queries=current_search_queries,
                    retrieved_chunk_ids=[r.chunk_id for r in pass_retrieved_results],
                    retrieved_results=pass_retrieved_results,
                    audit_result=audit_res,
                    reformulated_query=None,
                    filter_fallback_event=pass_filter_fallback_event
                ))
                break

        # =========================================================================
        # Stage 4: Grounded Generation or Refusal
        # =========================================================================
        t_gen_start = time.perf_counter()
        gen_result: GenerationResult = self.generator.generate(
            query=query,
            audit_result=last_audit or AuditResult(verdict="DEFINITIVELY_ABSENT"),
            decision=final_decision
        )
        generator_ms = (time.perf_counter() - t_gen_start) * 1000.0
        num_llm_calls += 1

        total_latency = (time.perf_counter() - start_time) * 1000.0

        return AgentTrace(
            trace_id=trace_id,
            query=query,
            planner=plan,
            passes=passes_history,
            retry_count=retry_count,
            final_decision=final_decision,
            generation=gen_result,
            total_latency_ms=round(total_latency, 2),
            planner_latency_ms=round(planner_ms, 2),
            retrieval_latency_ms=round(total_retrieval_ms, 2),
            auditor_latency_ms=round(total_auditor_ms, 2),
            generator_latency_ms=round(generator_ms, 2),
            num_llm_calls=num_llm_calls
        )
