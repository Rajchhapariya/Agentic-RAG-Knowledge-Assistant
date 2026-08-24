"""
Ablation Pipelines: Isolated component removals for experimental evaluation.
1. Ablation 1: No Planner
2. Ablation 2: No BM25 (Dense-Only Agentic)
3. Ablation 3: No Sufficiency Checker
4. Ablation 4: No Retry Loop (Single-Pass Auditor)
5. Ablation 5: No Multi-Hop Decomposition
"""

import time
import uuid
from typing import List, Optional, Dict, Any
from src.models.retrieval import SearchResult
from src.models.trace import (
    AgentTrace,
    QueryPlan,
    PassRecord,
    AuditResult,
    GenerationResult,
    CitationItem,
    EvidenceRelationship,
)
from src.retrieval.hybrid_retriever import HybridRetriever
from src.agent.planner import QueryPlanner
from src.agent.evidence_auditor import EvidenceAuditor
from src.agent.grounded_generator import GroundedGenerator
from src.agent.reformulator import QueryReformulator


class AblationNoPlanner:
    """Ablation 1: Removes Query Planner (uses raw user query directly for retrieval and auditing)."""

    def __init__(self, retriever: HybridRetriever, top_k: int = 5, max_retries: int = 2):
        self.retriever = retriever
        self.top_k = top_k
        self.max_retries = max_retries
        self.auditor = EvidenceAuditor(max_retries=max_retries)
        self.generator = GroundedGenerator()

    def run(self, query: str) -> AgentTrace:
        start_time = time.perf_counter()
        trace_id = f"abl1_{uuid.uuid4().hex[:8]}"

        # No planner: single monolithic sub-question and search query
        plan = QueryPlan(
            query_type="FACTUAL_SINGLE_HOP",
            needs_retrieval=True,
            target_concepts=[query],
            sub_questions=[query],
            initial_search_queries=[query]
        )

        passes_history: List[PassRecord] = []
        current_search_queries = [query]
        retry_count = 0
        final_audit_result = None

        for pass_num in range(1, self.max_retries + 2):
            pass_retrieved: List[SearchResult] = []
            for sq in current_search_queries:
                hits = self.retriever.retrieve(query=sq, top_k=self.top_k, mode="hybrid")
                pass_retrieved.extend(hits)

            # Deduplicate hits
            seen = set()
            unique_hits = []
            for h in pass_retrieved:
                if h.chunk_id not in seen:
                    seen.add(h.chunk_id)
                    unique_hits.append(h)

            audit_res = self.auditor.audit(
                sub_questions=[query],
                retrieved_results=unique_hits,
                retry_count=retry_count
            )
            final_audit_result = audit_res

            if audit_res.verdict == "SUFFICIENT" or audit_res.verdict == "DEFINITIVELY_ABSENT":
                passes_history.append(PassRecord(
                    pass_number=pass_num,
                    search_queries=current_search_queries,
                    retrieved_chunk_ids=[r.chunk_id for r in unique_hits],
                    retrieved_results=unique_hits,
                    audit_result=audit_res,
                    reformulated_query=None
                ))
                break
            elif retry_count < self.max_retries:
                gap_query = audit_res.diagnosed_search_gap or query
                previous_searches = [sq for p in passes_history for sq in p.search_queries] + current_search_queries
                reformulated_q = QueryReformulator.reformulate(
                    original_query=query,
                    diagnosed_gap=gap_query,
                    missing_information=audit_res.missing_information,
                    previous_queries=previous_searches,
                    target_concepts=[query]
                )
                passes_history.append(PassRecord(
                    pass_number=pass_num,
                    search_queries=current_search_queries,
                    retrieved_chunk_ids=[r.chunk_id for r in unique_hits],
                    retrieved_results=unique_hits,
                    audit_result=audit_res,
                    reformulated_query=reformulated_q
                ))
                current_search_queries = [reformulated_q]
                retry_count += 1
            else:
                break

        final_decision = "SUFFICIENT" if (final_audit_result and final_audit_result.verdict == "SUFFICIENT") else "REFUSE"
        gen_res = self.generator.generate(query=query, audit_result=final_audit_result, decision=final_decision)
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        return AgentTrace(
            trace_id=trace_id,
            query=query,
            planner=plan,
            passes=passes_history,
            retry_count=retry_count,
            final_decision=final_decision,
            generation=gen_res,
            total_latency_ms=round(elapsed_ms, 2)
        )


class AblationDenseOnlyAgentic:
    """Ablation 2: Pure Dense Retrieval in Agentic Loop (disables BM25 / RRF)."""

    def __init__(self, retriever: HybridRetriever, top_k: int = 5, max_retries: int = 2):
        self.retriever = retriever
        self.top_k = top_k
        self.max_retries = max_retries
        self.planner = QueryPlanner()
        self.auditor = EvidenceAuditor(max_retries=max_retries)
        self.generator = GroundedGenerator()

    def run(self, query: str) -> AgentTrace:
        start_time = time.perf_counter()
        trace_id = f"abl2_{uuid.uuid4().hex[:8]}"

        plan = self.planner.plan(query)
        if not plan.needs_retrieval:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return AgentTrace(
                trace_id=trace_id,
                query=query,
                planner=plan,
                passes=[],
                retry_count=0,
                final_decision="DIRECT_ANSWER",
                generation=GenerationResult(status="ANSWERED", response_text=plan.direct_response or ""),
                total_latency_ms=round(elapsed_ms, 2)
            )

        passes_history: List[PassRecord] = []
        current_search_queries = list(plan.initial_search_queries)
        retry_count = 0
        final_audit_result = None

        for pass_num in range(1, self.max_retries + 2):
            pass_retrieved: List[SearchResult] = []
            for sq in current_search_queries:
                # Dense-only retrieval mode
                hits = self.retriever.retrieve(query=sq, top_k=self.top_k, mode="dense")
                pass_retrieved.extend(hits)

            seen = set()
            unique_hits = []
            for h in pass_retrieved:
                if h.chunk_id not in seen:
                    seen.add(h.chunk_id)
                    unique_hits.append(h)

            audit_res = self.auditor.audit(
                sub_questions=plan.sub_questions,
                retrieved_results=unique_hits,
                retry_count=retry_count
            )
            final_audit_result = audit_res

            if audit_res.verdict == "SUFFICIENT" or audit_res.verdict == "DEFINITIVELY_ABSENT":
                passes_history.append(PassRecord(
                    pass_number=pass_num,
                    search_queries=current_search_queries,
                    retrieved_chunk_ids=[r.chunk_id for r in unique_hits],
                    retrieved_results=unique_hits,
                    audit_result=audit_res,
                    reformulated_query=None
                ))
                break
            elif retry_count < self.max_retries:
                gap_query = audit_res.diagnosed_search_gap or query
                previous_searches = [sq for p in passes_history for sq in p.search_queries] + current_search_queries
                reformulated_q = QueryReformulator.reformulate(
                    original_query=query,
                    diagnosed_gap=gap_query,
                    missing_information=audit_res.missing_information,
                    previous_queries=previous_searches,
                    target_concepts=plan.target_concepts
                )
                passes_history.append(PassRecord(
                    pass_number=pass_num,
                    search_queries=current_search_queries,
                    retrieved_chunk_ids=[r.chunk_id for r in unique_hits],
                    retrieved_results=unique_hits,
                    audit_result=audit_res,
                    reformulated_query=reformulated_q
                ))
                current_search_queries = [reformulated_q]
                retry_count += 1
            else:
                break

        final_decision = "SUFFICIENT" if (final_audit_result and final_audit_result.verdict == "SUFFICIENT") else "REFUSE"
        gen_res = self.generator.generate(query=query, audit_result=final_audit_result, decision=final_decision)
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        return AgentTrace(
            trace_id=trace_id,
            query=query,
            planner=plan,
            passes=passes_history,
            retry_count=retry_count,
            final_decision=final_decision,
            generation=gen_res,
            total_latency_ms=round(elapsed_ms, 2)
        )


class AblationNoSufficiencyChecker:
    """Ablation 3: Removes Sufficiency Checker (generates directly on top-k chunks without quote validation or refusal checks)."""

    def __init__(self, retriever: HybridRetriever, top_k: int = 5):
        self.retriever = retriever
        self.top_k = top_k
        self.planner = QueryPlanner()
        from src.baselines.common import BaselineGenerator
        self.generator = BaselineGenerator()

    def run(self, query: str) -> AgentTrace:
        start_time = time.perf_counter()
        trace_id = f"abl3_{uuid.uuid4().hex[:8]}"

        plan = self.planner.plan(query)
        retrieved_results: List[SearchResult] = []
        for sq in plan.initial_search_queries:
            hits = self.retriever.retrieve(query=sq, top_k=self.top_k, mode="hybrid")
            retrieved_results.extend(hits)

        seen = set()
        unique_hits = []
        for h in retrieved_results:
            if h.chunk_id not in seen:
                seen.add(h.chunk_id)
                unique_hits.append(h)

        # Bypass Evidence Auditor entirely
        dummy_audit = AuditResult(
            verdict="SUFFICIENT",
            subquestion_coverage={i: True for i in range(len(plan.sub_questions))},
            evidence_relationships=[],
            verified_supported_spans=[],
            missing_information=[],
            diagnosed_search_gap=None
        )

        pass_rec = PassRecord(
            pass_number=1,
            search_queries=plan.initial_search_queries,
            retrieved_chunk_ids=[r.chunk_id for r in unique_hits],
            retrieved_results=unique_hits,
            audit_result=dummy_audit,
            reformulated_query=None
        )

        gen_res = self.generator.generate(query=query, retrieved_chunks=unique_hits)
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        return AgentTrace(
            trace_id=trace_id,
            query=query,
            planner=plan,
            passes=[pass_rec],
            retry_count=0,
            final_decision="SUFFICIENT",
            generation=gen_res,
            total_latency_ms=round(elapsed_ms, 2)
        )


class AblationNoRetryLoop:
    """Ablation 4: Single-Pass Evidence Auditor (max_retries = 0, no query reformulation or re-retrieval)."""

    def __init__(self, retriever: HybridRetriever, top_k: int = 5):
        self.retriever = retriever
        self.top_k = top_k
        self.planner = QueryPlanner()
        self.auditor = EvidenceAuditor(max_retries=0)
        self.generator = GroundedGenerator()

    def run(self, query: str) -> AgentTrace:
        start_time = time.perf_counter()
        trace_id = f"abl4_{uuid.uuid4().hex[:8]}"

        plan = self.planner.plan(query)
        if not plan.needs_retrieval:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return AgentTrace(
                trace_id=trace_id,
                query=query,
                planner=plan,
                passes=[],
                retry_count=0,
                final_decision="DIRECT_ANSWER",
                generation=GenerationResult(status="ANSWERED", response_text=plan.direct_response or ""),
                total_latency_ms=round(elapsed_ms, 2)
            )

        retrieved_results: List[SearchResult] = []
        for sq in plan.initial_search_queries:
            hits = self.retriever.retrieve(query=sq, top_k=self.top_k, mode="hybrid")
            retrieved_results.extend(hits)

        seen = set()
        unique_hits = []
        for h in retrieved_results:
            if h.chunk_id not in seen:
                seen.add(h.chunk_id)
                unique_hits.append(h)

        # Audit with retry_count = 0 and max_retries = 0 (single pass)
        audit_res = self.auditor.audit(
            sub_questions=plan.sub_questions,
            retrieved_results=unique_hits,
            retry_count=0
        )

        pass_rec = PassRecord(
            pass_number=1,
            search_queries=plan.initial_search_queries,
            retrieved_chunk_ids=[r.chunk_id for r in unique_hits],
            retrieved_results=unique_hits,
            audit_result=audit_res,
            reformulated_query=None
        )

        final_decision = "SUFFICIENT" if audit_res.verdict == "SUFFICIENT" else "REFUSE"
        gen_res = self.generator.generate(query=query, audit_result=audit_res, decision=final_decision)
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        return AgentTrace(
            trace_id=trace_id,
            query=query,
            planner=plan,
            passes=[pass_rec],
            retry_count=0,
            final_decision=final_decision,
            generation=gen_res,
            total_latency_ms=round(elapsed_ms, 2)
        )


class AblationNoDecomposition:
    """Ablation 5: Removes Multi-Hop Decomposition (forces monolithic search query for all questions)."""

    def __init__(self, retriever: HybridRetriever, top_k: int = 5, max_retries: int = 2):
        self.retriever = retriever
        self.top_k = top_k
        self.max_retries = max_retries
        self.planner = QueryPlanner()
        self.auditor = EvidenceAuditor(max_retries=max_retries)
        self.generator = GroundedGenerator()

    def run(self, query: str) -> AgentTrace:
        start_time = time.perf_counter()
        trace_id = f"abl5_{uuid.uuid4().hex[:8]}"

        plan = self.planner.plan(query)
        if not plan.needs_retrieval:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return AgentTrace(
                trace_id=trace_id,
                query=query,
                planner=plan,
                passes=[],
                retry_count=0,
                final_decision="DIRECT_ANSWER",
                generation=GenerationResult(status="ANSWERED", response_text=plan.direct_response or ""),
                total_latency_ms=round(elapsed_ms, 2)
            )

        # Force single monolithic sub-question regardless of query type
        monolithic_sub_qs = [query]
        monolithic_search = [plan.initial_search_queries[0] if plan.initial_search_queries else query]

        passes_history: List[PassRecord] = []
        current_search_queries = list(monolithic_search)
        retry_count = 0
        final_audit_result = None

        for pass_num in range(1, self.max_retries + 2):
            pass_retrieved: List[SearchResult] = []
            for sq in current_search_queries:
                hits = self.retriever.retrieve(query=sq, top_k=self.top_k, mode="hybrid")
                pass_retrieved.extend(hits)

            seen = set()
            unique_hits = []
            for h in pass_retrieved:
                if h.chunk_id not in seen:
                    seen.add(h.chunk_id)
                    unique_hits.append(h)

            audit_res = self.auditor.audit(
                sub_questions=monolithic_sub_qs,
                retrieved_results=unique_hits,
                retry_count=retry_count
            )
            final_audit_result = audit_res

            if audit_res.verdict == "SUFFICIENT" or audit_res.verdict == "DEFINITIVELY_ABSENT":
                passes_history.append(PassRecord(
                    pass_number=pass_num,
                    search_queries=current_search_queries,
                    retrieved_chunk_ids=[r.chunk_id for r in unique_hits],
                    retrieved_results=unique_hits,
                    audit_result=audit_res,
                    reformulated_query=None
                ))
                break
            elif retry_count < self.max_retries:
                gap_query = audit_res.diagnosed_search_gap or query
                previous_searches = [sq for p in passes_history for sq in p.search_queries] + current_search_queries
                reformulated_q = QueryReformulator.reformulate(
                    original_query=query,
                    diagnosed_gap=gap_query,
                    missing_information=audit_res.missing_information,
                    previous_queries=previous_searches,
                    target_concepts=plan.target_concepts
                )
                passes_history.append(PassRecord(
                    pass_number=pass_num,
                    search_queries=current_search_queries,
                    retrieved_chunk_ids=[r.chunk_id for r in unique_hits],
                    retrieved_results=unique_hits,
                    audit_result=audit_res,
                    reformulated_query=reformulated_q
                ))
                current_search_queries = [reformulated_q]
                retry_count += 1
            else:
                break

        final_decision = "SUFFICIENT" if (final_audit_result and final_audit_result.verdict == "SUFFICIENT") else "REFUSE"
        gen_res = self.generator.generate(query=query, audit_result=final_audit_result, decision=final_decision)
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        return AgentTrace(
            trace_id=trace_id,
            query=query,
            planner=plan,
            passes=passes_history,
            retry_count=retry_count,
            final_decision=final_decision,
            generation=gen_res,
            total_latency_ms=round(elapsed_ms, 2)
        )
