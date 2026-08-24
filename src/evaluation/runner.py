"""
Benchmark Experiment Harness: Executes all 8 baseline and ablation configurations
across the manually verified benchmark dataset and exports structured evaluation results.

This module is ONLY invoked by scripts/run_evaluation_benchmark.py with the --real-api flag.
It must NEVER be automatically imported or executed by the pytest test suite.
"""

import os
import sys
import json
import time
import csv
from typing import List, Dict, Any, Optional, Tuple
from src.config import CHUNKS_JSON_PATH, MAX_EXPERIMENT_COST_USD
from src.models.document import Chunk
from src.retrieval.hybrid_retriever import HybridRetriever
from src.evaluation.benchmark_schema import BenchmarkQuestion, QuestionEvaluationRecord
from src.evaluation.metrics import evaluate_question_trace, compute_aggregate_metrics, SystemAggregateMetrics

from src.baselines.naive_rag import NaiveRAG
from src.baselines.hybrid_rag import HybridRAG
from src.agent.orchestrator import AgentOrchestrator
from src.agent.ablations import (
    AblationNoPlanner,
    AblationDenseOnlyAgentic,
    AblationNoSufficiencyChecker,
    AblationNoRetryLoop,
    AblationNoDecomposition,
)
from src.utils.cost_tracker import get_cost_tracker, BudgetExceededError

BENCHMARK_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "benchmark_dataset.json")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "evaluation_results")

# Systems that use the LLM planner (needed for dry-run call estimation)
_SYSTEMS_WITH_PLANNER = {
    "Baseline_B_HybridRAG",
    "Ablation_2_DenseOnlyAgentic",
    "Ablation_3_NoSufficiencyChecker",
    "Ablation_4_NoRetryLoop",
    "Ablation_5_NoDecomposition",
    "Agentic_System_C_Full",
}
# Systems that use the Evidence Auditor
_SYSTEMS_WITH_AUDITOR = {
    "Ablation_1_NoPlanner",
    "Ablation_2_DenseOnlyAgentic",
    "Ablation_4_NoRetryLoop",
    "Ablation_5_NoDecomposition",
    "Agentic_System_C_Full",
}


class ExperimentHarness:
    """Orchestrates benchmark evaluation runs across all frozen RAG systems."""

    def __init__(self, benchmark_path: str = BENCHMARK_PATH, chunks_path: str = CHUNKS_JSON_PATH):
        self.benchmark_path = benchmark_path
        self.chunks_path = chunks_path
        os.makedirs(RESULTS_DIR, exist_ok=True)

        with open(self.benchmark_path, "r", encoding="utf-8") as f:
            self.benchmark_questions = [BenchmarkQuestion(**item) for item in json.load(f)]

        with open(self.chunks_path, "r", encoding="utf-8") as f:
            chunks = [Chunk(**item) for item in json.load(f)]

        self.retriever = HybridRetriever.from_chunks(chunks)

        # Initialize all 8 fixed configurations
        self.systems = {
            "Baseline_A_NaiveDense": NaiveRAG(retriever=self.retriever, top_k=5),
            "Baseline_B_HybridRAG": HybridRAG(retriever=self.retriever, top_k=5),
            "Ablation_1_NoPlanner": AblationNoPlanner(retriever=self.retriever, top_k=5),
            "Ablation_2_DenseOnlyAgentic": AblationDenseOnlyAgentic(retriever=self.retriever, top_k=5),
            "Ablation_3_NoSufficiencyChecker": AblationNoSufficiencyChecker(retriever=self.retriever, top_k=5),
            "Ablation_4_NoRetryLoop": AblationNoRetryLoop(retriever=self.retriever, top_k=5),
            "Ablation_5_NoDecomposition": AblationNoDecomposition(retriever=self.retriever, top_k=5),
            "Agentic_System_C_Full": AgentOrchestrator(retriever=self.retriever),
        }

    def run_benchmark(
        self,
        split: str = "dev",
        system_names: Optional[List[str]] = None,
        max_cost_usd: float = MAX_EXPERIMENT_COST_USD,
        smoke_limit: Optional[int] = None,
    ) -> Tuple[List[QuestionEvaluationRecord], List[SystemAggregateMetrics]]:
        """
        Executes benchmark evaluation across selected systems and dataset split.

        Args:
            split: 'dev', 'test', or 'all'
            system_names: optional list of system names to run (default: all 8)
            max_cost_usd: hard budget limit — raises BudgetExceededError if exceeded mid-run
            smoke_limit: if set, only run this many questions total (for smoke tests)
        """
        from src.utils.cost_tracker import reset_cost_tracker
        reset_cost_tracker()

        selected_questions = [q for q in self.benchmark_questions if split == "all" or q.split == split]
        if smoke_limit:
            selected_questions = selected_questions[:smoke_limit]

        target_systems = system_names or list(self.systems.keys())

        all_records: List[QuestionEvaluationRecord] = []
        all_aggregates: List[SystemAggregateMetrics] = []
        failure_cases: List[Dict[str, Any]] = []

        print(f"\n{'=' * 80}")
        print(f"BENCHMARK EVALUATION  Split: '{split}'  Questions: {len(selected_questions)}  Systems: {len(target_systems)}")
        print(f"Budget limit: ${max_cost_usd:.2f}")
        print(f"{'=' * 80}")

        budget_exceeded = False

        try:
            for sys_name in target_systems:
                sys_instance = self.systems[sys_name]
                print(f"\nEvaluating: {sys_name} ...")
                sys_records: List[QuestionEvaluationRecord] = []

                for i, q in enumerate(selected_questions, start=1):
                    # ---- Budget check BEFORE each pipeline call ----
                    current_cost = get_cost_tracker().total_cost
                    if current_cost > max_cost_usd:
                        budget_exceeded = True
                        print(
                            f"\n⛔  BUDGET EXCEEDED: accumulated ${current_cost:.4f} > limit ${max_cost_usd:.2f}. "
                            f"Stopping after {len(all_records)} records."
                        )
                        raise BudgetExceededError(
                            f"Accumulated cost ${current_cost:.4f} exceeded budget ${max_cost_usd:.2f}"
                        )

                    t0 = time.perf_counter()
                    trace = sys_instance.run(q.question)
                    rec = evaluate_question_trace(question=q, trace=trace, system_name=sys_name)
                    sys_records.append(rec)
                    all_records.append(rec)

                    # Record candidate failure
                    if rec.candidate_failure_category or rec.correctness_score == 0.0:
                        failure_cases.append({
                            "question_id": q.question_id,
                            "system_name": sys_name,
                            "question": q.question,
                            "category": q.category,
                            "failure_category": rec.candidate_failure_category or "incorrect",
                            "response_text": rec.response_text,
                            "trace_id": trace.trace_id
                        })

                    status_flag = "✓" if rec.correctness_score == 1.0 or rec.is_true_refusal else (
                        "~" if rec.correctness_score == 0.5 else "✗"
                    )
                    print(
                        f"  [{i:02d}/{len(selected_questions):02d}] {q.question_id} "
                        f"({q.category[:15]}): {status_flag} {rec.correctness_label} | "
                        f"Latency: {rec.total_latency_ms:.0f}ms | Cost so far: ${get_cost_tracker().total_cost:.4f}"
                    )

                # Compute aggregate summary for this system
                if sys_records:
                    agg = compute_aggregate_metrics(sys_records, system_name=sys_name, split=split)
                    all_aggregates.append(agg)

        except BudgetExceededError:
            budget_exceeded = True

        finally:
            # Always save partial results — even on early termination
            self._export_results(
                all_records, all_aggregates, failure_cases, split, target_systems,
                len(selected_questions),
                is_partial=budget_exceeded,
            )

        if budget_exceeded:
            print(
                f"\n⚠  Experiment terminated early due to budget limit. "
                f"Partial results saved to {RESULTS_DIR}"
            )
            raise BudgetExceededError(
                f"Experiment stopped: budget ${max_cost_usd:.2f} exceeded."
            )

        return all_records, all_aggregates

    def _export_results(
        self,
        records: List[QuestionEvaluationRecord],
        aggregates: List[SystemAggregateMetrics],
        failures: List[Dict[str, Any]],
        split: str,
        systems: List[str],
        num_questions: int,
        is_partial: bool = False,
    ):
        """Exports JSON, CSV, Cost Breakdown, and Reproducibility Manifest.
        Safe to call on partial results (is_partial=True).
        """
        import hashlib
        from datetime import datetime, timezone
        from src.config import OPENAI_MODEL, OPENAI_EMBEDDING_MODEL

        suffix = f"_{split}_partial" if is_partial else f"_{split}"

        if not records:
            print("[No records to save]")
            return

        # 1. Question-level JSON
        q_path = os.path.join(RESULTS_DIR, f"question_level_results{suffix}.json")
        with open(q_path, "w", encoding="utf-8") as f:
            json.dump([r.model_dump() for r in records], f, indent=2)

        # 2. Aggregate Summary JSON
        agg_path = os.path.join(RESULTS_DIR, f"aggregate_summary{suffix}.json")
        with open(agg_path, "w", encoding="utf-8") as f:
            json.dump([a.model_dump() for a in aggregates], f, indent=2)

        # 3. Aggregate CSV
        csv_path = os.path.join(RESULTS_DIR, f"aggregate_summary{suffix}.csv")
        if aggregates:
            keys = list(aggregates[0].model_dump().keys())
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                for a in aggregates:
                    writer.writerow(a.model_dump())

        # 4. Failures JSON
        fail_path = os.path.join(RESULTS_DIR, f"failure_cases{suffix}.json")
        with open(fail_path, "w", encoding="utf-8") as f:
            json.dump(failures, f, indent=2)

        # 5. Cost Breakdown JSON
        cost_path = os.path.join(RESULTS_DIR, f"cost_breakdown{suffix}.json")
        with open(cost_path, "w", encoding="utf-8") as f:
            json.dump(get_cost_tracker().get_summary_dict(), f, indent=2)

        # 6. Reproducibility Manifest JSON
        dataset_content = open(BENCHMARK_PATH, "rb").read()
        dataset_hash = hashlib.sha256(dataset_content).hexdigest()
        manifest = {
            "evaluation_version": "v1.0_final",
            "is_partial_result": is_partial,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "split": split,
            "num_questions_selected": num_questions,
            "num_records_completed": len(records),
            "dataset_path": BENCHMARK_PATH,
            "dataset_sha256": dataset_hash,
            "systems_evaluated": systems,
            "models": {
                "llm_model": OPENAI_MODEL,
                "embedding_model": OPENAI_EMBEDDING_MODEL,
                "embedding_dimensions": 1536,
            },
            "corpus": {
                "papers_count": 10,
                "chunks_count": 243,
            },
            "platform": {
                "os": sys.platform,
                "python_version": sys.version,
            }
        }
        manifest_path = os.path.join(RESULTS_DIR, f"reproducibility_manifest{suffix}.json")
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        label = "[PARTIAL] " if is_partial else ""
        print(f"\n[{label}Saved benchmark artifacts to {RESULTS_DIR}]")


def compute_dry_run_estimate(
    harness: ExperimentHarness,
    split: str = "test",
    system_names: Optional[List[str]] = None,
    smoke_limit: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Estimates the number of API calls and cost without making any calls.

    Estimation model:
    - Planner: 1 call per run for systems that use it (avg 600 prompt + 150 completion tokens)
    - Auditor: 1.8 calls per run on average (some multi-pass) for systems that use it
              (avg 2,800 prompt + 400 completion tokens per call)
    - Generator: 1 call per run for all systems
              (avg 800 prompt + 300 completion tokens)
    - Query embeddings: ~0.7 of runs are cache misses (avg 20 tokens per query)
    """
    from src.config import (
        GPT4O_MINI_PROMPT_COST_PER_1M,
        GPT4O_MINI_COMPLETION_COST_PER_1M,
        EMBEDDING_COST_PER_1M,
    )

    questions = [q for q in harness.benchmark_questions if split == "all" or q.split == split]
    if smoke_limit:
        questions = questions[:smoke_limit]

    target_systems = system_names or list(harness.systems.keys())
    num_questions = len(questions)
    num_systems = len(target_systems)
    total_runs = num_questions * num_systems

    # Planner calls: only for systems with planner
    systems_with_planner = [s for s in target_systems if s in _SYSTEMS_WITH_PLANNER]
    est_planner_calls = len(systems_with_planner) * num_questions

    # Auditor calls: 1.8× average for systems with auditor
    systems_with_auditor = [s for s in target_systems if s in _SYSTEMS_WITH_AUDITOR]
    est_auditor_calls = int(len(systems_with_auditor) * num_questions * 1.8)

    # Generator calls: 1 per run for all systems
    est_generator_calls = total_runs

    # Query embedding calls: ~0.7 cache miss rate × total_runs (1 query per run)
    est_embedding_calls = int(total_runs * 0.7)

    # Token estimates
    planner_tokens = est_planner_calls * (600 + 150)
    auditor_tokens = est_auditor_calls * (2800 + 400)
    generator_tokens = est_generator_calls * (800 + 300)
    embedding_tokens = est_embedding_calls * 20

    total_llm_tokens = planner_tokens + auditor_tokens + generator_tokens

    # Cost estimates (simplified: all LLM costs use gpt-4o-mini average)
    # Weighted average: prompt is ~70% of tokens
    avg_prompt_fraction = 0.70
    llm_cost = (
        (total_llm_tokens * avg_prompt_fraction / 1_000_000) * GPT4O_MINI_PROMPT_COST_PER_1M
        + (total_llm_tokens * (1 - avg_prompt_fraction) / 1_000_000) * GPT4O_MINI_COMPLETION_COST_PER_1M
    )
    embedding_cost = (embedding_tokens / 1_000_000) * EMBEDDING_COST_PER_1M
    est_total_cost = llm_cost + embedding_cost

    return {
        "num_questions": num_questions,
        "num_systems": num_systems,
        "total_runs": total_runs,
        "est_planner_calls": est_planner_calls,
        "est_auditor_calls": est_auditor_calls,
        "est_generator_calls": est_generator_calls,
        "est_embedding_calls": est_embedding_calls,
        "est_total_tokens": total_llm_tokens + embedding_tokens,
        "est_cost_usd": round(est_total_cost, 4),
    }


def print_summary_table(aggregates: List[SystemAggregateMetrics]):
    """Prints a clean, formatted Markdown table of aggregate metrics with exact fractions."""
    print("\n" + "=" * 140)
    print("BENCHMARK EXPERIMENTAL COMPARISON MATRIX (Exact Counts & Multi-View Metrics)")
    print("=" * 140)
    header = (
        f"{'System':<30} | {'Pass1 R@5':<9} | {'Pool Rec':<9} | "
        f"{'Accuracy (Score/N)':<20} | {'Hallucination':<16} | {'False Refusal':<16} | "
        f"{'Debunk':<12} | {'Avg Latency':<11} | {'LLM Calls':<9}"
    )
    print(header)
    print("-" * len(header))
    for a in aggregates:
        row = (
            f"{a.system_name:<30} | {a.pass1_chunk_recall_at_5:>8.1%} | {a.cumulative_pool_chunk_recall:>8.1%} | "
            f"{a.accuracy_fraction:<20} | {a.hallucination_fraction:<16} | {a.false_refusal_fraction:<16} | "
            f"{a.debunk_fraction:<12} | {a.avg_latency_ms:>9.1f}ms | {a.avg_llm_calls:>8.1f}"
        )
        print(row)
    print("=" * 140)

    # Print Cost & Token Usage Breakdown
    print("\nBENCHMARK API USAGE, TOKEN CONSUMPTION & COST BREAKDOWN")
    print(get_cost_tracker().get_summary_table())
    print("\n")
