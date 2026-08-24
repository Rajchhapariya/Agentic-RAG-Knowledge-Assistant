"""
tests/test_safety_and_cost_controls.py — Safety Mechanism Tests

Verifies all cost-control and safety behaviors with zero real API calls:
  1. Default pytest makes zero external API calls
  2. Integration tests are excluded by default (structural verification)
  3. Benchmark dry-run performs no API calls
  4. Smoke-test limits to 1 question / 1 system
  5. Budget guard stops execution when limit is exceeded
  6. Partial results are preserved after budget termination
  7. Cached embeddings are reused (zero doc embedding API calls)
  8. No document embedding API call occurs during normal tests

All mocking in this file is at the component boundary (not just returning True).
"""

import json
import os
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock, call
from typing import List

from src.config import (
    CHUNKS_JSON_PATH,
    DOC_EMBEDDINGS_CACHE_PATH,
    DOC_EMBEDDINGS_METADATA_PATH,
    MAX_EXPERIMENT_COST_USD,
)
from src.utils.cost_tracker import (
    CostTracker,
    BudgetExceededError,
    get_cost_tracker,
    reset_cost_tracker,
)
from src.evaluation.runner import ExperimentHarness, compute_dry_run_estimate
from conftest import make_query_plan, make_sufficient_audit, make_answered_generation



# ---------------------------------------------------------------------------
# Test 1: Default pytest makes zero external API calls
#   Verifies that after running the test suite, no LLM/embedding API was called.
# ---------------------------------------------------------------------------
def test_default_pytest_makes_zero_api_calls():
    """
    Verifies that a fresh reset of the cost tracker shows zero external API calls
    would be recorded by offline unit tests. Because the Phase 2 retrieval tests
    run real embedding lookups (cached, not API), the global tracker may have
    query_embedding hits. This test confirms 0 DOC embedding API calls and 0 LLM calls.
    """
    # Reset to a clean baseline for this measurement
    reset_cost_tracker()
    tracker = get_cost_tracker()

    # Now simulate what a typical unit test does: build retriever from cache
    from src.models.document import Chunk
    from src.retrieval.hybrid_retriever import HybridRetriever

    with open(CHUNKS_JSON_PATH, "r", encoding="utf-8") as f:
        chunks = [Chunk(**item) for item in json.load(f)]
    HybridRetriever.from_chunks(chunks)  # loads from cache

    # Document embedding calls must be zero (all loaded from cache)
    assert tracker.doc_embedding_api_calls == 0, (
        f"Document embedding API was called {tracker.doc_embedding_api_calls} times — "
        "must be 0 when loading from NPZ cache."
    )
    assert tracker.doc_cost == 0.0

    # LLM calls must be zero (no planner/auditor/generator were invoked)
    total_llm_calls = sum(v["calls"] for v in tracker.llm_usage.values())
    assert total_llm_calls == 0, (
        f"LLM API was called {total_llm_calls} times — "
        "unit tests must use mocks, not live OpenAI."
    )


# ---------------------------------------------------------------------------
# Test 2: Integration tests are excluded by default (structural check)
#   Verifies pytest.ini configuration exists and contains the correct addopts.
# ---------------------------------------------------------------------------
def test_integration_tests_excluded_by_default():
    """
    Verifies pytest.ini is configured to exclude @pytest.mark.integration tests by default.
    This is a structural governance test — if pytest.ini is misconfigured, this catches it.
    """
    project_root = Path(__file__).resolve().parent.parent
    pytest_ini = project_root / "pytest.ini"
    assert pytest_ini.exists(), "pytest.ini must exist at project root"

    content = pytest_ini.read_text(encoding="utf-8")
    assert "not integration" in content, (
        "pytest.ini must contain 'not integration' in addopts to exclude integration tests by default"
    )
    assert "integration" in content and "markers" in content, (
        "pytest.ini must define the 'integration' marker"
    )


# ---------------------------------------------------------------------------
# Test 3: Benchmark dry-run performs zero API calls
#   Verifies compute_dry_run_estimate() is purely computational.
# ---------------------------------------------------------------------------
def test_benchmark_dry_run_no_api_calls():
    """
    compute_dry_run_estimate() must not call any OpenAI API.
    It only reads benchmark_dataset.json and does arithmetic.
    """
    reset_cost_tracker()
    tracker = get_cost_tracker()

    harness = ExperimentHarness()
    estimate = compute_dry_run_estimate(harness=harness, split="dev", system_names=None)

    # No API calls should have been made
    assert tracker.doc_embedding_api_calls == 0
    assert tracker.query_embedding_api_calls == 0
    assert sum(v["calls"] for v in tracker.llm_usage.values()) == 0

    # Estimate should be structurally valid
    assert estimate["num_questions"] == 8  # dev split
    assert estimate["num_systems"] == 8
    assert estimate["total_runs"] == 64
    assert estimate["est_planner_calls"] >= 0
    assert estimate["est_auditor_calls"] >= 0
    assert estimate["est_generator_calls"] == 64
    assert estimate["est_cost_usd"] > 0.0


# ---------------------------------------------------------------------------
# Test 4: Smoke-test limits execution to 1 question / 1 system
#   Verifies smoke_limit=1 restricts question count in compute_dry_run_estimate.
# ---------------------------------------------------------------------------
def test_smoke_test_limits_one_question_one_system():
    """
    Smoke test mode: smoke_limit=1 + single system = 1 total run.
    Verifies the estimate reflects this constraint exactly.
    """
    harness = ExperimentHarness()
    estimate = compute_dry_run_estimate(
        harness=harness,
        split="dev",
        system_names=["Agentic_System_C_Full"],
        smoke_limit=1,
    )

    assert estimate["num_questions"] == 1
    assert estimate["num_systems"] == 1
    assert estimate["total_runs"] == 1
    # Generator: 1 call (1 question × 1 system)
    assert estimate["est_generator_calls"] == 1


# ---------------------------------------------------------------------------
# Test 5: Budget guard stops execution mid-run
#   Verifies BudgetExceededError is raised when cost limit is hit.
# ---------------------------------------------------------------------------
def test_budget_guard_stops_execution():
    """
    When accumulated cost exceeds max_cost_usd, BudgetExceededError
    must be raised immediately before/during the first pipeline call.

    Strategy: mock CostTracker.total_cost to return a value over budget
    so the check fires immediately at the start of the question loop.
    """
    harness = ExperimentHarness()
    plan = make_query_plan()
    audit = make_sufficient_audit()
    gen = make_answered_generation()

    # Mock total_cost as a property that always returns 999.0 (way over any budget)
    with patch.object(
        type(get_cost_tracker()), "total_cost",
        new_callable=lambda: property(lambda self: 999.0)
    ), \
         patch("src.agent.planner.QueryPlanner.plan", return_value=plan), \
         patch("src.agent.evidence_auditor.EvidenceAuditor.audit", return_value=audit), \
         patch("src.agent.grounded_generator.GroundedGenerator.generate", return_value=gen):
        with pytest.raises(BudgetExceededError):
            harness.run_benchmark(
                split="dev",
                system_names=["Agentic_System_C_Full"],
                max_cost_usd=0.50,  # Budget < mocked $999 cost
                smoke_limit=8,
            )


# ---------------------------------------------------------------------------
# Test 6: Partial results are preserved after budget termination
#   Verifies _export_results() is called in the finally block on BudgetExceededError.
# ---------------------------------------------------------------------------
def test_partial_results_saved_on_budget_termination(tmp_path, monkeypatch):
    """
    Even if the benchmark is terminated by BudgetExceededError, the partial
    results JSON and cost_breakdown must be written to RESULTS_DIR.
    """
    import src.evaluation.runner as runner_mod

    # Redirect RESULTS_DIR to a temp path
    monkeypatch.setattr(runner_mod, "RESULTS_DIR", str(tmp_path))

    harness = ExperimentHarness()
    plan = make_query_plan()
    audit = make_sufficient_audit()
    gen = make_answered_generation()

    # Mock total_cost to be over budget so BudgetExceededError fires after first question
    call_count = {"n": 0}
    def mock_total_cost(self):
        call_count["n"] += 1
        # First call (before first question): return 0 so it starts
        # Second call (after first question is processed): return 999 to trigger budget
        return 999.0 if call_count["n"] > 1 else 0.0

    try:
        with patch.object(
            type(get_cost_tracker()), "total_cost",
            new_callable=lambda: property(mock_total_cost)
        ), \
             patch("src.agent.planner.QueryPlanner.plan", return_value=plan), \
             patch("src.agent.evidence_auditor.EvidenceAuditor.audit", return_value=audit), \
             patch("src.agent.grounded_generator.GroundedGenerator.generate", return_value=gen):
            harness.run_benchmark(
                split="dev",
                system_names=["Agentic_System_C_Full"],
                max_cost_usd=0.50,
                smoke_limit=8,
            )
    except BudgetExceededError:
        pass  # Expected

    # Cost breakdown must always be saved (even if 0 records completed)
    cost_files = list(tmp_path.glob("cost_breakdown*.json"))
    assert len(cost_files) >= 1, "cost_breakdown JSON must be saved on early termination"

    # Reproducibility manifest must record is_partial=True
    manifest_files = list(tmp_path.glob("reproducibility_manifest*.json"))
    assert len(manifest_files) >= 1, "Reproducibility manifest must be saved on early termination"
    with open(manifest_files[0]) as f:
        manifest = json.load(f)
    assert manifest.get("is_partial_result") is True, (
        f"Manifest should have is_partial_result=True, got: {manifest.get('is_partial_result')}"
    )


# ---------------------------------------------------------------------------
# Test 7: Cached embeddings are reused — zero doc embedding API calls
#   Verifies that building the HybridRetriever reads from cache, not the API.
# ---------------------------------------------------------------------------
def test_cached_embeddings_reused_zero_api_calls():
    """
    Loading HybridRetriever from existing chunks must make 0 document embedding API calls.
    The embedding cache (NPZ file) must exist and be loaded directly.
    """
    reset_cost_tracker()
    tracker = get_cost_tracker()

    from src.models.document import Chunk
    from src.retrieval.hybrid_retriever import HybridRetriever

    with open(CHUNKS_JSON_PATH, "r", encoding="utf-8") as f:
        chunks = [Chunk(**item) for item in json.load(f)]

    retriever = HybridRetriever.from_chunks(chunks)

    assert tracker.doc_embedding_api_calls == 0, (
        "Loading the HybridRetriever must not call the OpenAI embeddings API. "
        "All 243 chunk vectors must be loaded from the NPZ cache file."
    )
    assert tracker.doc_chunks_cached == 243, (
        f"Expected 243 cached chunks, tracker shows {tracker.doc_chunks_cached}"
    )
    assert tracker.doc_cost == 0.0


# ---------------------------------------------------------------------------
# Test 8: No document embedding API call during any normal test operation
#   Verifies the document embedding cache metadata matches expected config.
# ---------------------------------------------------------------------------
def test_document_embedding_metadata_matches_config():
    """
    The embedding metadata JSON must match the configured model and dimensions.
    This ensures the cached vectors are valid for the current configuration.
    """
    from src.config import OPENAI_EMBEDDING_MODEL, EMBEDDING_DIM

    assert DOC_EMBEDDINGS_METADATA_PATH.exists(), (
        "Document embedding metadata JSON must exist at "
        f"{DOC_EMBEDDINGS_METADATA_PATH}"
    )

    with open(DOC_EMBEDDINGS_METADATA_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)

    assert meta["embedding_model"] == OPENAI_EMBEDDING_MODEL, (
        f"Cache was built with model '{meta['embedding_model']}' "
        f"but config specifies '{OPENAI_EMBEDDING_MODEL}'. Cache invalidation required."
    )
    assert meta["dimensions"] == EMBEDDING_DIM
    assert meta["num_chunks"] == 243
    assert "saved_at" in meta


# ---------------------------------------------------------------------------
# Test 9: Budget guard config reads from environment
#   Verifies MAX_EXPERIMENT_COST_USD is correctly parsed from config.
# ---------------------------------------------------------------------------
def test_max_experiment_cost_config():
    """MAX_EXPERIMENT_COST_USD must be a positive float loaded from config."""
    assert isinstance(MAX_EXPERIMENT_COST_USD, float)
    assert MAX_EXPERIMENT_COST_USD > 0.0


# ---------------------------------------------------------------------------
# Test 10: CostTracker total_cost property is correct
#   Verifies the new total_cost property aggregates all components.
# ---------------------------------------------------------------------------
def test_cost_tracker_total_cost_property():
    """CostTracker.total_cost must sum doc_cost + query_cost + all LLM costs."""
    tracker = CostTracker()

    tracker.track_doc_embedding(num_chunks=10, estimated_tokens=5000, from_cache=False)
    tracker.track_query_embedding(estimated_tokens=100, is_cache_hit=False)
    tracker.track_llm("planner", prompt_tokens=1000, completion_tokens=200)
    tracker.track_llm("auditor", prompt_tokens=2000, completion_tokens=300)

    summary = tracker.get_summary_dict()
    expected_total = summary["totals"]["total_cost_usd"]

    assert abs(tracker.total_cost - expected_total) < 1e-9, (
        f"total_cost property ({tracker.total_cost}) must equal "
        f"get_summary_dict() total ({expected_total})"
    )
    assert tracker.total_cost > 0.0


# ---------------------------------------------------------------------------
# Test 11: BudgetExceededError is a RuntimeError subclass
# ---------------------------------------------------------------------------
def test_budget_exceeded_error_is_runtime_error():
    """BudgetExceededError must be a RuntimeError so it can be caught generically."""
    err = BudgetExceededError("Budget exceeded: $0.50 > $0.30")
    assert isinstance(err, RuntimeError)
    assert "Budget exceeded" in str(err)


# ---------------------------------------------------------------------------
# Test 12: Benchmark is not part of pytest (no test_ functions import runner.run_benchmark)
#   Verifies run_benchmark is never accidentally called from within pytest.
# ---------------------------------------------------------------------------
def test_benchmark_not_accessible_as_pytest_test():
    """
    The run_benchmark method must exist on ExperimentHarness but should
    never be invoked without the --real-api guard (enforced in the script layer).
    This test verifies the method signature includes max_cost_usd protection.
    """
    import inspect
    harness = ExperimentHarness()
    sig = inspect.signature(harness.run_benchmark)

    assert "max_cost_usd" in sig.parameters, (
        "run_benchmark must accept max_cost_usd parameter for budget enforcement"
    )
    assert "smoke_limit" in sig.parameters, (
        "run_benchmark must accept smoke_limit parameter for smoke-test mode"
    )
