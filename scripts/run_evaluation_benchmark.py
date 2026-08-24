"""
Evaluation Benchmark Runner

USAGE
-----
Dry-run (default — NO API calls, shows cost estimate):
    python -m scripts.run_evaluation_benchmark

Full benchmark on dev split:
    python -m scripts.run_evaluation_benchmark --split dev --real-api

Full benchmark on test split:
    python -m scripts.run_evaluation_benchmark --split test --real-api

Single-system smoke test (1 question, 1 system):
    python -m scripts.run_evaluation_benchmark --smoke-test --real-api

Restrict to specific systems:
    python -m scripts.run_evaluation_benchmark --split dev --systems Agentic_System_C_Full --real-api

Override budget limit:
    MAX_EXPERIMENT_COST_USD=2.00 python -m scripts.run_evaluation_benchmark --split test --real-api

WARNING: --real-api flag is REQUIRED to make any external OpenAI API calls.
Without it, the benchmark will only print a cost estimate and exit safely.
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import argparse
import json

from src.config import MAX_EXPERIMENT_COST_USD
from src.evaluation.runner import (
    ExperimentHarness,
    print_summary_table,
    compute_dry_run_estimate,
    BENCHMARK_PATH,
)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "RAG Benchmark Evaluation Runner.\n\n"
            "By default (without --real-api) this command performs a DRY RUN ONLY: "
            "it estimates cost and number of API calls, then exits without making any calls."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["dev", "test", "all"],
        help="Dataset split to evaluate (default: test)"
    )
    parser.add_argument(
        "--systems",
        nargs="+",
        default=None,
        help="Optional subset of system names to evaluate (default: all 8)"
    )
    parser.add_argument(
        "--real-api",
        action="store_true",
        help=(
            "REQUIRED to make real OpenAI API calls. "
            "Without this flag the command performs a dry-run cost estimate only."
        )
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help=(
            "Run only 1 representative question through 1 system (Agentic_System_C_Full). "
            "Useful for quick end-to-end verification. Combine with --real-api."
        )
    )
    parser.add_argument(
        "--budget",
        type=float,
        default=MAX_EXPERIMENT_COST_USD,
        help=f"Maximum allowed experiment cost in USD (default: ${MAX_EXPERIMENT_COST_USD:.2f})"
    )
    parser.add_argument(
        "--force-over-budget",
        action="store_true",
        help="Override the budget limit and run anyway (use with extreme caution)."
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Resolve smoke-test configuration
    # ------------------------------------------------------------------
    if args.smoke_test:
        args.split = "dev"
        args.systems = ["Agentic_System_C_Full"]
        smoke_limit = 1  # 1 question only
        print("\n" + "=" * 72)
        print("SMOKE TEST MODE: 1 question × 1 system (Agentic_System_C_Full)")
        print("=" * 72)
    else:
        smoke_limit = None

    # ------------------------------------------------------------------
    # Dry-run estimate (always shown, even when --real-api is passed)
    # ------------------------------------------------------------------
    harness = ExperimentHarness()
    estimate = compute_dry_run_estimate(
        harness=harness,
        split=args.split,
        system_names=args.systems,
        smoke_limit=smoke_limit,
    )

    print("\n" + "=" * 72)
    print("DRY RUN ESTIMATE")
    print("=" * 72)
    print(f"  Split:                    {args.split}")
    print(f"  Questions:                {estimate['num_questions']}")
    print(f"  Systems:                  {estimate['num_systems']}")
    print(f"  Total runs:               {estimate['total_runs']}")
    print(f"  Est. planner calls:       {estimate['est_planner_calls']}")
    print(f"  Est. auditor calls:       {estimate['est_auditor_calls']}")
    print(f"  Est. generator calls:     {estimate['est_generator_calls']}")
    print(f"  Est. embedding calls:     {estimate['est_embedding_calls']}")
    print(f"  Est. total tokens:        {estimate['est_total_tokens']:,}")
    print(f"  Est. cost:                ${estimate['est_cost_usd']:.4f}")
    print(f"  Budget limit:             ${args.budget:.2f}")
    print("=" * 72)

    # ------------------------------------------------------------------
    # Budget guard
    # ------------------------------------------------------------------
    if estimate["est_cost_usd"] > args.budget and not args.force_over_budget:
        print(
            f"\n⚠  BUDGET EXCEEDED: Estimated cost ${estimate['est_cost_usd']:.4f} "
            f"exceeds budget ${args.budget:.2f}."
        )
        if not args.real_api:
            print("   Add --real-api --force-over-budget to run anyway.")
        else:
            print("   Add --force-over-budget to override the budget limit.")
        print("   Exiting safely (no API calls made).\n")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Dry-run exit (no --real-api)
    # ------------------------------------------------------------------
    if not args.real_api:
        print(
            "\n✓  DRY RUN COMPLETE. No API calls made.\n"
            "   Add --real-api to execute the benchmark.\n"
            "   Estimated cost is within budget.\n"
        )
        sys.exit(0)

    # ------------------------------------------------------------------
    # Real execution
    # ------------------------------------------------------------------
    print(f"\n🚀  EXECUTING BENCHMARK (real-api=True, budget=${args.budget:.2f})")
    if args.smoke_test:
        print("   Smoke-test mode: 1 question × 1 system.\n")

    try:
        records, aggregates = harness.run_benchmark(
            split=args.split,
            system_names=args.systems,
            max_cost_usd=args.budget,
            smoke_limit=smoke_limit,
        )
        print_summary_table(aggregates)
    except Exception as exc:
        print(f"\n⛔  Benchmark stopped: {exc}")
        print("   Partial results (if any) have been saved to data/evaluation_results/")
        sys.exit(2)


if __name__ == "__main__":
    main()
