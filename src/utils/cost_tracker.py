"""
Usage and Cost Tracking Instrumentation.

Tracks:
- Document embedding API calls, chunks, tokens, and estimated costs.
- Query embedding API calls, cache hits, cache misses, tokens, and estimated costs.
- LLM calls (Planner, Auditor, Generator, Baselines), prompt/completion tokens, and costs.
"""

from typing import Dict, Any, Optional
import threading


class OpenAIAPIDisabledError(RuntimeError):
    """Raised when an OpenAI API call is attempted while OPENAI_API_ENABLED=false or CACHE_ONLY_MODE=true."""
    pass


class BudgetExceededError(RuntimeError):
    """Raised when accumulated experiment cost exceeds the configured MAX_EXPERIMENT_COST_USD limit."""
    pass


# Official pricing as of 2024-2026 (per 1,000,000 tokens)
# text-embedding-3-small: $0.02 / 1M tokens ($0.00002 / 1k tokens)
EMBEDDING_COST_PER_1M = 0.02

# gpt-4o-mini: $0.15 / 1M prompt tokens, $0.60 / 1M completion tokens
GPT4O_MINI_PROMPT_COST_PER_1M = 0.15
GPT4O_MINI_COMPLETION_COST_PER_1M = 0.60


class CostTracker:
    """Thread-safe usage and cost tracker across all retrieval and agent components."""

    def __init__(self):
        self._lock = threading.Lock()
        self.reset()

    def reset(self) -> None:
        """Resets all metrics to zero."""
        with self._lock:
            # Document embeddings
            self.doc_embedding_api_calls = 0
            self.doc_chunks_embedded = 0
            self.doc_chunks_cached = 0
            self.doc_tokens = 0
            self.doc_cost = 0.0

            # Query embeddings
            self.query_embedding_api_calls = 0
            self.query_cache_hits = 0
            self.query_cache_misses = 0
            self.query_tokens = 0
            self.query_cost = 0.0

            # LLM Components: {component_name: {"calls": int, "prompt_tokens": int, "completion_tokens": int, "cost": float}}
            self.llm_usage: Dict[str, Dict[str, Any]] = {
                "planner": {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "cost": 0.0},
                "auditor": {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "cost": 0.0},
                "generator": {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "cost": 0.0},
                "reformulator": {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "cost": 0.0},
                "baseline_generator": {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "cost": 0.0},
            }

    def track_doc_embedding(self, num_chunks: int, estimated_tokens: int, from_cache: bool = False) -> None:
        """Records document embedding activity."""
        with self._lock:
            if from_cache:
                self.doc_chunks_cached += num_chunks
            else:
                self.doc_embedding_api_calls += 1
                self.doc_chunks_embedded += num_chunks
                self.doc_tokens += estimated_tokens
                self.doc_cost += (estimated_tokens / 1_000_000.0) * EMBEDDING_COST_PER_1M

    def track_query_embedding(self, estimated_tokens: int, is_cache_hit: bool = False) -> None:
        """Records query embedding activity."""
        with self._lock:
            if is_cache_hit:
                self.query_cache_hits += 1
            else:
                self.query_cache_misses += 1
                self.query_embedding_api_calls += 1
                self.query_tokens += estimated_tokens
                self.query_cost += (estimated_tokens / 1_000_000.0) * EMBEDDING_COST_PER_1M

    def track_llm(
        self,
        component: str,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> None:
        """Records LLM chat completion token usage and calculates cost."""
        with self._lock:
            comp_key = component.lower()
            if comp_key not in self.llm_usage:
                self.llm_usage[comp_key] = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "cost": 0.0}

            call_cost = (
                (prompt_tokens / 1_000_000.0) * GPT4O_MINI_PROMPT_COST_PER_1M
                + (completion_tokens / 1_000_000.0) * GPT4O_MINI_COMPLETION_COST_PER_1M
            )

            self.llm_usage[comp_key]["calls"] += 1
            self.llm_usage[comp_key]["prompt_tokens"] += prompt_tokens
            self.llm_usage[comp_key]["completion_tokens"] += completion_tokens
            self.llm_usage[comp_key]["cost"] += call_cost

    @property
    def total_cost(self) -> float:
        """Returns current total accumulated cost across all components (thread-safe)."""
        with self._lock:
            total_llm_cost = sum(v["cost"] for v in self.llm_usage.values())
            return self.doc_cost + self.query_cost + total_llm_cost

    def get_summary_dict(self) -> Dict[str, Any]:
        """Returns structured dictionary of usage and cost metrics."""
        with self._lock:
            total_llm_calls = sum(v["calls"] for v in self.llm_usage.values())
            total_llm_tokens = sum(v["prompt_tokens"] + v["completion_tokens"] for v in self.llm_usage.values())
            total_llm_cost = sum(v["cost"] for v in self.llm_usage.values())

            total_cost = self.doc_cost + self.query_cost + total_llm_cost
            total_tokens = self.doc_tokens + self.query_tokens + total_llm_tokens
            total_api_calls = self.doc_embedding_api_calls + self.query_embedding_api_calls + total_llm_calls

            total_query_requests = self.query_cache_hits + self.query_cache_misses
            query_cache_hit_rate = (self.query_cache_hits / total_query_requests) if total_query_requests > 0 else 0.0

            return {
                "document_embeddings": {
                    "api_calls": self.doc_embedding_api_calls,
                    "chunks_embedded": self.doc_chunks_embedded,
                    "chunks_cached": self.doc_chunks_cached,
                    "tokens": self.doc_tokens,
                    "estimated_cost_usd": round(self.doc_cost, 6),
                },
                "query_embeddings": {
                    "api_calls": self.query_embedding_api_calls,
                    "cache_hits": self.query_cache_hits,
                    "cache_misses": self.query_cache_misses,
                    "cache_hit_rate": round(query_cache_hit_rate, 4),
                    "tokens": self.query_tokens,
                    "estimated_cost_usd": round(self.query_cost, 6),
                },
                "llm_breakdown": {
                    comp: {
                        "calls": data["calls"],
                        "prompt_tokens": data["prompt_tokens"],
                        "completion_tokens": data["completion_tokens"],
                        "total_tokens": data["prompt_tokens"] + data["completion_tokens"],
                        "estimated_cost_usd": round(data["cost"], 6),
                    }
                    for comp, data in self.llm_usage.items()
                },
                "totals": {
                    "total_api_calls": total_api_calls,
                    "total_tokens": total_tokens,
                    "total_cost_usd": round(total_cost, 6),
                },
            }

    def get_summary_table(self) -> str:
        """Returns a cleanly formatted Markdown table breaking down API calls, tokens, and estimated cost."""
        summary = self.get_summary_dict()
        doc = summary["document_embeddings"]
        query = summary["query_embeddings"]
        llm = summary["llm_breakdown"]
        tot = summary["totals"]

        lines = [
            "+------------------------------------+------------+---------------+----------------+",
            "| Component                          |  API Calls |  Total Tokens | Estimated Cost |",
            "+------------------------------------+------------+---------------+----------------+",
            f"| Document Embeddings (Offline)      | {doc['api_calls']:>10d} | {doc['tokens']:>13d} | ${doc['estimated_cost_usd']:>13.6f} |",
            f"| Query Embeddings (Online)          | {query['api_calls']:>10d} | {query['tokens']:>13d} | ${query['estimated_cost_usd']:>13.6f} |",
            f"|   -- Query Cache Hits: {query['cache_hits']:<11d} |            |               |                |",
            f"|   -- Query Cache Hit Rate: {query['cache_hit_rate']:>6.1%}  |            |               |                |",
        ]

        for comp, data in llm.items():
            if data["calls"] > 0:
                name = f"LLM: {comp.capitalize()}"
                lines.append(
                    f"| {name:<34} | {data['calls']:>10d} | {data['total_tokens']:>13d} | ${data['estimated_cost_usd']:>13.6f} |"
                )

        lines.append("+------------------------------------+------------+---------------+----------------+")
        lines.append(
            f"| TOTAL USAGE & COST                 | {tot['total_api_calls']:>10d} | {tot['total_tokens']:>13d} | ${tot['total_cost_usd']:>13.6f} |"
        )
        lines.append("+------------------------------------+------------+---------------+----------------+")
        return "\n".join(lines)


# Global singleton instance
_GLOBAL_TRACKER = CostTracker()


def get_cost_tracker() -> CostTracker:
    """Returns global CostTracker instance."""
    return _GLOBAL_TRACKER


def reset_cost_tracker() -> None:
    """Resets global CostTracker instance."""
    _GLOBAL_TRACKER.reset()
