"""
Query Planner: Analyzes user intent, classifies query types, decomposes multi-hop questions,
and constructs targeted dense/sparse search queries with schema normalization.
"""

import os
import json
from typing import Optional, Dict, Any, List
from openai import OpenAI
from src.config import (
    OPENAI_API_KEY,
    OPENAI_MODEL,
    OPENAI_API_ENABLED,
    CACHE_ONLY_MODE,
)
from src.models.trace import QueryPlan
from src.utils.cost_tracker import get_cost_tracker, OpenAIAPIDisabledError


CANONICAL_DOC_IDS = [
    "RAG_Lewis_2020",
    "REALM_Guu_2020",
    "DPR_Karpukhin_2020",
    "ReAct_Yao_2022",
    "SelfRAG_Asai_2023",
    "Toolformer_Schick_2023",
    "FLARE_Jiang_2023",
    "Reflexion_Shinn_2023",
    "MemGPT_Packer_2023",
    "CRAG_Yan_2024",
]

DOC_ID_ALIAS_MAP = {
    "rag": "RAG_Lewis_2020",
    "realm": "REALM_Guu_2020",
    "dpr": "DPR_Karpukhin_2020",
    "react": "ReAct_Yao_2022",
    "selfrag": "SelfRAG_Asai_2023",
    "self-rag": "SelfRAG_Asai_2023",
    "toolformer": "Toolformer_Schick_2023",
    "flare": "FLARE_Jiang_2023",
    "reflexion": "Reflexion_Shinn_2023",
    "memgpt": "MemGPT_Packer_2023",
    "crag": "CRAG_Yan_2024",
}


def normalize_canonical_doc_id(raw_id: Optional[str]) -> Optional[str]:
    """Normalizes any document ID or alias to the canonical corpus document ID."""
    if not raw_id or not isinstance(raw_id, str):
        return None
    cleaned = raw_id.strip()
    if cleaned in CANONICAL_DOC_IDS:
        return cleaned
    lookup = cleaned.lower().replace("_", "").replace("-", "")
    for alias, canonical in DOC_ID_ALIAS_MAP.items():
        if alias.replace("-", "") == lookup or canonical.lower() == cleaned.lower():
            return canonical
    return None


class QueryPlanner:
    """Classifies user queries, detects multi-hop complexity, and decomposes into atomic search steps."""

    SYSTEM_PROMPT = f"""You are the Query Planner for an Agentic RAG Knowledge Assistant specializing in 10 landmark AI research papers:
1. RAG_Lewis_2020 - Retrieval-Augmented Generation
2. REALM_Guu_2020 - Neural retriever pre-training with MLM
3. DPR_Karpukhin_2020 - Dense Passage Retrieval
4. ReAct_Yao_2022 - Synergizing reasoning and acting
5. SelfRAG_Asai_2023 - Reflection tokens ([Retrieve], [IsREL], [IsSUP], [IsUSE])
6. Toolformer_Schick_2023 - Self-supervised tool use
7. FLARE_Jiang_2023 - Active retrieval on low confidence tokens
8. Reflexion_Shinn_2023 - Verbal reinforcement learning & memory
9. MemGPT_Packer_2023 - Virtual context paging & OS memory
10. CRAG_Yan_2024 - Corrective RAG (evaluator, confidence intervals)

Output a JSON object with these EXACT keys:
{{
  "query_type": "DIRECT_CONVERSATIONAL" | "FACTUAL_SINGLE_HOP" | "MULTI_HOP_COMPARATIVE" | "SEARCH_SYNTHESIS" | "OUT_OF_SCOPE_SUSPECT",
  "needs_retrieval": true | false,
  "direct_response": string or null,
  "target_concepts": ["concept1", "concept2"],
  "sub_questions": ["subquestion 1 string", "subquestion 2 string"],
  "initial_search_queries": ["search query 1 string", "search query 2 string"],
  "suggested_filters": null or {{"doc_id": "<Exact Canonical Doc ID>"}}
}}

Rules:
- "sub_questions" MUST be a list of plain strings (not objects).
- "initial_search_queries" MUST be a list of plain strings.
- For suggested_filters, "doc_id" MUST only be one of the 10 exact canonical IDs listed above, or null if cross-paper or unsure.
- For comparative or multi-faceted questions, query_type must be "MULTI_HOP_COMPARATIVE" and decompose into 2 (max 3) sub_questions.
- For single factual questions, query_type is "FACTUAL_SINGLE_HOP" with 1 sub_question."""

    def __init__(self, api_key: Optional[str] = None, model: str = OPENAI_MODEL):
        self.api_key = api_key or OPENAI_API_KEY or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is not set.")
        self.client = OpenAI(api_key=self.api_key)
        self.model = model

    def plan(self, query: str) -> QueryPlan:
        """Analyzes a query and produces a structured QueryPlan."""
        lower_q = query.strip().lower()
        if lower_q in {"hi", "hello", "hey", "help", "who are you?", "what can you do?"}:
            return QueryPlan(
                query_type="DIRECT_CONVERSATIONAL",
                needs_retrieval=False,
                direct_response=(
                    "Hello! I am an Agentic RAG Knowledge Assistant indexing 10 landmark research papers "
                    "in Retrieval-Augmented Generation and Agentic Reasoning (RAG, REALM, DPR, ReAct, "
                    "Self-RAG, Toolformer, FLARE, Reflexion, MemGPT, CRAG). How can I assist your research?"
                ),
                target_concepts=["overview", "capabilities"],
                sub_questions=[],
                initial_search_queries=[],
                suggested_filters=None
            )

        if not OPENAI_API_ENABLED or CACHE_ONLY_MODE:
            raise OpenAIAPIDisabledError(
                f"OpenAI API is disabled (OPENAI_API_ENABLED={OPENAI_API_ENABLED}, CACHE_ONLY_MODE={CACHE_ONLY_MODE}) "
                f"cannot execute QueryPlanner for query: {query[:40]}..."
            )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": f"Please analyze this user query and return your plan in JSON format:\nUser Query: {query}"}
                ],
                response_format={"type": "json_object"},
                temperature=0.0
            )
            
            # Track usage
            if hasattr(response, "usage") and response.usage:
                get_cost_tracker().track_llm(
                    "planner",
                    prompt_tokens=response.usage.prompt_tokens,
                    completion_tokens=response.usage.completion_tokens
                )

            raw_text = response.choices[0].message.content or "{}"
            data: Dict[str, Any] = json.loads(raw_text)

            # Ensure needs_retrieval is True for any factual or search query
            query_type = data.get("query_type", "FACTUAL_SINGLE_HOP")
            needs_retrieval = data.get("needs_retrieval", True)
            if query_type != "DIRECT_CONVERSATIONAL":
                needs_retrieval = True

            # Normalize and validate suggested_filters with canonical vocabulary
            filters = data.get("suggested_filters")
            if filters and isinstance(filters, dict) and "doc_id" in filters:
                canonical_id = normalize_canonical_doc_id(filters["doc_id"])
                if canonical_id:
                    filters = {"doc_id": canonical_id}
                else:
                    filters = None
            else:
                filters = None

            # Robust parsing of sub_questions
            raw_sub_q = data.get("sub_questions", [query])
            sub_questions: List[str] = []
            if isinstance(raw_sub_q, list):
                for item in raw_sub_q:
                    if isinstance(item, str) and item.strip():
                        sub_questions.append(item.strip())
                    elif isinstance(item, dict):
                        sub_questions.append(item.get("question") or item.get("sub_question") or str(item))
            if not sub_questions:
                sub_questions = [query]

            # Robust parsing of search queries
            raw_sq = data.get("initial_search_queries", [query])
            search_queries: List[str] = []
            if isinstance(raw_sq, list):
                for item in raw_sq:
                    if isinstance(item, str) and item.strip():
                        search_queries.append(item.strip())
                    elif isinstance(item, dict):
                        search_queries.append(item.get("query") or str(item))
            if not search_queries:
                search_queries = [query]

            return QueryPlan(
                query_type=query_type,
                needs_retrieval=needs_retrieval,
                direct_response=data.get("direct_response"),
                target_concepts=data.get("target_concepts", [query]),
                sub_questions=sub_questions,
                initial_search_queries=search_queries,
                suggested_filters=filters
            )

        except Exception as e:
            # Fallback plan on API or schema error
            return QueryPlan(
                query_type="FACTUAL_SINGLE_HOP",
                needs_retrieval=True,
                direct_response=None,
                target_concepts=[query],
                sub_questions=[query],
                initial_search_queries=[query],
                suggested_filters=None
            )

    def _normalize_plan_dict(self, data: Dict[str, Any], query: str) -> Dict[str, Any]:
        """Normalizes dict fields so Pydantic validation never fails due to minor LLM schema variations."""
        q_type = data.get("query_type", "FACTUAL_SINGLE_HOP")
        valid_types = {
            "DIRECT_CONVERSATIONAL",
            "FACTUAL_SINGLE_HOP",
            "MULTI_HOP_COMPARATIVE",
            "SEARCH_SYNTHESIS",
            "OUT_OF_SCOPE_SUSPECT"
        }
        if q_type not in valid_types:
            q_type = "FACTUAL_SINGLE_HOP"
        data["query_type"] = q_type

        # Ensure needs_retrieval boolean: ONLY DIRECT_CONVERSATIONAL is False
        if q_type == "DIRECT_CONVERSATIONAL":
            data["needs_retrieval"] = False
        else:
            data["needs_retrieval"] = True

        # Normalize sub_questions to List[str]
        raw_sub_qs = data.get("sub_questions", [])
        clean_sub_qs: List[str] = []
        if isinstance(raw_sub_qs, list):
            for item in raw_sub_qs:
                if isinstance(item, dict):
                    clean_sub_qs.append(item.get("sub_question") or item.get("question") or str(item))
                elif isinstance(item, str) and item.strip():
                    clean_sub_qs.append(item.strip())
        elif isinstance(raw_sub_qs, str) and raw_sub_qs.strip():
            clean_sub_qs.append(raw_sub_qs.strip())

        if not clean_sub_qs and data["needs_retrieval"]:
            clean_sub_qs = [query]
        data["sub_questions"] = clean_sub_qs

        # Normalize initial_search_queries to List[str]
        raw_search_qs = data.get("initial_search_queries", [])
        clean_search_qs: List[str] = []
        if isinstance(raw_search_qs, list):
            for item in raw_search_qs:
                if isinstance(item, dict):
                    clean_search_qs.append(item.get("search_query") or item.get("query") or str(item))
                elif isinstance(item, str) and item.strip():
                    clean_search_qs.append(item.strip())
        elif isinstance(raw_search_qs, str) and raw_search_qs.strip():
            clean_search_qs.append(raw_search_qs.strip())

        if not clean_search_qs:
            clean_search_qs = clean_sub_qs
        data["initial_search_queries"] = clean_search_qs

        # Normalize target_concepts
        raw_concepts = data.get("target_concepts", [])
        if isinstance(raw_concepts, list):
            data["target_concepts"] = [str(c) for c in raw_concepts if c]
        else:
            data["target_concepts"] = [str(raw_concepts)] if raw_concepts else []

        return data
