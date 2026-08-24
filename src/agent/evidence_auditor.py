"""
Evidence Auditor / Sufficiency Checker: Evaluates retrieved passages using a 4-way
evidence relationship model (SUPPORTED, UNSUPPORTED, CONTRADICTED, IRRELEVANT) with
conservative multi-stage quote normalization, false-premise detection, and contradiction handling.
"""

import os
import json
import re
from typing import List, Dict, Any, Optional
from openai import OpenAI
from src.config import (
    OPENAI_API_KEY,
    OPENAI_MODEL,
    MAX_RETRIES,
    OPENAI_API_ENABLED,
    CACHE_ONLY_MODE,
)
from src.models.retrieval import SearchResult
from src.models.trace import (
    EvidenceRelationship,
    ContradictionDetail,
    AuditResult,
)
from src.utils.cost_tracker import get_cost_tracker, OpenAIAPIDisabledError


def normalize_text_for_matching(text: str) -> str:
    """
    Conservative normalization for verifying quotes against chunk content.
    Handles:
    - Whitespace, newlines, tabs
    - Unicode quotes, dashes, non-breaking spaces
    - LaTeX formatting and common math/superscript notations (e.g. 10^-5, 10^{-5}, 10⁻⁵, 1e-5)
    - Subscript/superscript markers
    """
    if not text:
        return ""
    
    t = text
    
    # 1. Unicode punctuation normalization
    t = t.replace("\u2018", "'").replace("\u2019", "'")
    t = t.replace("\u201c", '"').replace("\u201d", '"')
    t = t.replace("\u2014", "-").replace("\u2013", "-").replace("\u2212", "-")
    t = t.replace("\u00a0", " ").replace("\u200b", "").replace("\u202f", " ")
    
    # 2. Whitespace normalization
    t = re.sub(r'\s+', ' ', t).strip()
    
    # 3. LaTeX and Math symbol normalization
    t = t.replace("\\times", "*").replace("×", "*")
    t = t.replace("_{", "_").replace("^{", "^").replace("}", "")
    
    # 4. Superscript digits translation (e.g. 10⁻⁵ -> 10^-5)
    superscript_map = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁻", "0123456789-")
    t = re.sub(r'10([⁰¹²³⁴⁵⁶⁷⁸⁹⁻]+)', lambda m: f"10^{m.group(1).translate(superscript_map)}", t)
    
    # 5. Scientific notation canonicalization: e.g. 1 * 10^-5, 1.0 * 10^-5, 10^-5 -> 1e-5
    def norm_sci(match):
        coeff = match.group(1)
        exp = match.group(2)
        try:
            exp_int = int(exp)
            if coeff:
                c_val = float(coeff.replace("*", "").strip())
                if c_val == 1.0:
                    return f"1e{exp_int}"
                return f"{c_val}e{exp_int}"
            return f"1e{exp_int}"
        except Exception:
            return match.group(0)

    t = re.sub(r'(?:(\d+(?:\.\d+)?)\s*\*\s*)?10\^([+-]?\d+)', norm_sci, t)
    
    # Standardize 1e-05 -> 1e-5
    t = re.sub(r'(\d+(?:\.\d+)?)e([+-])0*(\d+)', r'\1e\2\3', t)
    
    return t


def verify_quote_in_text(quote: str, content: str) -> bool:
    """
    Verifies whether an extracted quote is grounded in the candidate chunk content
    using a multi-stage conservative verification strategy.
    
    Stages:
    1. Exact verbatim character substring
    2. Whitespace-normalized substring
    3. Conservative Unicode and scientific/math normalized substring
    """
    if not quote or not content:
        return False
        
    q_raw = quote.strip()
    c_raw = content.strip()
    
    # Stage 1: Exact substring
    if q_raw in c_raw:
        return True
        
    # Stage 2: Whitespace-normalized substring
    q_ws = re.sub(r'\s+', ' ', q_raw)
    c_ws = re.sub(r'\s+', ' ', c_raw)
    if q_ws in c_ws:
        return True
        
    # Stage 3: Conservative Unicode, LaTeX, and scientific-notation normalized substring
    q_norm = normalize_text_for_matching(q_raw)
    c_norm = normalize_text_for_matching(c_raw)
    if q_norm and c_norm and q_norm in c_norm:
        return True
        
    return False


class EvidenceAuditor:
    """Audits retrieved chunks against sub-questions using structured 4-way evidence relations."""

    SYSTEM_PROMPT = """You are an adversarial Evidence Auditor evaluating whether retrieved research paper passages provide sufficient factual proof to answer user sub-questions, or whether the user query contains a false factual premise contradicted by the corpus.

You must evaluate candidate chunks against each sub-question with extreme rigor:
1. For each sub-question, inspect every retrieved chunk and assign an evidence relationship:
   - "SUPPORTED": The chunk contains DIRECT, EXPLICIT factual evidence that answers the sub-question. You must provide an exact verbatim quote (exact_quote) from the chunk text that proves the claim.
   - "UNSUPPORTED": The chunk mentions the topic, entity, or background, but DOES NOT provide the specific fact, value, parameter, or proof required by the sub-question.
   - "CONTRADICTED": The chunk makes a factual statement that directly contradicts the premise of the user's question, or contradicts another retrieved chunk.
   - "IRRELEVANT": The chunk is unrelated to the sub-question.

2. Strict Quote Rule:
   - "exact_quote" MUST be copied from the chunk text. Do NOT paraphrase or alter words.
   - Quoting a sentence that merely mentions a topic without answering the specific question MUST be labeled "UNSUPPORTED", not "SUPPORTED".

3. False-Premise and Contradiction Detection:
   - False Premise in Query: If the user's question asserts a false factual premise or incorrect mechanism (e.g. asking "How does X use Y?" when the corpus proves X uses Z, or X does NOT use Y), you MUST:
     a) Set contradiction.has_conflict = true
     b) Set contradiction.is_false_premise = true
     c) Set contradiction.claim_a = "<the false premise in the user query>"
     d) Set contradiction.source_a = "User Question"
     e) Set contradiction.claim_b = "<the true factual finding from the paper>"
     f) Set contradiction.source_b = "<doc_id of refuting chunk>"
     g) Set contradiction.conflict_summary = "<clear factual explanation of why the question's premise is incorrect and what the paper actually demonstrates>"
     h) Label the refuting chunk relationship as "CONTRADICTED" with the exact quote proving the refutation.
   - Contradiction between Chunks: If two retrieved chunks report conflicting empirical results, conclusions, or mechanisms, set contradiction.has_conflict = true, is_false_premise = false, and fill claim_a, source_a, claim_b, source_b, conflict_summary.

4. Output Schema:
{
  "evidence_relationships": [
    {
      "subquestion_idx": 0,
      "chunk_id": "DocID_c000",
      "doc_id": "DocID",
      "section_title": "Section Title",
      "exact_quote": "verbatim text copied from chunk",
      "relationship": "SUPPORTED" | "UNSUPPORTED" | "CONTRADICTED" | "IRRELEVANT",
      "justification": "why this chunk supports/refutes/fails to support"
    }
  ],
  "missing_information": ["specific missing fact 1", "missing fact 2"],
  "contradiction": {
    "has_conflict": false,
    "is_false_premise": false,
    "claim_a": "",
    "source_a": "",
    "claim_b": "",
    "source_b": "",
    "conflict_summary": ""
  },
  "diagnosed_search_gap": "search query to find missing info, or null if sufficient"
}"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = OPENAI_MODEL,
        max_retries: int = MAX_RETRIES,
        partial_sufficiency_threshold: float = 0.5
    ):
        self.api_key = api_key or OPENAI_API_KEY or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is not set.")
        self.client = OpenAI(api_key=self.api_key)
        self.model = model
        self.max_retries = max_retries
        self.partial_sufficiency_threshold = partial_sufficiency_threshold

    def audit(
        self,
        sub_questions: List[str],
        retrieved_results: List[SearchResult],
        retry_count: int = 0
    ) -> AuditResult:
        """Audits candidate chunks against all sub-questions and computes verified coverage."""
        if not retrieved_results:
            return AuditResult(
                verdict="DEFINITIVELY_ABSENT",
                subquestion_coverage={i: False for i in range(len(sub_questions))},
                evidence_relationships=[],
                verified_supported_spans=[],
                missing_information=["No passages were retrieved for the query."],
                contradiction=ContradictionDetail(),
                diagnosed_search_gap="Expand search terms or remove restrictive filters."
            )

        # Build chunks representation for LLM
        chunks_payload = []
        chunk_lookup: Dict[str, SearchResult] = {}
        for r in retrieved_results:
            chunk_lookup[r.chunk_id] = r
            chunks_payload.append({
                "chunk_id": r.chunk_id,
                "doc_id": r.doc_id,
                "section_title": r.section_title,
                "content": r.content
            })

        user_content = f"Please audit the following candidate chunks and return your assessment in JSON format:\n{json.dumps({'sub_questions': [{'index': i, 'question': sq} for i, sq in enumerate(sub_questions)], 'candidate_chunks': chunks_payload, 'current_retry_count': retry_count, 'max_retries': self.max_retries}, indent=2)}"

        if not OPENAI_API_ENABLED or CACHE_ONLY_MODE:
            raise OpenAIAPIDisabledError(
                f"OpenAI API is disabled (OPENAI_API_ENABLED={OPENAI_API_ENABLED}, CACHE_ONLY_MODE={CACHE_ONLY_MODE}) "
                f"cannot execute EvidenceAuditor."
            )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": user_content}
                ],
                response_format={"type": "json_object"},
                temperature=0.0
            )

            # Track usage
            if hasattr(response, "usage") and response.usage:
                get_cost_tracker().track_llm(
                    "auditor",
                    prompt_tokens=response.usage.prompt_tokens,
                    completion_tokens=response.usage.completion_tokens
                )

            resp_content = response.choices[0].message.content or "{}"
            try:
                raw_data = json.loads(resp_content)
            except json.JSONDecodeError:
                # Sanitize unescaped backslashes (e.g. \url, \times, \text common in LaTeX from LLMs)
                sanitized = re.sub(r'\\(?!["\\/bfnrt]|u[0-9a-fA-F]{4})', r'\\\\', resp_content)
                raw_data = json.loads(sanitized)
        except OpenAIAPIDisabledError:
            raise
        except Exception as e:
            raw_data = {
                "evidence_relationships": [],
                "missing_information": [f"Audit API error: {e}"],
                "contradiction": {"has_conflict": False},
                "diagnosed_search_gap": sub_questions[0] if sub_questions else "retry search"
            }

        # Parse raw relationships
        raw_rels = raw_data.get("evidence_relationships", [])
        if not isinstance(raw_rels, list):
            raw_rels = []
            
        parsed_rels: List[EvidenceRelationship] = []
        for item in raw_rels:
            if not isinstance(item, dict):
                continue
            try:
                sq_idx = item.get("subquestion_idx", 0)
                if not isinstance(sq_idx, int) or sq_idx >= len(sub_questions):
                    sq_idx = 0
                sq_text = sub_questions[sq_idx] if sub_questions else ""
                cid = str(item.get("chunk_id", ""))
                chunk_obj = chunk_lookup.get(cid)
                doc_id = chunk_obj.doc_id if chunk_obj else str(item.get("doc_id", "Unknown"))
                sec_title = chunk_obj.section_title if chunk_obj else str(item.get("section_title", "Unknown"))
                
                rel = EvidenceRelationship(
                    subquestion_idx=sq_idx,
                    subquestion_text=sq_text,
                    chunk_id=cid,
                    doc_id=doc_id,
                    section_title=sec_title,
                    exact_quote=str(item.get("exact_quote", "")),
                    relationship=item.get("relationship", "IRRELEVANT"),
                    is_quote_verified=False,
                    justification=str(item.get("justification", ""))
                )
                parsed_rels.append(rel)
            except Exception:
                pass

        # --- Multi-Stage Deterministic Quote Verification in Python ---
        verified_supported_spans: List[EvidenceRelationship] = []
        subquestion_coverage: Dict[int, bool] = {i: False for i in range(len(sub_questions))}

        for rel in parsed_rels:
            chunk = chunk_lookup.get(rel.chunk_id)
            quote = rel.exact_quote.strip()
            
            # Deterministic multi-stage quote verification
            if chunk and quote:
                rel.is_quote_verified = verify_quote_in_text(quote, chunk.content)
            else:
                rel.is_quote_verified = False

            # If quote was not found verbatim/normalized, downgrade relationship
            if not rel.is_quote_verified and rel.relationship == "SUPPORTED":
                rel.relationship = "UNSUPPORTED"
                rel.justification += " [REJECTED: Quote does not exist in candidate chunk text]"

            # Support Criteria: Must be SUPPORTED AND Quote must be verified
            if rel.relationship == "SUPPORTED" and rel.is_quote_verified:
                verified_supported_spans.append(rel)
                subquestion_coverage[rel.subquestion_idx] = True
            elif rel.relationship == "CONTRADICTED" and rel.is_quote_verified:
                # Include verified refuting quotes for debunking synthesis
                verified_supported_spans.append(rel)

        # Parse Contradiction & False-Premise Details
        contra_data = raw_data.get("contradiction", {})
        if not isinstance(contra_data, dict):
            contra_data = {}
        contradiction = ContradictionDetail(
            has_conflict=bool(contra_data.get("has_conflict", False)),
            is_false_premise=bool(contra_data.get("is_false_premise", False)),
            claim_a=str(contra_data.get("claim_a", "")),
            source_a=str(contra_data.get("source_a", "")),
            claim_b=str(contra_data.get("claim_b", "")),
            source_b=str(contra_data.get("source_b", "")),
            conflict_summary=str(contra_data.get("conflict_summary", ""))
        )

        # Check if any individual relationship flagged a false-premise contradiction
        has_contradicted_rel = any(r.relationship == "CONTRADICTED" and r.is_quote_verified for r in parsed_rels)
        if has_contradicted_rel and not contradiction.has_conflict:
            contradiction.has_conflict = True
            contra_rels = [r for r in parsed_rels if r.relationship == "CONTRADICTED" and r.is_quote_verified]
            if contra_rels:
                contradiction.conflict_summary = contra_rels[0].justification or "Evidence contradicts question premise or document claim."

        # Normalize missing_information to List[str]
        raw_missing = raw_data.get("missing_information", [])
        missing_info: List[str] = []
        if isinstance(raw_missing, list):
            for m in raw_missing:
                if isinstance(m, str) and m.strip() and m.lower() not in {"none", "n/a", "no missing information", "null"}:
                    missing_info.append(m.strip())
        elif isinstance(raw_missing, str) and raw_missing.strip() and raw_missing.lower() not in {"none", "n/a", "no missing information", "null"}:
            missing_info.append(raw_missing.strip())

        # If any sub-question lacks coverage and no contradiction/false-premise, record missing
        for sq_idx, covered in subquestion_coverage.items():
            if not covered and not contradiction.has_conflict:
                sq_name = sub_questions[sq_idx] if sq_idx < len(sub_questions) else f"Sub-question {sq_idx+1}"
                msg = f"Sub-question {sq_idx+1} ('{sq_name}') has no verified supporting evidence."
                if not any(sq_name in m for m in missing_info):
                    missing_info.append(msg)

        diagnosed_gap = raw_data.get("diagnosed_search_gap")
        if not diagnosed_gap and missing_info:
            diagnosed_gap = missing_info[0]

        # --- Deterministic Sufficiency & False-Premise Classification Policy ---
        all_covered = all(subquestion_coverage.values()) and len(sub_questions) > 0
        has_supported_spans = len(verified_supported_spans) >= len(sub_questions)

        # Distinguish between False Premise in query vs Inter-document Contradiction
        is_query_false_premise = contradiction.has_conflict and (
            contradiction.is_false_premise or 
            "user" in contradiction.source_a.lower() or 
            "question" in contradiction.source_a.lower() or 
            "premise" in contradiction.conflict_summary.lower()
        )

        if is_query_false_premise:
            verdict = "DEBUNK_FALSE_PREMISE"
        elif contradiction.has_conflict or has_contradicted_rel:
            verdict = "CONTRADICTED"
        elif all_covered and has_supported_spans:
            verdict = "SUFFICIENT"
            missing_info = []  # Clear missing info if all required sub-questions are supported
        elif retry_count < self.max_retries:
            has_any_relation = any(r.relationship in {"SUPPORTED", "UNSUPPORTED", "CONTRADICTED"} for r in parsed_rels)
            if has_any_relation or len(retrieved_results) > 0:
                verdict = "INSUFFICIENT_RETRY"
            else:
                verdict = "DEFINITIVELY_ABSENT"
        else:
            # Retries exhausted -> evaluate partial sufficiency vs complete refusal
            n_sub_q = max(1, len(sub_questions))
            covered_count = sum(1 for c in subquestion_coverage.values() if c)
            coverage_ratio = covered_count / float(n_sub_q)

            if coverage_ratio >= self.partial_sufficiency_threshold and len(verified_supported_spans) > 0:
                verdict = "PARTIALLY_SUFFICIENT"
            else:
                verdict = "DEFINITIVELY_ABSENT"

        return AuditResult(
            verdict=verdict,
            subquestion_coverage=subquestion_coverage,
            evidence_relationships=parsed_rels,
            verified_supported_spans=verified_supported_spans,
            missing_information=missing_info,
            contradiction=contradiction,
            diagnosed_search_gap=diagnosed_gap if verdict == "INSUFFICIENT_RETRY" else None
        )
