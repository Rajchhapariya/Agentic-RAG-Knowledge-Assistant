"""
Grounded Generator: Synthesizes strictly evidence-backed answers with in-text citations,
explicit contradiction attribution, false-premise debunking, and guaranteed hallucination refusal.
"""

import os
import json
from typing import Optional, List, Dict, Any
from openai import OpenAI
from src.config import (
    OPENAI_API_KEY,
    get_openai_api_key,
    OPENAI_MODEL,
    OPENAI_API_ENABLED,
    CACHE_ONLY_MODE,
)
from src.models.trace import (
    AuditResult,
    GenerationResult,
    CitationItem,
    EvidenceRelationship,
)
from src.utils.cost_tracker import get_cost_tracker, OpenAIAPIDisabledError


class GroundedGenerator:
    """Generates grounded answers backed by verified evidence spans or produces explicit refusals."""

    SYSTEM_PROMPT = """You are a Grounded Research Assistant generating natural, fluent, evidence-backed answers strictly from verified document excerpts.

Rules:
1. Complete Faithfulness: Answer the user's question factually using the provided verified evidence spans. Do NOT extrapolate or introduce external facts.
2. Natural Synthesis: Synthesize the evidence into a clear, cohesive, natural-language response. Do NOT mechanically concatenate quotes or repeat "According to [DocID]" for every sentence. Write like an expert assistant explaining the findings clearly.
3. In-Text Citations: Reference your evidence cleanly using numbered citation tags like [1], [2] corresponding to the citation_index of the evidence spans.
4. False-Premise Correction: If Decision is DEBUNK_FALSE_PREMISE, explicitly state that the premise in the question is incorrect, explain why based on the evidence, and cite the refuting source.
5. Contradiction Attribution: If contradictory evidence is flagged, explicitly compare both perspectives and attribute each claim to its specific document.
6. Refusal Rule: If instructed to REFUSE or if evidence is insufficient, do NOT guess. Explicitly state what information was searched and why evidence is lacking.

You MUST respond with valid JSON adhering to this EXACT schema:
{
  "response_text": "Your clear, fluent, natural-language synthesized answer here with inline citation tags like [1], [2]...",
  "citations": [
    {
      "doc_id": "document ID from the span",
      "document_title": "document title",
      "section_title": "section title",
      "chunk_id": "chunk ID",
      "exact_quote": "exact quote from the verified span",
      "claim_text": "the specific claim being supported"
    }
  ]
}"""

    def __init__(self, api_key: Optional[str] = None, model: str = OPENAI_MODEL):
        self.api_key = api_key or get_openai_api_key() or OPENAI_API_KEY or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is not set.")
        self.client = OpenAI(api_key=self.api_key)
        self.model = model

    def generate(
        self,
        query: str,
        audit_result: AuditResult,
        decision: str,
        direct_response: Optional[str] = None
    ) -> GenerationResult:
        """Generates a grounded response or explicit refusal based on the audit state."""
        
        # 1. Direct conversational response (No retrieval needed)
        if decision == "DIRECT_ANSWER" and direct_response:
            return GenerationResult(
                status="DIRECT_CONVERSATIONAL",
                response_text=direct_response,
                citations=[],
                refusal_reason=None,
                has_conflict_acknowledged=False
            )

        # 2. Explicit Refusal (Hallucination Prevention Gate)
        if decision in {"REFUSE", "DEFINITIVELY_ABSENT"}:
            missing_reasons = "\n".join([f"- {m}" for m in audit_result.missing_information]) if audit_result.missing_information else "- No relevant evidence found in indexed documents."
            refusal_text = (
                f"Based on the indexed document(s), there is insufficient evidence to answer this question.\n\n"
                f"**Missing Evidence Details:**\n{missing_reasons}\n\n"
                f"*The system refused to generate an unverified answer to prevent hallucination.*"
            )
            return GenerationResult(
                status="REFUSED",
                response_text=refusal_text,
                citations=[],
                refusal_reason=missing_reasons,
                has_conflict_acknowledged=False
            )

        # 3. Grounded Synthesis from Verified Spans (Full, Partial, or Debunking)
        spans_payload = []
        for i, span in enumerate(audit_result.verified_supported_spans, start=1):
            spans_payload.append({
                "citation_index": i,
                "chunk_id": span.chunk_id,
                "doc_id": span.doc_id,
                "section_title": span.section_title,
                "exact_quote": span.exact_quote,
                "subquestion": span.subquestion_text
            })

        guidance = "Synthesize a clear, cohesive, natural-language answer to the question using the verified evidence spans. Reference evidence with inline citation numbers like [1], [2]."
        if decision == "PARTIALLY_SUFFICIENT":
            guidance = (
                "Synthesize a clear, cohesive answer to the supported parts of the query factually with citations like [1], [2]. "
                "At the end of your response, explicitly state which parts could NOT be confirmed from the retrieved evidence."
            )
        elif decision == "DEBUNK_FALSE_PREMISE":
            guidance = (
                "The user query asserts a premise that is contradicted or unsupported by the documents. "
                "Explicitly correct and debunk the false premise using the verified evidence spans, citing the refuting source with [1], [2]."
            )

        user_content = (
            f"Query: {query}\n"
            f"Decision: {decision}\n"
            f"Guidance: {guidance}\n\n"
            f"Verified Evidence Spans:\n{json.dumps(spans_payload, indent=2)}\n\n"
            f"Contradiction / False Premise Details:\n{json.dumps(audit_result.contradiction.model_dump(), indent=2)}"
        )

        if not OPENAI_API_ENABLED or CACHE_ONLY_MODE:
            raise OpenAIAPIDisabledError(
                f"OpenAI API is disabled (OPENAI_API_ENABLED={OPENAI_API_ENABLED}, CACHE_ONLY_MODE={CACHE_ONLY_MODE}) "
                f"cannot execute GroundedGenerator."
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
                    "generator",
                    prompt_tokens=response.usage.prompt_tokens,
                    completion_tokens=response.usage.completion_tokens
                )

            raw_data = json.loads(response.choices[0].message.content or "{}")
        except OpenAIAPIDisabledError:
            raise
        except Exception as e:
            # Fallback deterministic extraction from verified quotes
            citations = [
                CitationItem(
                    citation_id=idx,
                    doc_id=s.doc_id,
                    document_title=s.doc_id,
                    section_title=s.section_title,
                    chunk_id=s.chunk_id,
                    exact_quote=s.exact_quote,
                    claim_text=s.exact_quote
                )
                for idx, s in enumerate(audit_result.verified_supported_spans, start=1)
            ]
            summary_quotes = " ".join([f'"{s.exact_quote}" [{s.doc_id}: {s.section_title}]' for s in audit_result.verified_supported_spans])
            caveats = audit_result.missing_information if decision == "PARTIALLY_SUFFICIENT" else None
            
            if decision == "DEBUNK_FALSE_PREMISE":
                conflict_summary = audit_result.contradiction.conflict_summary or "The premise asserted in the question is contradicted by the literature."
                resp_text = f"The premise in the question is incorrect: {conflict_summary} According to the literature: {summary_quotes}"
            else:
                resp_text = f"According to the literature: {summary_quotes}"

            return GenerationResult(
                status="ANSWERED",
                response_text=resp_text,
                citations=citations,
                refusal_reason=None,
                missing_caveats=caveats,
                has_conflict_acknowledged=audit_result.contradiction.has_conflict
            )

        # Build verified citation items
        citations: List[CitationItem] = []
        raw_citations = raw_data.get("citations", [])
        
        if raw_citations:
            for idx, c_item in enumerate(raw_citations, start=1):
                citations.append(CitationItem(
                    citation_id=idx,
                    doc_id=c_item.get("doc_id", "Unknown"),
                    document_title=c_item.get("document_title", c_item.get("doc_id", "Unknown")),
                    section_title=c_item.get("section_title", ""),
                    chunk_id=c_item.get("chunk_id", ""),
                    exact_quote=c_item.get("exact_quote", ""),
                    claim_text=c_item.get("claim_text", "")
                ))
        else:
            for idx, s in enumerate(audit_result.verified_supported_spans, start=1):
                citations.append(CitationItem(
                    citation_id=idx,
                    doc_id=s.doc_id,
                    document_title=s.doc_id,
                    section_title=s.section_title,
                    chunk_id=s.chunk_id,
                    exact_quote=s.exact_quote,
                    claim_text=s.justification or s.exact_quote
                ))

        response_text = (
            raw_data.get("response_text")
            or raw_data.get("answer")
            or raw_data.get("response")
            or raw_data.get("text")
            or raw_data.get("content")
            or raw_data.get("summary")
            or ""
        )
        if not response_text:
            evidence_points = " ".join([f"{s.exact_quote} [{idx}]" for idx, s in enumerate(audit_result.verified_supported_spans, start=1)])
            if decision == "DEBUNK_FALSE_PREMISE":
                conflict_summary = audit_result.contradiction.conflict_summary or "The premise in the question is contradicted by the verified documents."
                response_text = f"The premise in the question is incorrect: {conflict_summary}\n\nBased on verified evidence: {evidence_points}"
            else:
                response_text = f"Based on verified document evidence: {evidence_points}"

        missing_caveats = audit_result.missing_information if decision == "PARTIALLY_SUFFICIENT" else None

        # Append explicit missing caveat paragraph if not already in response_text
        if decision == "PARTIALLY_SUFFICIENT" and audit_result.missing_information:
            missing_bullet_points = "\n".join([f"- {m}" for m in audit_result.missing_information])
            if "missing evidence" not in response_text.lower() and "unconfirmed" not in response_text.lower():
                response_text += f"\n\n**Missing Evidence Note:**\n{missing_bullet_points}"

        return GenerationResult(
            status="ANSWERED",
            response_text=response_text,
            citations=citations,
            refusal_reason=None,
            missing_caveats=missing_caveats,
            has_conflict_acknowledged=bool(audit_result.contradiction.has_conflict)
        )
