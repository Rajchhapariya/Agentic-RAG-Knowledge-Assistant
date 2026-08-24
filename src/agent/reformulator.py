"""
Query Reformulator: Analyzes diagnosed evidence gaps and previous queries to generate
targeted, non-redundant search reformulations for retry passes.
"""

import re
from typing import List, Optional, Set


class QueryReformulator:
    """Generates targeted reformulations based on diagnosed evidence gaps and query history."""

    @staticmethod
    def reformulate(
        original_query: str,
        diagnosed_gap: str,
        missing_information: List[str],
        previous_queries: List[str],
        target_concepts: List[str]
    ) -> str:
        """
        Produces a new search query targeting the missing evidence without repeating terms.
        """
        # Collect all terms already queried to avoid verbatim repetition
        seen_queries = {q.strip().lower() for q in previous_queries}
        
        # 1. If diagnosed gap provides specific missing concepts/parameters
        gap_clean = re.sub(r'^(missing|sub-question \d+|has no verified supporting evidence|specifically)', '', diagnosed_gap, flags=re.IGNORECASE).strip()
        gap_clean = re.sub(r'[^\w\s\-_]', ' ', gap_clean).strip()
        
        # Extract keywords from target concepts and missing info
        concept_terms = [c for c in target_concepts if len(c) > 2]
        
        missing_terms = []
        for m in missing_information:
            m_clean = re.sub(r'^(sub-question \d+ \([^\)]+\) has no verified supporting evidence|no evidence)', '', m, flags=re.IGNORECASE).strip()
            if m_clean and len(m_clean) > 3:
                missing_terms.append(m_clean)

        # Build candidate candidates
        candidates = []
        
        # Strategy A: Missing terms combined with primary concept
        if missing_terms:
            primary_concept = concept_terms[0] if concept_terms else ""
            candidate_a = f"{primary_concept} {missing_terms[0]}".strip()
            candidates.append(candidate_a)
            
        # Strategy B: Direct cleaned gap
        if gap_clean:
            candidate_b = gap_clean
            candidates.append(candidate_b)
            
        # Strategy C: Expanded keywords from original query + gap
        orig_words = set(re.findall(r'\w+', original_query.lower()))
        gap_words = [w for w in re.findall(r'\w+', gap_clean.lower()) if w not in orig_words and len(w) > 2]
        if gap_words:
            candidate_c = f"{' '.join(target_concepts)} {' '.join(gap_words)}".strip()
            candidates.append(candidate_c)

        # Fallback Strategy: Target concepts + specific sub-question focus
        candidate_fallback = f"{' '.join(target_concepts)} {' '.join(missing_terms[:1])}".strip()
        candidates.append(candidate_fallback)

        # Pick first non-duplicate, non-empty candidate
        for cand in candidates:
            cand_norm = cand.strip().lower()
            # Clean up duplicated adjacent words
            cand_norm = re.sub(r'\b(\w+)( \1\b)+', r'\1', cand_norm)
            if cand_norm and cand_norm not in seen_queries and len(cand_norm) > 4:
                return cand_norm

        # If all candidates match seen queries, append targeted modifier
        return f"{target_concepts[0] if target_concepts else ''} {gap_clean} detail parameters".strip()
