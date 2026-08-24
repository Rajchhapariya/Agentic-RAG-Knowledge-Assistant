"""
Data models for retrieval queries, search results, and scores.
"""

from typing import Optional, List, Dict, Any, Literal
from pydantic import BaseModel, Field
from src.models.document import ChunkMetadata


class SearchResult(BaseModel):
    """A scored and ranked search result representing a retrieved chunk."""
    chunk_id: str = Field(..., description="Unique chunk identifier")
    doc_id: str = Field(..., description="Document identifier")
    document_title: str = Field(..., description="Title of parent document")
    section_id: str = Field(..., description="Section identifier")
    section_title: str = Field(..., description="Section heading title")
    content: str = Field(..., description="Chunk text content")
    
    # Scores and Ranks
    dense_score: Optional[float] = Field(None, description="Cosine similarity score [-1.0, 1.0]")
    dense_rank: Optional[int] = Field(None, description="1-indexed rank in dense retrieval")
    bm25_score: Optional[float] = Field(None, description="BM25 keyword relevance score")
    bm25_rank: Optional[int] = Field(None, description="1-indexed rank in BM25 retrieval")
    rrf_score: Optional[float] = Field(None, description="Reciprocal Rank Fusion score")
    final_rank: int = Field(..., description="1-indexed final rank in output list")
    
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Full chunk metadata")

    def to_citation_header(self) -> str:
        """Returns a standardized citation label: [DocID: §SectionTitle, Chunk #]"""
        return f"[{self.doc_id}: §{self.section_title}, Chunk {self.chunk_id.split('_c')[-1]}]"


class RetrievalQuery(BaseModel):
    """Encapsulates a retrieval request with filters and parameters."""
    query: str = Field(..., description="Search query string")
    top_k: int = Field(5, description="Number of chunks to return")
    mode: Literal["dense", "sparse", "hybrid"] = Field("hybrid", description="Retrieval mode")
    filters: Optional[Dict[str, Any]] = Field(None, description="Metadata filters (e.g. {'doc_id': 'CRAG_Yan_2024'})")
    rrf_k: int = Field(60, description="RRF constant parameter")
    dense_weight: float = Field(1.0, description="Weight for dense rank in fusion")
    sparse_weight: float = Field(1.0, description="Weight for sparse rank in fusion")


class RetrievalTrace(BaseModel):
    """Observability trace for a single retrieval execution."""
    query: str
    mode: str
    top_k: int
    filters: Optional[Dict[str, Any]]
    results: List[SearchResult]
    latency_ms: float
    total_candidates_scored: int
