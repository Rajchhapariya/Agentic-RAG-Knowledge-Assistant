"""
Pydantic data models for documents, sections, and section-aware chunks.
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class Section(BaseModel):
    """Represents a structural section within a document."""
    section_id: str = Field(..., description="Unique section identifier (e.g. RAG_Lewis_2020_s02)")
    section_title: str = Field(..., description="Section heading title (e.g. '2 Methods')")
    section_number: Optional[str] = Field(None, description="Section number if present (e.g. '2.1')")
    level: int = Field(1, description="Header depth level (1=H1, 2=H2, 3=H3)")
    content: str = Field(..., description="Raw text content of the section")
    start_char: int = Field(0, description="Start character offset in full document")
    end_char: int = Field(0, description="End character offset in full document")


class Document(BaseModel):
    """Represents an ingested research paper."""
    doc_id: str = Field(..., description="Unique document ID (e.g. RAG_Lewis_2020)")
    title: str = Field(..., description="Full paper title")
    authors: str = Field(..., description="Authors list")
    year: int = Field(..., description="Publication year")
    venue: str = Field(..., description="Publication venue or arXiv")
    arxiv_id: Optional[str] = Field(None, description="arXiv ID (e.g. 2005.11401)")
    full_text: str = Field(..., description="Cleaned full text of document")
    sections: List[Section] = Field(default_factory=list, description="Extracted structural sections")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional document metadata")


class ChunkMetadata(BaseModel):
    """Metadata attached to each chunk for filtering, citation, and provenance."""
    chunk_id: str = Field(..., description="Globally unique chunk identifier (e.g. CRAG_Yan_2024_c014)")
    doc_id: str = Field(..., description="Parent document ID")
    document_title: str = Field(..., description="Parent document title")
    authors: str = Field(..., description="Paper authors")
    year: int = Field(..., description="Publication year")
    section_id: str = Field(..., description="Parent section ID")
    section_title: str = Field(..., description="Parent section title")
    section_number: Optional[str] = Field(None, description="Section number")
    chunk_index_in_doc: int = Field(..., description="0-indexed chunk order within document")
    chunk_index_in_section: int = Field(..., description="0-indexed chunk order within section")
    token_count: int = Field(..., description="Token count using cl100k_base tokenizer")
    char_start: int = Field(..., description="Character start offset in document text")
    char_end: int = Field(..., description="Character end offset in document text")


class Chunk(BaseModel):
    """A discrete, section-aware passage of text with enriched metadata."""
    chunk_id: str = Field(..., description="Unique chunk identifier")
    content: str = Field(..., description="Cleaned text content of the chunk")
    metadata: ChunkMetadata = Field(..., description="Rich provenance metadata")

    def __str__(self) -> str:
        return f"[{self.metadata.doc_id} | {self.metadata.section_title} | Chunk {self.metadata.chunk_index_in_doc}] (Tokens: {self.metadata.token_count})\n{self.content[:150]}..."
