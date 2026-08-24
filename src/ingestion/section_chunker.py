"""
Section-aware recursive token chunker for research papers.
Ensures chunk boundaries strictly respect section hierarchies, token bounds, and overlap.
"""

import re
from typing import List, Dict, Any, Tuple
import tiktoken
from src.config import TARGET_CHUNK_TOKENS, CHUNK_OVERLAP_TOKENS, MIN_CHUNK_TOKENS, TOKENIZER_ENCODING
from src.models.document import Document, Section, Chunk, ChunkMetadata


class SectionAwareChunker:
    """Chunks documents hierarchically by section using token-accurate sentence splitting."""

    def __init__(
        self,
        target_tokens: int = TARGET_CHUNK_TOKENS,
        overlap_tokens: int = CHUNK_OVERLAP_TOKENS,
        min_tokens: int = MIN_CHUNK_TOKENS,
        encoding_name: str = TOKENIZER_ENCODING
    ):
        self.target_tokens = target_tokens
        self.overlap_tokens = overlap_tokens
        self.min_tokens = min_tokens
        self.tokenizer = tiktoken.get_encoding(encoding_name)

    def count_tokens(self, text: str) -> int:
        """Accurately counts tokens using tiktoken."""
        return len(self.tokenizer.encode(text))

    def chunk_document(self, doc: Document) -> List[Chunk]:
        """Chunks an entire Document, respecting Section boundaries."""
        chunks: List[Chunk] = []
        doc_chunk_counter = 0

        for sec in doc.sections:
            sec_chunks = self.chunk_section(sec, doc, doc_chunk_counter)
            chunks.extend(sec_chunks)
            doc_chunk_counter += len(sec_chunks)

        return chunks

    def chunk_section(self, section: Section, doc: Document, start_doc_idx: int = 0) -> List[Chunk]:
        """Chunks a single Section into one or more section-aware Chunks."""
        text = section.content.strip()
        if not text:
            return []

        total_tokens = self.count_tokens(text)
        
        # If section is small enough, it becomes a single chunk
        if total_tokens <= self.target_tokens:
            chunk_id = f"{doc.doc_id}_c{start_doc_idx:03d}"
            metadata = ChunkMetadata(
                chunk_id=chunk_id,
                doc_id=doc.doc_id,
                document_title=doc.title,
                authors=doc.authors,
                year=doc.year,
                section_id=section.section_id,
                section_title=section.section_title,
                section_number=section.section_number,
                chunk_index_in_doc=start_doc_idx,
                chunk_index_in_section=0,
                token_count=total_tokens,
                char_start=section.start_char,
                char_end=section.end_char
            )
            return [Chunk(chunk_id=chunk_id, content=text, metadata=metadata)]

        # Otherwise, split into paragraphs / sentences
        units = self._split_into_units(text)
        chunks: List[Chunk] = []
        current_units: List[str] = []
        current_tokens = 0
        sec_chunk_idx = 0

        for unit in units:
            unit_tokens = self.count_tokens(unit)
            
            if current_tokens + unit_tokens > self.target_tokens and current_units:
                # Emit chunk
                chunk_text = " ".join(current_units).strip()
                chunk_token_count = self.count_tokens(chunk_text)
                
                if chunk_token_count >= self.min_tokens:
                    c_idx = start_doc_idx + len(chunks)
                    chunk_id = f"{doc.doc_id}_c{c_idx:03d}"
                    metadata = ChunkMetadata(
                        chunk_id=chunk_id,
                        doc_id=doc.doc_id,
                        document_title=doc.title,
                        authors=doc.authors,
                        year=doc.year,
                        section_id=section.section_id,
                        section_title=section.section_title,
                        section_number=section.section_number,
                        chunk_index_in_doc=c_idx,
                        chunk_index_in_section=sec_chunk_idx,
                        token_count=chunk_token_count,
                        char_start=section.start_char,
                        char_end=section.end_char
                    )
                    chunks.append(Chunk(chunk_id=chunk_id, content=chunk_text, metadata=metadata))
                    sec_chunk_idx += 1

                # Carry over overlap
                overlap_units = self._get_overlap_units(current_units, self.overlap_tokens)
                current_units = overlap_units + [unit]
                current_tokens = sum(self.count_tokens(u) for u in current_units)
            else:
                current_units.append(unit)
                current_tokens += unit_tokens

        # Emit remaining units as final chunk of section
        if current_units:
            chunk_text = " ".join(current_units).strip()
            chunk_token_count = self.count_tokens(chunk_text)
            
            # If the last chunk is very small and we already have chunks, append to previous chunk
            if chunk_token_count < self.min_tokens and chunks:
                prev_chunk = chunks[-1]
                merged_text = f"{prev_chunk.content} {chunk_text}".strip()
                merged_tokens = self.count_tokens(merged_text)
                # Update previous chunk
                chunks[-1] = Chunk(
                    chunk_id=prev_chunk.chunk_id,
                    content=merged_text,
                    metadata=ChunkMetadata(
                        chunk_id=prev_chunk.metadata.chunk_id,
                        doc_id=prev_chunk.metadata.doc_id,
                        document_title=prev_chunk.metadata.document_title,
                        authors=prev_chunk.metadata.authors,
                        year=prev_chunk.metadata.year,
                        section_id=prev_chunk.metadata.section_id,
                        section_title=prev_chunk.metadata.section_title,
                        section_number=prev_chunk.metadata.section_number,
                        chunk_index_in_doc=prev_chunk.metadata.chunk_index_in_doc,
                        chunk_index_in_section=prev_chunk.metadata.chunk_index_in_section,
                        token_count=merged_tokens,
                        char_start=prev_chunk.metadata.char_start,
                        char_end=section.end_char
                    )
                )
            else:
                c_idx = start_doc_idx + len(chunks)
                chunk_id = f"{doc.doc_id}_c{c_idx:03d}"
                metadata = ChunkMetadata(
                    chunk_id=chunk_id,
                    doc_id=doc.doc_id,
                    document_title=doc.title,
                    authors=doc.authors,
                    year=doc.year,
                    section_id=section.section_id,
                    section_title=section.section_title,
                    section_number=section.section_number,
                    chunk_index_in_doc=c_idx,
                    chunk_index_in_section=sec_chunk_idx,
                    token_count=chunk_token_count,
                    char_start=section.start_char,
                    char_end=section.end_char
                )
                chunks.append(Chunk(chunk_id=chunk_id, content=chunk_text, metadata=metadata))

        return chunks

    def _split_into_units(self, text: str) -> List[str]:
        """Splits text into fine-grained sentence or paragraph units."""
        paragraphs = text.split("\n\n")
        units: List[str] = []
        
        for p in paragraphs:
            p = p.strip()
            if not p:
                continue
            if self.count_tokens(p) <= self.target_tokens:
                units.append(p)
            else:
                # Split large paragraphs by sentence
                sentences = re.split(r'(?<=[.?!])\s+', p)
                for s in sentences:
                    s = s.strip()
                    if not s:
                        continue
                    if self.count_tokens(s) <= self.target_tokens:
                        units.append(s)
                    else:
                        # Fallback for huge run-on sentences: slice tokens
                        tokens = self.tokenizer.encode(s)
                        for i in range(0, len(tokens), self.target_tokens - self.overlap_tokens):
                            sub_tokens = tokens[i : i + self.target_tokens]
                            units.append(self.tokenizer.decode(sub_tokens))
        return units

    def _get_overlap_units(self, units: List[str], max_overlap_tokens: int) -> List[str]:
        """Selects trailing units up to max_overlap_tokens."""
        selected: List[str] = []
        accumulated = 0
        for u in reversed(units):
            toks = self.count_tokens(u)
            if accumulated + toks > max_overlap_tokens:
                break
            selected.insert(0, u)
            accumulated += toks
        return selected
