"""
Ingestion pipeline: Coordinates paper fetching, parsing, section-aware chunking,
and SQLite/JSON persistence.
"""

import json
import sqlite3
from pathlib import Path
from typing import List, Dict, Any
from src.config import (
    RAW_DOCS_DIR,
    PROCESSED_DIR,
    CHUNKS_JSON_PATH,
    DOCS_JSON_PATH,
    SQLITE_DB_PATH,
    TARGET_CHUNK_TOKENS,
    CHUNK_OVERLAP_TOKENS,
)
from src.models.document import Document, Chunk
from src.ingestion.corpus_fetcher import CorpusFetcher
from src.ingestion.section_chunker import SectionAwareChunker


class IngestionPipeline:
    """End-to-end ingestion pipeline for research paper corpus."""

    def __init__(self, target_tokens: int = TARGET_CHUNK_TOKENS, overlap_tokens: int = CHUNK_OVERLAP_TOKENS):
        self.fetcher = CorpusFetcher(raw_dir=RAW_DOCS_DIR)
        self.chunker = SectionAwareChunker(target_tokens=target_tokens, overlap_tokens=overlap_tokens)

    def run(self) -> Dict[str, Any]:
        """Runs full ingestion: fetch -> parse -> chunk -> persist -> statistics."""
        print("=== Phase 1: Starting Corpus Ingestion & Section-Aware Chunking ===")
        
        # 1. Fetch & parse all documents
        documents: List[Document] = self.fetcher.fetch_all()
        
        # 2. Chunk documents
        all_chunks: List[Chunk] = []
        for doc in documents:
            doc_chunks = self.chunker.chunk_document(doc)
            all_chunks.extend(doc_chunks)
            print(f"  -> {doc.doc_id}: {len(doc_chunks)} section-aware chunks generated.")

        # 3. Persist to JSON files
        self._save_to_json(documents, all_chunks)

        # 4. Persist to SQLite metadata database
        self._save_to_sqlite(documents, all_chunks)

        # 5. Compute summary statistics
        stats = self._compute_statistics(documents, all_chunks)
        print("=== Phase 1 Ingestion Completed Successfully ===")
        return stats

    def _save_to_json(self, documents: List[Document], chunks: List[Chunk]) -> None:
        """Saves documents and chunks to JSON format."""
        docs_data = [doc.model_dump() for doc in documents]
        chunks_data = [chunk.model_dump() for chunk in chunks]

        with open(DOCS_JSON_PATH, "w", encoding="utf-8") as f:
            json.dump(docs_data, f, indent=2)

        with open(CHUNKS_JSON_PATH, "w", encoding="utf-8") as f:
            json.dump(chunks_data, f, indent=2)

        print(f"Saved {len(documents)} documents to {DOCS_JSON_PATH.name}")
        print(f"Saved {len(chunks)} chunks to {CHUNKS_JSON_PATH.name}")

    def _save_to_sqlite(self, documents: List[Document], chunks: List[Chunk]) -> None:
        """Saves documents, sections, and chunk metadata into SQLite for fast querying."""
        conn = sqlite3.connect(SQLITE_DB_PATH)
        cursor = conn.cursor()

        # Create tables
        cursor.execute("DROP TABLE IF EXISTS chunks")
        cursor.execute("DROP TABLE IF EXISTS sections")
        cursor.execute("DROP TABLE IF EXISTS documents")

        cursor.execute("""
            CREATE TABLE documents (
                doc_id TEXT PRIMARY KEY,
                title TEXT,
                authors TEXT,
                year INTEGER,
                venue TEXT,
                arxiv_id TEXT,
                section_count INTEGER,
                word_count INTEGER
            )
        """)

        cursor.execute("""
            CREATE TABLE sections (
                section_id TEXT PRIMARY KEY,
                doc_id TEXT,
                section_title TEXT,
                section_number TEXT,
                level INTEGER,
                content TEXT,
                start_char INTEGER,
                end_char INTEGER,
                FOREIGN KEY (doc_id) REFERENCES documents (doc_id)
            )
        """)

        cursor.execute("""
            CREATE TABLE chunks (
                chunk_id TEXT PRIMARY KEY,
                doc_id TEXT,
                document_title TEXT,
                section_id TEXT,
                section_title TEXT,
                section_number TEXT,
                chunk_index_in_doc INTEGER,
                token_count INTEGER,
                content TEXT,
                char_start INTEGER,
                char_end INTEGER,
                FOREIGN KEY (doc_id) REFERENCES documents (doc_id),
                FOREIGN KEY (section_id) REFERENCES sections (section_id)
            )
        """)

        # Insert documents
        for doc in documents:
            cursor.execute(
                "INSERT INTO documents VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    doc.doc_id,
                    doc.title,
                    doc.authors,
                    doc.year,
                    doc.venue,
                    doc.arxiv_id,
                    len(doc.sections),
                    len(doc.full_text.split())
                )
            )
            # Insert sections
            for sec in doc.sections:
                cursor.execute(
                    "INSERT INTO sections VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        sec.section_id,
                        doc.doc_id,
                        sec.section_title,
                        sec.section_number,
                        sec.level,
                        sec.content,
                        sec.start_char,
                        sec.end_char
                    )
                )

        # Insert chunks
        for c in chunks:
            m = c.metadata
            cursor.execute(
                "INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    c.chunk_id,
                    m.doc_id,
                    m.document_title,
                    m.section_id,
                    m.section_title,
                    m.section_number,
                    m.chunk_index_in_doc,
                    m.token_count,
                    c.content,
                    m.char_start,
                    m.char_end
                )
            )

        conn.commit()
        conn.close()
        print(f"Populated SQLite database: {SQLITE_DB_PATH.name}")

    def _compute_statistics(self, documents: List[Document], chunks: List[Chunk]) -> Dict[str, Any]:
        """Computes comprehensive token, section, and chunk distribution statistics."""
        token_counts = [c.metadata.token_count for c in chunks]
        word_counts = [len(doc.full_text.split()) for doc in documents]
        
        per_doc_chunks = {}
        for c in chunks:
            doc_id = c.metadata.doc_id
            per_doc_chunks[doc_id] = per_doc_chunks.get(doc_id, 0) + 1

        stats = {
            "total_documents": len(documents),
            "total_sections": sum(len(doc.sections) for doc in documents),
            "total_chunks": len(chunks),
            "total_words": sum(word_counts),
            "total_tokens": sum(token_counts),
            "avg_tokens_per_chunk": round(sum(token_counts) / len(token_counts), 1) if chunks else 0,
            "min_tokens_per_chunk": min(token_counts) if token_counts else 0,
            "max_tokens_per_chunk": max(token_counts) if token_counts else 0,
            "per_document_chunk_distribution": per_doc_chunks,
        }
        return stats


if __name__ == "__main__":
    pipeline = IngestionPipeline()
    stats = pipeline.run()
    print("\n--- Ingestion Statistics Summary ---")
    print(json.dumps(stats, indent=2))
