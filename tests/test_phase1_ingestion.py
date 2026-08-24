"""
Phase 1 Automated Test Suite: Document Ingestion, Section Hierarchy & Chunk Integrity.
"""

import json
import sqlite3
import pytest
from src.config import (
    DOCS_JSON_PATH,
    CHUNKS_JSON_PATH,
    SQLITE_DB_PATH,
    CORPUS_REGISTRY,
    RAW_DOCS_DIR,
)


def test_raw_files_exist():
    """Verify all 10 PDFs and HTML source files were downloaded and cached."""
    for item in CORPUS_REGISTRY:
        doc_id = item["doc_id"]
        pdf_file = RAW_DOCS_DIR / f"{doc_id}.pdf"
        html_file = RAW_DOCS_DIR / f"{doc_id}.html"
        assert pdf_file.exists(), f"Missing raw PDF for {doc_id}"
        assert html_file.exists(), f"Missing raw HTML for {doc_id}"
        assert pdf_file.stat().st_size > 50_000, f"PDF for {doc_id} is unexpectedly small"
        assert html_file.stat().st_size > 50_000, f"HTML for {doc_id} is unexpectedly small"


def test_documents_json_integrity():
    """Verify documents.json schema and completeness."""
    assert DOCS_JSON_PATH.exists(), "documents.json does not exist"
    with open(DOCS_JSON_PATH, "r", encoding="utf-8") as f:
        docs = json.load(f)

    assert len(docs) == 10, f"Expected 10 documents, got {len(docs)}"
    doc_ids = {d["doc_id"] for d in docs}
    expected_ids = {item["doc_id"] for item in CORPUS_REGISTRY}
    assert doc_ids == expected_ids, f"Doc ID mismatch: {doc_ids ^ expected_ids}"

    for d in docs:
        assert len(d["title"]) > 10, f"Document {d['doc_id']} has empty title"
        assert len(d["sections"]) >= 5, f"Document {d['doc_id']} has too few sections ({len(d['sections'])})"
        assert len(d["full_text"]) > 1000, f"Document {d['doc_id']} full text is too short"


def test_chunks_json_integrity():
    """Verify chunks.json schema, token counts, and unique identifiers."""
    assert CHUNKS_JSON_PATH.exists(), "chunks.json does not exist"
    with open(CHUNKS_JSON_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    assert len(chunks) >= 200, f"Expected >= 200 chunks, got {len(chunks)}"

    chunk_ids = set()
    for c in chunks:
        cid = c["chunk_id"]
        assert cid not in chunk_ids, f"Duplicate chunk_id detected: {cid}"
        chunk_ids.add(cid)

        meta = c["metadata"]
        assert meta["chunk_id"] == cid
        assert meta["doc_id"] in {item["doc_id"] for item in CORPUS_REGISTRY}
        assert len(meta["document_title"]) > 5
        assert len(meta["section_title"]) > 0
        assert meta["token_count"] > 0
        assert len(c["content"]) > 20


def test_sqlite_database_integrity():
    """Verify SQLite relational tables, counts, and foreign key integrity."""
    assert SQLITE_DB_PATH.exists(), "metadata.db does not exist"
    conn = sqlite3.connect(SQLITE_DB_PATH)
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM documents")
    doc_count = cursor.fetchone()[0]
    assert doc_count == 10, f"Expected 10 documents in SQLite, got {doc_count}"

    cursor.execute("SELECT COUNT(*) FROM sections")
    sec_count = cursor.fetchone()[0]
    assert sec_count >= 100, f"Expected >= 100 sections in SQLite, got {sec_count}"

    cursor.execute("SELECT COUNT(*) FROM chunks")
    chunk_count = cursor.fetchone()[0]
    assert chunk_count >= 200, f"Expected >= 200 chunks in SQLite, got {chunk_count}"

    # Check foreign key consistency
    cursor.execute("""
        SELECT COUNT(*) FROM chunks 
        WHERE doc_id NOT IN (SELECT doc_id FROM documents)
    """)
    orphan_chunks = cursor.fetchone()[0]
    assert orphan_chunks == 0, "Found orphan chunks not matching any document in SQLite"

    conn.close()


def test_representative_content_fidelity():
    """Verify key technical concepts from the literature exist in the chunk index."""
    with open(CHUNKS_JSON_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    all_chunk_text = " ".join(c["content"] for c in chunks)

    # 1. CRAG evaluator concepts
    assert "retrieval evaluator" in all_chunk_text.lower() or "evaluator" in all_chunk_text.lower()
    assert "correct" in all_chunk_text.lower() and "ambiguous" in all_chunk_text.lower()

    # 2. Self-RAG reflection tokens
    assert "retrieve" in all_chunk_text.lower()
    assert "reflection" in all_chunk_text.lower()

    # 3. Toolformer API calls
    assert "calculator" in all_chunk_text.lower() or "api" in all_chunk_text.lower()

    # 4. ReAct Thought/Action/Observation
    assert "thought" in all_chunk_text.lower() and "action" in all_chunk_text.lower()

    # 5. DPR bi-encoders
    assert "question encoder" in all_chunk_text.lower() or "passage encoder" in all_chunk_text.lower() or "bi-encoder" in all_chunk_text.lower()
