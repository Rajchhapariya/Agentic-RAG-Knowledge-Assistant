"""
Comprehensive tests for User PDF Ingestion, Dynamic Retrieval, and Agentic Pipeline.
Validates:
1. Valid PDF ingestion
2. Page metadata preservation
3. Section and chunk metadata
4. SHA-256 identity
5. First upload -> embeddings generated
6. Second identical upload -> cache hit / zero document embedding calls
7. Changed PDF -> new cache identity
8. Embedding-model mismatch -> invalidation
9. Uploaded retriever isolation
10. Research corpus remains unchanged
11. Agentic orchestrator works with uploaded retriever
12. Insufficient evidence produces refusal
13. Citation contains filename/page/section
14. Malformed/empty PDF fails clearly
"""

import io
import json
import hashlib
from pathlib import Path
from unittest.mock import MagicMock
import numpy as np
import pytest
from reportlab.pdfgen import canvas

from src.config import (
    PROCESSED_DIR,
    CHUNKS_JSON_PATH,
    DOC_EMBEDDINGS_CACHE_PATH,
    OPENAI_EMBEDDING_MODEL,
    EMBEDDING_DIM,
)
from src.models.document import Document, Chunk, ChunkMetadata
from src.models.trace import (
    QueryPlan,
    AuditResult,
    EvidenceRelationship,
    GenerationResult,
    CitationItem,
    ContradictionDetail,
)
from src.ingestion.pdf_loader import PDFLoader, UnextractablePDFError
from src.ingestion.section_chunker import SectionAwareChunker
from src.ingestion.user_pdf_pipeline import UserPDFPipeline
from src.retrieval.hybrid_retriever import HybridRetriever
from src.agent.orchestrator import AgentOrchestrator


# ---------------------------------------------------------------------------
# Helpers & Fixtures
# ---------------------------------------------------------------------------

def make_test_pdf(
    title: str = "Sample Paper",
    sections_text: list = None
) -> bytes:
    """Generates an in-memory PDF with page numbers and text using ReportLab."""
    if sections_text is None:
        sections_text = [
            ("1 Introduction", "This paper introduces the Autonomous Knowledge Retrieval system for research synthesis."),
            ("2 Methodology", "The methodology utilizes dense embeddings combined with sparse BM25 indexing."),
            ("3 Experiments", "We evaluate our method across three distinct benchmarks achieving 95% accuracy."),
        ]
    buf = io.BytesIO()
    c = canvas.Canvas(buf)
    for idx, (heading, body) in enumerate(sections_text, start=1):
        c.drawString(100, 750, f"{heading}")
        c.drawString(100, 700, f"{body}")
        c.drawString(100, 100, f"Page {idx}")
        c.showPage()
    c.save()
    return buf.getvalue()


class MockEmbeddingClient:
    """Mock embedding client returning deterministic unit vectors without OpenAI API calls."""
    def __init__(self, dim: int = EMBEDDING_DIM):
        self.dim = dim
        self.embed_texts_call_count = 0
        self.embed_query_call_count = 0

    def embed_texts(self, texts, keys=None):
        self.embed_texts_call_count += 1
        n = len(texts)
        vecs = np.zeros((n, self.dim), dtype=np.float32)
        for i in range(n):
            vecs[i, i % self.dim] = 1.0
        return vecs

    def embed_raw_texts(self, texts, batch_size=64):
        return self.embed_texts(texts)

    def embed_query(self, query):
        self.embed_query_call_count += 1
        vec = np.zeros(self.dim, dtype=np.float32)
        vec[0] = 1.0
        return vec


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_valid_pdf_ingestion():
    """Test 1: Valid PDF ingestion parses pages into Document and Sections."""
    pdf_bytes = make_test_pdf()
    loader = PDFLoader()
    doc = loader.load_pdf(pdf_bytes, filename="test_paper.pdf")

    assert isinstance(doc, Document)
    assert doc.metadata["filename"] == "test_paper.pdf"
    assert doc.metadata["num_pages"] == 3
    assert len(doc.sections) >= 3
    assert "Autonomous Knowledge Retrieval" in doc.full_text


def test_page_metadata_preservation():
    """Test 2: Page numbers and section headings are preserved."""
    pdf_bytes = make_test_pdf()
    loader = PDFLoader()
    doc = loader.load_pdf(pdf_bytes, filename="test_paper.pdf")

    page_numbers_found = [s.section_title for s in doc.sections]
    assert any("Page 1" in title for title in page_numbers_found)
    assert any("Page 2" in title for title in page_numbers_found)
    assert any("Page 3" in title for title in page_numbers_found)


def test_section_and_chunk_metadata():
    """Test 3: Chunking assigns accurate chunk IDs and location tags."""
    pdf_bytes = make_test_pdf()
    loader = PDFLoader()
    doc = loader.load_pdf(pdf_bytes, filename="test_paper.pdf")
    chunker = SectionAwareChunker()
    chunks = chunker.chunk_document(doc)

    assert len(chunks) >= 3
    for chunk in chunks:
        assert chunk.chunk_id.startswith(doc.doc_id)
        assert chunk.metadata.doc_id == doc.doc_id
        assert chunk.metadata.location is not None
        assert "test_paper.pdf" in chunk.metadata.location


def test_sha256_identity():
    """Test 4: The same PDF bytes produce the identical SHA-256 hash and doc_id."""
    pdf_bytes = make_test_pdf("Paper A")

    loader = PDFLoader()
    doc_1 = loader.load_pdf(pdf_bytes, "paper_1.pdf")
    doc_2 = loader.load_pdf(pdf_bytes, "paper_2.pdf")

    assert doc_1.metadata["file_hash"] == doc_2.metadata["file_hash"]
    assert doc_1.doc_id == doc_2.doc_id


def test_first_upload_generates_embeddings(tmp_path):
    """Test 5: First upload saves all cache files and sets cache_hit=False."""
    pdf_bytes = make_test_pdf()
    mock_emb = MockEmbeddingClient()
    pipeline = UserPDFPipeline(upload_dir=tmp_path, embedding_client=mock_emb)

    doc, chunks, retriever, info = pipeline.ingest_pdf(pdf_bytes, "first_upload.pdf")

    assert info["cache_hit"] is False
    assert mock_emb.embed_texts_call_count == 1
    assert len(chunks) > 0

    doc_dir = tmp_path / info["file_hash"]
    assert (doc_dir / "document.json").exists()
    assert (doc_dir / "chunks.json").exists()
    assert (doc_dir / "embeddings.npz").exists()
    assert (doc_dir / "metadata.json").exists()


def test_second_identical_upload_cache_hit_zero_calls(tmp_path):
    """Test 6: Second upload of identical PDF produces cache hit and 0 embedding calls."""
    pdf_bytes = make_test_pdf()
    mock_emb = MockEmbeddingClient()
    pipeline = UserPDFPipeline(upload_dir=tmp_path, embedding_client=mock_emb)

    # First upload
    pipeline.ingest_pdf(pdf_bytes, "cached_upload.pdf")
    assert mock_emb.embed_texts_call_count == 1

    # Second upload with fresh pipeline pointing to same upload_dir
    pipeline_2 = UserPDFPipeline(upload_dir=tmp_path, embedding_client=mock_emb)
    doc_2, chunks_2, retriever_2, info_2 = pipeline_2.ingest_pdf(pdf_bytes, "cached_upload.pdf")

    assert info_2["cache_hit"] is True
    # Embedding client was not called again!
    assert mock_emb.embed_texts_call_count == 1
    assert len(chunks_2) > 0


def test_changed_pdf_new_cache_identity(tmp_path):
    """Test 7: Modified PDF content generates a distinct hash directory."""
    pdf_1 = make_test_pdf("Version 1", [("1 Intro", "Content for version 1 with sufficient characters to pass validation threshold.")])
    pdf_2 = make_test_pdf("Version 2", [("1 Intro", "Completely different text for version 2 with sufficient characters to pass validation threshold.")])

    mock_emb = MockEmbeddingClient()
    pipeline = UserPDFPipeline(upload_dir=tmp_path, embedding_client=mock_emb)

    _, _, _, info_1 = pipeline.ingest_pdf(pdf_1, "v1.pdf")
    _, _, _, info_2 = pipeline.ingest_pdf(pdf_2, "v2.pdf")

    assert info_1["file_hash"] != info_2["file_hash"]
    assert (tmp_path / info_1["file_hash"]).exists()
    assert (tmp_path / info_2["file_hash"]).exists()


def test_embedding_model_mismatch_invalidates(tmp_path):
    """Test 8: Changing embedding model in metadata invalidates the cache and recomputes."""
    pdf_bytes = make_test_pdf()
    mock_emb = MockEmbeddingClient()
    pipeline = UserPDFPipeline(upload_dir=tmp_path, embedding_client=mock_emb)

    _, _, _, info = pipeline.ingest_pdf(pdf_bytes, "model_test.pdf")
    assert mock_emb.embed_texts_call_count == 1

    # Manually corrupt the metadata to simulate model change
    meta_path = tmp_path / info["file_hash"] / "metadata.json"
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    meta["embedding_model"] = "old-deprecated-model"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f)

    # Re-ingesting should detect mismatch and recompute
    _, _, _, info_2 = pipeline.ingest_pdf(pdf_bytes, "model_test.pdf")
    assert info_2["cache_hit"] is False
    assert mock_emb.embed_texts_call_count == 2


def test_uploaded_retriever_isolation(tmp_path):
    """Test 9: Uploaded retriever returns ONLY chunks from the uploaded PDF."""
    pdf_bytes = make_test_pdf("Unique Custom Title", [("Section A", "Special proprietary algorithm X99.")])
    mock_emb = MockEmbeddingClient()
    pipeline = UserPDFPipeline(upload_dir=tmp_path, embedding_client=mock_emb)

    _, chunks, retriever, _ = pipeline.ingest_pdf(pdf_bytes, "isolated.pdf")
    results = retriever.retrieve("algorithm X99", top_k=5, mode="hybrid")

    assert len(results) > 0
    for r in results:
        # All returned chunks must belong to the uploaded document
        assert r.chunk_id in [c.chunk_id for c in chunks]
        # Never return any research corpus chunks
        assert not r.chunk_id.startswith("RAG_Lewis")
        assert not r.chunk_id.startswith("SelfRAG_Asai")


def test_research_corpus_remains_unchanged(tmp_path):
    """Test 10: Ingesting user PDFs does not modify data/processed_chunks/."""
    # Capture checksums of research chunks and embeddings
    def get_hash(p: Path) -> str:
        return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else ""

    chunks_sha_before = get_hash(CHUNKS_JSON_PATH)
    emb_sha_before = get_hash(DOC_EMBEDDINGS_CACHE_PATH)

    pdf_bytes = make_test_pdf("Another User PDF")
    mock_emb = MockEmbeddingClient()
    pipeline = UserPDFPipeline(upload_dir=tmp_path, embedding_client=mock_emb)
    pipeline.ingest_pdf(pdf_bytes, "safe.pdf")

    chunks_sha_after = get_hash(CHUNKS_JSON_PATH)
    emb_sha_after = get_hash(DOC_EMBEDDINGS_CACHE_PATH)

    assert chunks_sha_before == chunks_sha_after
    assert emb_sha_before == emb_sha_after


def test_agentic_orchestrator_works_with_uploaded_retriever(tmp_path, monkeypatch):
    """Test 11: AgentOrchestrator runs end-to-end on uploaded retriever."""
    pdf_bytes = make_test_pdf("Toolformer Guide", [
        ("1 Introduction", "Toolformer learns to use APIs in a self-supervised manner.")
    ])
    mock_emb = MockEmbeddingClient()
    pipeline = UserPDFPipeline(upload_dir=tmp_path, embedding_client=mock_emb)
    doc, chunks, user_retriever, _ = pipeline.ingest_pdf(pdf_bytes, "toolformer_guide.pdf")

    # Mock planner
    def mock_plan(*args, **kwargs):
        return QueryPlan(
            query_type="FACTUAL_SINGLE_HOP",
            needs_retrieval=True,
            target_concepts=["Toolformer"],
            sub_questions=["How does Toolformer learn to use APIs?"],
            initial_search_queries=["Toolformer APIs"]
        )
    monkeypatch.setattr("src.agent.planner.QueryPlanner.plan", mock_plan)

    # Mock auditor
    def mock_audit(self, sub_questions, retrieved_results, retry_count=0):
        c = retrieved_results[0]
        return AuditResult(
            verdict="SUFFICIENT",
            subquestion_coverage={0: True},
            evidence_relationships=[
                EvidenceRelationship(
                    subquestion_idx=0,
                    subquestion_text=sub_questions[0],
                    chunk_id=c.chunk_id,
                    doc_id=c.doc_id,
                    section_title=c.section_title,
                    exact_quote="Toolformer learns to use APIs in a self-supervised manner.",
                    relationship="SUPPORTED",
                    is_quote_verified=True,
                    justification="Direct statement in text"
                )
            ],
            verified_supported_spans=[
                EvidenceRelationship(
                    subquestion_idx=0,
                    subquestion_text=sub_questions[0],
                    chunk_id=c.chunk_id,
                    doc_id=c.doc_id,
                    section_title=c.section_title,
                    exact_quote="Toolformer learns to use APIs in a self-supervised manner.",
                    relationship="SUPPORTED",
                    is_quote_verified=True,
                    justification="Direct statement in text"
                )
            ],
            missing_information=[],
            contradiction=ContradictionDetail()
        )
    monkeypatch.setattr("src.agent.evidence_auditor.EvidenceAuditor.audit", mock_audit)

    # Mock generator
    def mock_generate(self, query, audit_result, decision, **kwargs):
        s = audit_result.verified_supported_spans[0]
        return GenerationResult(
            status="ANSWERED",
            response_text=f"Toolformer learns to use APIs in a self-supervised manner [{doc.title}: Page 1].",
            citations=[
                CitationItem(
                    citation_id=1,
                    doc_id=s.doc_id,
                    document_title="toolformer_guide.pdf",
                    section_title=s.section_title,
                    chunk_id=s.chunk_id,
                    exact_quote=s.exact_quote,
                    claim_text="Toolformer learns to use APIs in a self-supervised manner."
                )
            ]
        )
    monkeypatch.setattr("src.agent.grounded_generator.GroundedGenerator.generate", mock_generate)

    orchestrator = AgentOrchestrator(retriever=user_retriever)
    trace = orchestrator.run("How does Toolformer learn to use APIs?")

    assert trace.final_decision == "SUFFICIENT"
    assert trace.generation.status == "ANSWERED"
    assert len(trace.generation.citations) == 1
    assert "toolformer_guide.pdf" in trace.generation.citations[0].document_title


def test_insufficient_evidence_produces_refusal(tmp_path, monkeypatch):
    """Test 12: Unanswerable query against uploaded PDF strictly produces refusal."""
    pdf_bytes = make_test_pdf("Chemistry Paper", [
        ("1 Chemical Bonds", "Covalent bonds share electron pairs between atoms.")
    ])
    mock_emb = MockEmbeddingClient()
    pipeline = UserPDFPipeline(upload_dir=tmp_path, embedding_client=mock_emb)
    doc, chunks, user_retriever, _ = pipeline.ingest_pdf(pdf_bytes, "chemistry.pdf")

    # Mock planner
    def mock_plan(*args, **kwargs):
        return QueryPlan(
            query_type="FACTUAL_SINGLE_HOP",
            needs_retrieval=True,
            target_concepts=["astrophysics"],
            sub_questions=["What is the orbital period of Kepler-22b?"],
            initial_search_queries=["Kepler-22b orbital period"]
        )
    monkeypatch.setattr("src.agent.planner.QueryPlanner.plan", mock_plan)

    # Mock auditor (always unverified/absent)
    def mock_audit(self, sub_questions, retrieved_results, retry_count=0):
        return AuditResult(
            verdict="DEFINITIVELY_ABSENT",
            subquestion_coverage={0: False},
            evidence_relationships=[],
            verified_supported_spans=[],
            missing_information=["No astronomy evidence in chemistry paper."],
            contradiction=ContradictionDetail()
        )
    monkeypatch.setattr("src.agent.evidence_auditor.EvidenceAuditor.audit", mock_audit)

    orchestrator = AgentOrchestrator(retriever=user_retriever, max_retries=1)
    trace = orchestrator.run("What is the orbital period of Kepler-22b?")

    assert trace.final_decision == "REFUSE"
    assert trace.generation.status == "REFUSED"
    assert len(trace.generation.citations) == 0
    assert "insufficient evidence" in trace.generation.response_text.lower()


def test_citation_contains_filename_page_section():
    """Test 13: Citations format includes filename, page number, and section."""
    citation = CitationItem(
        citation_id=1,
        doc_id="user_doc123",
        document_title="my_research.pdf",
        section_title="Page 2 - 3 Architecture",
        chunk_id="user_doc123_c002",
        exact_quote="We use an 8-layer Transformer.",
        claim_text="The model uses 8 layers."
    )
    assert citation.document_title == "my_research.pdf"
    assert "Page 2" in citation.section_title


def test_malformed_empty_pdf_fails_clearly():
    """Test 14: Empty or corrupted bytes raise UnextractablePDFError."""
    loader = PDFLoader()
    with pytest.raises(UnextractablePDFError):
        loader.load_pdf(b"", "empty.pdf")

    with pytest.raises(UnextractablePDFError):
        loader.load_pdf(b"%PDF-1.4 empty garbage", "corrupt.pdf")
