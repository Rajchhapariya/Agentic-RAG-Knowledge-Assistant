"""
User PDF Ingestion Pipeline: Coordinates PDF parsing, section-aware chunking,
isolated persistent vector caching, and dynamic HybridRetriever creation.
"""

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Union, Optional, List, Dict, Any, Tuple
import numpy as np

from src.config import (
    USER_UPLOADS_DIR,
    OPENAI_EMBEDDING_MODEL,
    EMBEDDING_DIM,
)
from src.models.document import Document, Chunk
from src.ingestion.pdf_loader import PDFLoader, UnextractablePDFError
from src.ingestion.section_chunker import SectionAwareChunker
from src.retrieval.embeddings import EmbeddingClient
from src.retrieval.hybrid_retriever import HybridRetriever


class UserPDFPipeline:
    """
    Manages the lifecycle of user-uploaded PDFs:
    - Parses PDF into structured Document with page/section provenance
    - Chunks using SectionAwareChunker
    - Checks persistent per-document cache (data/user_uploads/{file_hash}/)
    - Re-uses pre-computed embeddings on cache hit (0 external API calls)
    - Generates & saves embeddings on cache miss
    - Builds an isolated in-memory HybridRetriever completely separated from the research corpus
    """

    CHUNK_VERSION = "v1.0"

    def __init__(
        self,
        upload_dir: Path = USER_UPLOADS_DIR,
        embedding_client: Optional[EmbeddingClient] = None,
        loader: Optional[PDFLoader] = None,
        chunker: Optional[SectionAwareChunker] = None
    ):
        self.upload_dir = Path(upload_dir)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.embedding_client = embedding_client or EmbeddingClient()
        self.loader = loader or PDFLoader()
        self.chunker = chunker or SectionAwareChunker()

    def ingest_pdf(
        self,
        file_source: Union[str, Path, bytes],
        filename: Optional[str] = None,
        force_recompute: bool = False
    ) -> Tuple[Document, List[Chunk], HybridRetriever, Dict[str, Any]]:
        """
        Ingests a user-uploaded PDF, returning (Document, List[Chunk], HybridRetriever, info_dict).
        
        Guarantees:
        - Cache Hit: Loads pre-computed embeddings and chunks (0 document embedding calls).
        - Cache Miss: Generates embeddings, saves cache artifacts, and builds isolated retriever.
        """
        # Step 1: Parse PDF
        doc = self.loader.load_pdf(file_source, filename)
        file_hash = doc.metadata.get("file_hash", "")
        doc_dir = self.upload_dir / file_hash

        doc_path = doc_dir / "document.json"
        chunks_path = doc_dir / "chunks.json"
        embeddings_path = doc_dir / "embeddings.npz"
        metadata_path = doc_dir / "metadata.json"

        # Step 2: Check Cache Validity
        if not force_recompute and all(p.exists() for p in [doc_path, chunks_path, embeddings_path, metadata_path]):
            try:
                with open(metadata_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)

                # Invalidate if embedding model or dimension changed
                model_match = meta.get("embedding_model") == OPENAI_EMBEDDING_MODEL
                dim_match = meta.get("embedding_dimensions") == EMBEDDING_DIM
                version_match = meta.get("chunk_version") == self.CHUNK_VERSION

                if model_match and dim_match and version_match:
                    with open(chunks_path, "r", encoding="utf-8") as f:
                        chunks_data = json.load(f)
                    chunks = [Chunk(**c) for c in chunks_data]

                    npz_data = np.load(embeddings_path)
                    vectors = npz_data["vectors"]

                    if len(chunks) == vectors.shape[0] and vectors.shape[1] == EMBEDDING_DIM:
                        retriever = HybridRetriever.from_chunks_and_vectors(chunks, vectors)
                        info = {
                            "cache_hit": True,
                            "file_hash": file_hash,
                            "filename": meta.get("filename", doc.title),
                            "num_chunks": len(chunks),
                            "num_pages": doc.metadata.get("num_pages", 1),
                            "embedding_model": meta.get("embedding_model"),
                            "dimensions": vectors.shape[1],
                            "doc_dir": str(doc_dir)
                        }
                        return doc, chunks, retriever, info
            except Exception:
                # If cache corrupted, fall through to recompute
                pass

        # Step 3: Cache Miss -> Chunk & Embed
        chunks = self.chunker.chunk_document(doc)
        if not chunks:
            raise UnextractablePDFError(f"No valid text chunks could be created from '{doc.title}'.")

        texts = [c.content for c in chunks]
        vectors = self.embedding_client.embed_raw_texts(texts)

        # Step 4: Persist Cache Artifacts
        doc_dir.mkdir(parents=True, exist_ok=True)

        with open(doc_path, "w", encoding="utf-8") as f:
            json.dump(doc.model_dump(), f, indent=2)

        with open(chunks_path, "w", encoding="utf-8") as f:
            json.dump([c.model_dump() for c in chunks], f, indent=2)

        np.savez_compressed(
            embeddings_path,
            vectors=vectors,
            chunk_ids=np.array([c.chunk_id for c in chunks])
        )

        metadata_payload = {
            "document_hash": file_hash,
            "filename": doc.metadata.get("filename", doc.title),
            "doc_id": doc.doc_id,
            "embedding_model": OPENAI_EMBEDDING_MODEL,
            "embedding_dimensions": vectors.shape[1],
            "chunk_version": self.CHUNK_VERSION,
            "num_chunks": len(chunks),
            "num_pages": doc.metadata.get("num_pages", 1),
            "creation_timestamp": datetime.now(timezone.utc).isoformat()
        }
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata_payload, f, indent=2)

        retriever = HybridRetriever.from_chunks_and_vectors(chunks, vectors)
        info = {
            "cache_hit": False,
            "file_hash": file_hash,
            "filename": doc.metadata.get("filename", doc.title),
            "num_chunks": len(chunks),
            "num_pages": doc.metadata.get("num_pages", 1),
            "embedding_model": OPENAI_EMBEDDING_MODEL,
            "dimensions": vectors.shape[1],
            "doc_dir": str(doc_dir)
        }
        return doc, chunks, retriever, info
