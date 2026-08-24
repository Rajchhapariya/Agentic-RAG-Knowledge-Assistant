"""
Embedding client: Generates dense vector embeddings using OpenAI text-embedding-3-small,
with persistent disk caching for both document chunks and query strings, metadata verification,
safety mode switches, and cost instrumentation.
"""

import os
import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Union, Optional, Tuple, Any
import numpy as np
from openai import OpenAI
import tiktoken

from src.config import (
    OPENAI_API_KEY,
    OPENAI_EMBEDDING_MODEL,
    EMBEDDING_DIM,
    EMBEDDINGS_CACHE_PATH,
    DOC_EMBEDDINGS_CACHE_PATH,
    DOC_EMBEDDINGS_METADATA_PATH,
    QUERY_EMBEDDINGS_CACHE_PATH,
    OPENAI_API_ENABLED,
    CACHE_ONLY_MODE,
)
from src.utils.cost_tracker import get_cost_tracker, OpenAIAPIDisabledError


class EmbeddingClient:
    """
    Handles dense vector embedding generation with:
    1. Persistent document embedding cache + metadata invalidation guard.
    2. Persistent query embedding cache for repeated query strings.
    3. Safety mode (OPENAI_API_ENABLED=false / CACHE_ONLY_MODE=true).
    4. Cost and token usage instrumentation.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = OPENAI_EMBEDDING_MODEL,
        doc_cache_path: Path = DOC_EMBEDDINGS_CACHE_PATH,
        metadata_path: Path = DOC_EMBEDDINGS_METADATA_PATH,
        query_cache_path: Path = QUERY_EMBEDDINGS_CACHE_PATH,
        api_enabled: Optional[bool] = None,
        cache_only: Optional[bool] = None,
    ):
        self.api_key = api_key or OPENAI_API_KEY or os.getenv("OPENAI_API_KEY")
        self.api_enabled = OPENAI_API_ENABLED if api_enabled is None else api_enabled
        self.cache_only = CACHE_ONLY_MODE if cache_only is None else cache_only

        # Initialize OpenAI client only if API is enabled and API key is present
        self.client: Optional[OpenAI] = None
        if self.api_enabled and not self.cache_only:
            if self.api_key:
                self.client = OpenAI(api_key=self.api_key)

        self.model = model
        self.dim = EMBEDDING_DIM
        self.doc_cache_path = Path(doc_cache_path)
        self.metadata_path = Path(metadata_path)
        self.query_cache_path = Path(query_cache_path)

        # In-memory vector caches
        self._doc_cache: Dict[str, np.ndarray] = {}
        self._query_cache: Dict[str, np.ndarray] = {}
        
        # Tokenizer for exact token counting
        try:
            self._tokenizer = tiktoken.get_encoding("cl100k_base")
        except Exception:
            self._tokenizer = None

        # Load existing disk caches
        self._load_doc_cache()
        self._load_query_cache()

    def _count_tokens(self, text: str) -> int:
        """Calculates token count using cl100k_base tokenizer."""
        if self._tokenizer:
            return len(self._tokenizer.encode(text))
        return max(1, len(text) // 4)

    def _compute_corpus_hash(self, texts: List[str], keys: List[str]) -> str:
        """Computes SHA-256 hash over sorted chunk keys and content."""
        hasher = hashlib.sha256()
        for k, t in sorted(zip(keys, texts)):
            hasher.update(k.encode("utf-8"))
            hasher.update(t.encode("utf-8"))
        return hasher.hexdigest()

    def _load_doc_cache(self) -> None:
        """Loads cached document embeddings from disk if available."""
        # Check primary doc cache path or fallback legacy path
        target_path = self.doc_cache_path if self.doc_cache_path.exists() else EMBEDDINGS_CACHE_PATH
        if target_path.exists():
            try:
                data = np.load(target_path, allow_pickle=True)
                for key in data.files:
                    self._doc_cache[key] = data[key]
                print(f"[EmbeddingClient] Loaded {len(self._doc_cache)} document embeddings from {target_path.name}")
            except Exception as e:
                print(f"[EmbeddingClient] Warning: Failed to load document embeddings from {target_path}: {e}")
                self._doc_cache = {}

    def _load_query_cache(self) -> None:
        """Loads cached query embeddings from disk if available."""
        if self.query_cache_path.exists():
            try:
                data = np.load(self.query_cache_path, allow_pickle=True)
                for key in data.files:
                    self._query_cache[key] = data[key]
                print(f"[EmbeddingClient] Loaded {len(self._query_cache)} query embeddings from {self.query_cache_path.name}")
            except Exception as e:
                print(f"[EmbeddingClient] Warning: Failed to load query embeddings: {e}")
                self._query_cache = {}

    def save_doc_cache(self, corpus_hash: Optional[str] = None) -> None:
        """Saves document embedding cache and metadata to disk."""
        if self._doc_cache:
            self.doc_cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(self.doc_cache_path, **self._doc_cache)
            # Also keep EMBEDDINGS_CACHE_PATH in sync for backwards compatibility
            np.savez_compressed(EMBEDDINGS_CACHE_PATH, **self._doc_cache)

            # Save metadata
            meta = {
                "embedding_model": self.model,
                "dimensions": self.dim,
                "num_chunks": len(self._doc_cache),
                "corpus_hash": corpus_hash or "unknown",
                "corpus_version": "v1.0",
                "saved_at": datetime.now(timezone.utc).isoformat(),
            }
            with open(self.metadata_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)
            print(f"[EmbeddingClient] Persisted {len(self._doc_cache)} document embeddings + metadata to disk.")

    def save_query_cache(self) -> None:
        """Saves query embedding cache to disk."""
        if self._query_cache:
            self.query_cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(self.query_cache_path, **self._query_cache)

    def _hash_query(self, text: str) -> str:
        """Generates SHA-256 hash for a normalized query string namespaced by embedding model."""
        normalized = text.strip().lower()
        content_hash = hashlib.sha256(normalized.encode('utf-8')).hexdigest()[:24]
        return f"q_{self.model}_{content_hash}"

    def embed_query(self, text: str) -> np.ndarray:
        """
        Embeds a single query string, returning a normalized 1D numpy array.
        Checks persistent query cache before making any API call.
        """
        tracker = get_cost_tracker()
        q_key = self._hash_query(text)

        # 1. Check in-memory / persistent query cache
        if q_key in self._query_cache:
            vec = self._query_cache[q_key]
            tracker.track_query_embedding(estimated_tokens=self._count_tokens(text), is_cache_hit=True)
            return vec

        # 2. Cache miss: verify safety constraints
        if not self.api_enabled or self.cache_only:
            raise OpenAIAPIDisabledError(
                f"OpenAI API is disabled (OPENAI_API_ENABLED={self.api_enabled}, CACHE_ONLY_MODE={self.cache_only}) "
                f"and query embedding for '{text[:40]}...' was not found in cache."
            )

        if not self.client:
            raise ValueError("OpenAI client not initialized. Check OPENAI_API_KEY.")

        # 3. Call OpenAI Embedding API
        tokens = self._count_tokens(text)
        response = self.client.embeddings.create(
            model=self.model,
            input=text.replace("\n", " "),
        )
        vec = np.array(response.data[0].embedding, dtype=np.float32)

        # L2-normalize
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm

        # Save to query cache and persist
        self._query_cache[q_key] = vec
        self.save_query_cache()

        # Track usage
        tracker.track_query_embedding(estimated_tokens=tokens, is_cache_hit=False)
        return vec

    def embed_raw_texts(
        self,
        texts: List[str],
        batch_size: int = 64
    ) -> np.ndarray:
        """
        Embeds arbitrary text passages directly for dynamic user uploads
        WITHOUT writing to or modifying the frozen research document cache.
        """
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)

        tracker = get_cost_tracker()
        if not self.api_enabled or self.cache_only:
            raise OpenAIAPIDisabledError(
                f"OpenAI API is disabled (OPENAI_API_ENABLED={self.api_enabled}, CACHE_ONLY_MODE={self.cache_only}) "
                f"cannot generate embeddings for uploaded documents."
            )

        if not self.client:
            raise ValueError("OpenAI client not initialized. Check OPENAI_API_KEY.")

        vectors = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            cleaned_batch = [t.replace("\n", " ") for t in batch]
            total_tokens = sum(self._count_tokens(t) for t in batch)

            response = self.client.embeddings.create(
                model=self.model,
                input=cleaned_batch
            )
            for item in response.data:
                v = np.array(item.embedding, dtype=np.float32)
                norm = np.linalg.norm(v)
                if norm > 0:
                    v = v / norm
                vectors.append(v)

            tracker.track_doc_embedding(num_chunks=len(batch), estimated_tokens=total_tokens, from_cache=False)

        return np.vstack(vectors)

    def embed_texts(
        self,
        texts: List[str],
        keys: Optional[List[str]] = None,
        batch_size: int = 64,
        force_reembed: bool = False
    ) -> np.ndarray:
        """
        Embeds a batch of document texts, utilizing disk cache for already-embedded chunks.
        Never calls OpenAI API for chunks that already exist in cache.
        """
        tracker = get_cost_tracker()
        if keys is None:
            keys = [f"doc_{hashlib.md5(t.encode('utf-8')).hexdigest()}" for t in texts]

        corpus_hash = self._compute_corpus_hash(texts, keys)

        # Check metadata validity if all chunks match
        all_cached = (
            not force_reembed
            and len(self._doc_cache) >= len(keys)
            and all(k in self._doc_cache for k in keys)
        )

        if all_cached:
            # 100% Cache hit - load all from disk
            results = [self._doc_cache[k] for k in keys]
            tracker.track_doc_embedding(num_chunks=len(keys), estimated_tokens=0, from_cache=True)
            return np.vstack(results)

        # Find missing chunks
        results: List[Optional[np.ndarray]] = [None if force_reembed else self._doc_cache.get(k) for k in keys]
        missing_indices = [i for i, r in enumerate(results) if r is None]

        # Record cached hits
        cached_count = len(keys) - len(missing_indices)
        if cached_count > 0:
            tracker.track_doc_embedding(num_chunks=cached_count, estimated_tokens=0, from_cache=True)

        if missing_indices:
            # Check safety mode
            if not self.api_enabled or self.cache_only:
                raise OpenAIAPIDisabledError(
                    f"OpenAI API is disabled (OPENAI_API_ENABLED={self.api_enabled}, CACHE_ONLY_MODE={self.cache_only}) "
                    f"and {len(missing_indices)} document chunks require embedding."
                )

            if not self.client:
                raise ValueError("OpenAI client not initialized. Check OPENAI_API_KEY.")

            print(f"[EmbeddingClient] Embedding {len(missing_indices)} missing chunks via OpenAI API ({self.model})...")
            
            for i in range(0, len(missing_indices), batch_size):
                batch_idxs = missing_indices[i : i + batch_size]
                batch_texts = [texts[idx].replace("\n", " ") for idx in batch_idxs]
                batch_tokens = sum(self._count_tokens(t) for t in batch_texts)

                resp = self.client.embeddings.create(
                    model=self.model,
                    input=batch_texts,
                )
                for j, item in enumerate(resp.data):
                    target_idx = batch_idxs[j]
                    vec = np.array(item.embedding, dtype=np.float32)
                    norm = np.linalg.norm(vec)
                    if norm > 0:
                        vec = vec / norm
                    results[target_idx] = vec
                    self._doc_cache[keys[target_idx]] = vec

                tracker.track_doc_embedding(
                    num_chunks=len(batch_idxs),
                    estimated_tokens=batch_tokens,
                    from_cache=False,
                )

            # Persist newly embedded chunks to disk
            self.save_doc_cache(corpus_hash=corpus_hash)

        return np.vstack(results)  # Shape: (N, D)

    def get_cache_stats(self) -> Dict[str, Any]:
        """Returns statistics on document and query embedding caches."""
        meta = {}
        if self.metadata_path.exists():
            try:
                with open(self.metadata_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
            except Exception:
                pass

        return {
            "embedding_model": self.model,
            "dimensions": self.dim,
            "document_chunks_cached": len(self._doc_cache),
            "query_embeddings_cached": len(self._query_cache),
            "doc_cache_path": str(self.doc_cache_path),
            "query_cache_path": str(self.query_cache_path),
            "metadata": meta,
        }
