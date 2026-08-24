"""
Corpus fetcher: Downloads arXiv papers and fetches ar5iv structured HTML / PDF files.
"""

import os
import urllib.request
import time
from pathlib import Path
from typing import Dict, Any, List
from src.config import CORPUS_REGISTRY, RAW_DOCS_DIR
from src.models.document import Document
from src.ingestion.parser import ResearchPaperParser


class CorpusFetcher:
    """Manages downloading, caching, and parsing the 10 landmark research papers."""

    def __init__(self, raw_dir: Path = RAW_DOCS_DIR):
        self.raw_dir = raw_dir
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.parser = ResearchPaperParser()

    def fetch_all(self, delay_seconds: float = 1.0) -> List[Document]:
        """Fetches and parses all 10 registered research papers."""
        documents: List[Document] = []
        
        for item in CORPUS_REGISTRY:
            doc = self.fetch_single(item)
            documents.append(doc)
            time.sleep(delay_seconds)
            
        return documents

    def fetch_single(self, doc_meta: Dict[str, Any]) -> Document:
        """Fetches or loads from cache a single research paper."""
        doc_id = doc_meta["doc_id"]
        arxiv_id = doc_meta["arxiv_id"]
        
        html_cache_path = self.raw_dir / f"{doc_id}.html"
        pdf_path = self.raw_dir / f"{doc_id}.pdf"
        
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        
        # 1. Download PDF if not present
        if not pdf_path.exists():
            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
            print(f"Downloading PDF for {doc_id} from {pdf_url}...")
            try:
                req = urllib.request.Request(pdf_url, headers=headers)
                with urllib.request.urlopen(req, timeout=30) as resp:
                    with open(pdf_path, "wb") as f:
                        f.write(resp.read())
                print(f"  Saved {pdf_path.name} ({pdf_path.stat().st_size // 1024} KB)")
            except Exception as e:
                print(f"  Warning: Could not download PDF for {doc_id}: {e}")

        # 2. Fetch or load ar5iv HTML for rich structured section extraction
        if html_cache_path.exists():
            print(f"Loading cached structured HTML for {doc_id}...")
            with open(html_cache_path, "r", encoding="utf-8") as f:
                html_content = f.read()
        else:
            html_url = f"https://ar5iv.labs.arxiv.org/html/{arxiv_id}"
            print(f"Fetching structured HTML for {doc_id} from {html_url}...")
            req = urllib.request.Request(html_url, headers=headers)
            with urllib.request.urlopen(req, timeout=30) as resp:
                html_content = resp.read().decode("utf-8")
            with open(html_cache_path, "w", encoding="utf-8") as f:
                f.write(html_content)
            print(f"  Cached {html_cache_path.name} ({len(html_content) // 1024} KB)")

        # Parse HTML into structured Document
        doc = self.parser.parse_ar5iv_html(html_content, doc_meta)
        print(f"Parsed {doc.doc_id}: {len(doc.sections)} sections, ~{len(doc.full_text.split())} words.")
        return doc
