"""
PDF Document Loader: Extracts structured text, sections, and page metadata
from user-uploaded PDFs using pdfplumber (primary) and pypdf (fallback).
"""

import io
import re
import hashlib
from pathlib import Path
from typing import Union, Optional, List, Dict, Any, Tuple
import pdfplumber
import pypdf

from src.models.document import Document, Section


class UnextractablePDFError(Exception):
    """Raised when an uploaded PDF has no extractable text (e.g. scanned images only or corrupted)."""
    pass


class PDFLoader:
    """Extracts structured text, headings, and page numbers from PDF files or bytes."""

    # Heading patterns to detect sections in raw text
    HEADING_PATTERNS = [
        re.compile(r'^(?:#+\s*|\d+(\.\d+)*\s+)([A-Z][A-Za-z0-9\s,\-:]{2,60})$'),
        re.compile(r'^(Abstract|Introduction|Related Work|Background|Methodology|Methods|Architecture|Approach|Experiments|Results|Discussion|Conclusion|Limitations)\b', re.IGNORECASE),
    ]

    def load_pdf(
        self,
        file_source: Union[str, Path, bytes],
        filename: Optional[str] = None
    ) -> Document:
        """
        Loads and parses a PDF into a structured Document object.
        Accepts either a filesystem path or raw bytes.
        """
        if isinstance(file_source, (str, Path)):
            path = Path(file_source)
            if not path.exists():
                raise FileNotFoundError(f"PDF file not found: {path}")
            pdf_bytes = path.read_bytes()
            doc_name = filename or path.name
        elif isinstance(file_source, bytes):
            pdf_bytes = file_source
            doc_name = filename or "uploaded_document.pdf"
        else:
            raise ValueError("file_source must be a file path (str/Path) or bytes.")

        if not pdf_bytes or len(pdf_bytes) < 32:
            raise UnextractablePDFError(f"PDF file '{doc_name}' is empty or invalid.")

        # Compute deterministic SHA-256 fingerprint
        file_hash = hashlib.sha256(pdf_bytes).hexdigest()
        doc_id = f"user_{file_hash[:12]}"

        # Attempt extraction: pdfplumber -> fallback to pypdf
        page_texts, extraction_method = self._extract_pages(pdf_bytes, doc_name)

        # Validate extracted text
        total_extracted_chars = sum(len(t.strip()) for _, t in page_texts)
        if total_extracted_chars < 50:
            raise UnextractablePDFError(
                f"PDF '{doc_name}' contains no extractable text ({total_extracted_chars} characters found). "
                "The document may be a scanned image-only PDF, encrypted, or corrupted."
            )

        # Build structured sections with page provenance
        sections = self._build_sections(page_texts, doc_id, doc_name)
        full_text = "\n\n".join([f"## {s.section_title}\n{s.content}" for s in sections])

        # Derive title
        first_section_title = sections[0].section_title if sections else doc_name
        title = doc_name if first_section_title.startswith("Page") else first_section_title

        return Document(
            doc_id=doc_id,
            title=title,
            authors="User Upload",
            year=2024,
            venue="User Document",
            arxiv_id=None,
            full_text=full_text,
            sections=sections,
            metadata={
                "filename": doc_name,
                "file_hash": file_hash,
                "num_pages": len(page_texts),
                "extraction_method": extraction_method,
                "source_type": "user_upload",
            }
        )

    def _extract_pages(self, pdf_bytes: bytes, doc_name: str) -> Tuple[List[Tuple[int, str]], str]:
        """Extracts text page by page using pdfplumber with fallback to pypdf."""
        # 1. Primary: pdfplumber
        try:
            pages: List[Tuple[int, str]] = []
            with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
                for page_idx, page in enumerate(pdf.pages, start=1):
                    txt = page.extract_text() or ""
                    cleaned = self._clean_page_text(txt)
                    if cleaned:
                        pages.append((page_idx, cleaned))
            if sum(len(t) for _, t in pages) >= 50:
                return pages, "pdfplumber"
        except Exception:
            pass

        # 2. Fallback: pypdf
        try:
            pages = []
            reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
            for page_idx, page in enumerate(reader.pages, start=1):
                txt = page.extract_text() or ""
                cleaned = self._clean_page_text(txt)
                if cleaned:
                    pages.append((page_idx, cleaned))
            if sum(len(t) for _, t in pages) >= 50:
                return pages, "pypdf"
        except Exception as e:
            raise UnextractablePDFError(f"Failed to extract text from PDF '{doc_name}': {e}")

        return pages, "none"

    def _clean_page_text(self, text: str) -> str:
        """Cleans headers, footers, whitespace, and Unicode artifacts from page text."""
        if not text:
            return ""
        # Remove null bytes
        t = text.replace('\x00', '')
        # Normalize whitespace
        t = re.sub(r'[ \t]+', ' ', t)
        t = re.sub(r'\n{3,}', '\n\n', t)
        # Normalize quotes and dashes
        t = t.replace('“', '"').replace('”', '"').replace('’', "'").replace('—', ' - ')
        return t.strip()

    def _build_sections(self, page_texts: List[Tuple[int, str]], doc_id: str, doc_name: str) -> List[Section]:
        """Splits page texts into structured sections while preserving page-level provenance."""
        sections: List[Section] = []
        char_offset = 0
        sec_counter = 1

        for page_num, page_content in page_texts:
            lines = page_content.split('\n')
            current_section_title = f"Page {page_num}"
            current_lines: List[str] = []
            
            for line in lines:
                line_str = line.strip()
                if not line_str:
                    continue

                # Check if line looks like a major heading
                is_heading = False
                matched_title = ""
                for pattern in self.HEADING_PATTERNS:
                    m = pattern.match(line_str)
                    if m:
                        is_heading = True
                        matched_title = m.group(0).strip("# ").strip()
                        break

                if is_heading and current_lines:
                    # Flush previous section
                    content = "\n".join(current_lines).strip()
                    if content:
                        sec_id = f"{doc_id}_s{sec_counter:02d}"
                        sections.append(Section(
                            section_id=sec_id,
                            section_title=current_section_title,
                            section_number=str(sec_counter),
                            level=1,
                            content=content,
                            start_char=char_offset,
                            end_char=char_offset + len(content)
                        ))
                        char_offset += len(content) + 2
                        sec_counter += 1
                    current_lines = []
                    current_section_title = f"Page {page_num} - {matched_title}"
                else:
                    current_lines.append(line_str)

            # Flush remaining lines for this page
            if current_lines:
                content = "\n".join(current_lines).strip()
                if content:
                    sec_id = f"{doc_id}_s{sec_counter:02d}"
                    sections.append(Section(
                        section_id=sec_id,
                        section_title=current_section_title,
                        section_number=str(sec_counter),
                        level=1,
                        content=content,
                        start_char=char_offset,
                        end_char=char_offset + len(content)
                    ))
                    char_offset += len(content) + 2
                    sec_counter += 1

        return sections
