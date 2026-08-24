"""
Structure-aware parser for research papers.
Extracts sections, headings, abstracts, and body text while filtering out bibliography/references.
"""

import re
from typing import List, Dict, Any, Optional
from bs4 import BeautifulSoup
from src.models.document import Document, Section


class ResearchPaperParser:
    """Parses structured research papers into clean Document and Section objects."""

    # Headers indicating bibliography/references or appendices that shouldn't pollute retrieval
    STOP_SECTION_PATTERNS = [
        re.compile(r'^(references|bibliography|works cited|literature cited)', re.IGNORECASE),
        re.compile(r'^acknowledg(e)?ments', re.IGNORECASE),
    ]

    def parse_ar5iv_html(self, html_content: str, doc_meta: Dict[str, Any]) -> Document:
        """Parses arXiv ar5iv HTML into a structured Document with Sections."""
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Extract title
        title_el = soup.find('h1', class_=re.compile(r'ltx_title'))
        title = title_el.get_text().strip() if title_el else doc_meta.get("title", "Untitled")

        sections: List[Section] = []
        full_text_parts: List[str] = []
        char_offset = 0

        # 1. Extract Abstract
        abstract_el = soup.find('div', class_=re.compile(r'ltx_abstract'))
        if abstract_el:
            abs_text = self._clean_text(abstract_el.get_text())
            # Strip the leading word "Abstract" if duplicated
            abs_text = re.sub(r'^abstract\s*:?', '', abs_text, flags=re.IGNORECASE).strip()
            if abs_text:
                sec_id = f"{doc_meta['doc_id']}_s00_abstract"
                sec = Section(
                    section_id=sec_id,
                    section_title="Abstract",
                    section_number="0",
                    level=1,
                    content=abs_text,
                    start_char=char_offset,
                    end_char=char_offset + len(abs_text)
                )
                sections.append(sec)
                full_text_parts.append(f"## Abstract\n{abs_text}")
                char_offset += len(abs_text) + 2

        # 2. Extract Body Sections
        section_tags = soup.find_all('section', class_=re.compile(r'ltx_section|ltx_subsection'))
        
        sec_counter = 1
        for stag in section_tags:
            # Check if this tag is inside an already processed parent section to avoid duplication
            # (ar5iv nests subsections inside sections)
            h_tag = stag.find(['h2', 'h3', 'h4', 'h5', 'h6'], class_=re.compile(r'ltx_title'))
            if not h_tag:
                continue
                
            heading_text = h_tag.get_text().strip()
            
            # Check if we reached References / Acknowledgments
            if any(p.match(heading_text) for p in self.STOP_SECTION_PATTERNS):
                continue
                
            # Extract section number and title
            num_match = re.match(r'^(\d+(\.\d+)*)\s+(.*)', heading_text)
            if num_match:
                sec_num = num_match.group(1)
                sec_title = num_match.group(3).strip()
            else:
                sec_num = str(sec_counter)
                sec_title = heading_text
                
            level = 2 if stag.get('class') and 'ltx_subsection' in stag.get('class') else 1

            # Extract paragraphs within this section (excluding nested subsections to prevent duplicate text)
            paragraphs = []
            for child in stag.children:
                if child.name == 'p' or (child.name == 'div' and 'ltx_para' in (child.get('class') or [])):
                    p_text = self._clean_text(child.get_text())
                    if p_text and len(p_text) > 15:
                        paragraphs.append(p_text)

            if not paragraphs:
                continue

            sec_content = "\n\n".join(paragraphs)
            sec_id = f"{doc_meta['doc_id']}_s{sec_counter:02d}"
            
            section_obj = Section(
                section_id=sec_id,
                section_title=f"{sec_num} {sec_title}" if sec_num else sec_title,
                section_number=sec_num,
                level=level,
                content=sec_content,
                start_char=char_offset,
                end_char=char_offset + len(sec_content)
            )
            sections.append(section_obj)
            full_text_parts.append(f"## {section_obj.section_title}\n{sec_content}")
            char_offset += len(sec_content) + 2
            sec_counter += 1

        full_text = "\n\n".join(full_text_parts)

        return Document(
            doc_id=doc_meta["doc_id"],
            title=title,
            authors=doc_meta.get("authors", ""),
            year=doc_meta.get("year", 2020),
            venue=doc_meta.get("venue", "arXiv"),
            arxiv_id=doc_meta.get("arxiv_id", None),
            full_text=full_text,
            sections=sections,
            metadata=doc_meta
        )

    def parse_markdown(self, markdown_text: str, doc_meta: Dict[str, Any]) -> Document:
        """Parses structured Markdown paper into a Document with Sections."""
        lines = markdown_text.split('\n')
        sections: List[Section] = []
        current_title = "Introduction"
        current_num = "1"
        current_lines: List[str] = []
        sec_counter = 1
        char_offset = 0

        for line in lines:
            h_match = re.match(r'^(#{1,3})\s+(.*)', line)
            if h_match:
                if current_lines:
                    content = "\n".join(current_lines).strip()
                    if content:
                        sec_id = f"{doc_meta['doc_id']}_s{sec_counter:02d}"
                        sections.append(Section(
                            section_id=sec_id,
                            section_title=current_title,
                            section_number=current_num,
                            level=1,
                            content=content,
                            start_char=char_offset,
                            end_char=char_offset + len(content)
                        ))
                        char_offset += len(content) + 2
                        sec_counter += 1
                    current_lines = []
                
                heading = h_match.group(2).strip()
                current_title = heading
                num_m = re.match(r'^(\d+(\.\d+)*)\s+(.*)', heading)
                current_num = num_m.group(1) if num_m else str(sec_counter)
            else:
                current_lines.append(line)

        if current_lines:
            content = "\n".join(current_lines).strip()
            if content:
                sec_id = f"{doc_meta['doc_id']}_s{sec_counter:02d}"
                sections.append(Section(
                    section_id=sec_id,
                    section_title=current_title,
                    section_number=current_num,
                    level=1,
                    content=content,
                    start_char=char_offset,
                    end_char=char_offset + len(content)
                ))

        full_text = "\n\n".join([f"## {s.section_title}\n{s.content}" for s in sections])

        return Document(
            doc_id=doc_meta["doc_id"],
            title=doc_meta.get("title", "Untitled"),
            authors=doc_meta.get("authors", ""),
            year=doc_meta.get("year", 2020),
            venue=doc_meta.get("venue", "arXiv"),
            arxiv_id=doc_meta.get("arxiv_id", None),
            full_text=full_text,
            sections=sections,
            metadata=doc_meta
        )

    def _clean_text(self, text: str) -> str:
        """Cleans whitespace, citation markers [1, 2], and LaTeX artifacts."""
        # Replace multiple whitespace/newlines
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        # Normalize quotes and dashes
        text = text.replace('“', '"').replace('”', '"').replace('’', "'").replace('—', ' - ')
        return text.strip()
