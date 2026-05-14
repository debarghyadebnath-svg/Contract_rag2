"""
contract_chunker.py
-------------------
3-layer chunking pipeline for the IHDA Mortgage Purchase Agreement (36 pages).

Layer 1 → Definition Chunks   (Section 1 — every quoted term = 1 atomic chunk)
Layer 2 → Section Chunks      (Sections 2–14, split at subsection boundaries)
Layer 3 → Table Chunks        (recapture tables + worked examples, 3-format serialisation)

Rules enforced per the spec:
- pdfplumber ONLY for PDF extraction + table detection
- No LangChain splitters — SectionAwareSplitter is fully custom
- Every page_content starts with a bracketed breadcrumb header
- Definition chunks have ZERO overlap (atomic + immutable)
- Section chunks overlap 150 tokens from prev chunk tail
- Tables are ALWAYS atomic (never split mid-row or mid-example)
- Tables serialised in 3 formats: markdown + pipe-delimited + prose
- Duplicate table in MCC Addendum (page 36) gets is_duplicate=True
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pdfplumber

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Chunk:
    page_content: str
    metadata: dict[str, Any]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_KNOWN_DEFINED_TERMS: set[str] = set()  # populated during Layer 1 pass

# Approx 1 token ≈ 4 chars (GPT-style). Good enough for overlap budgeting.
def _token_count(text: str) -> int:
    return max(1, len(text) // 4)


def _slugify(text: str) -> str:
    """Turn arbitrary text into a snake_case id fragment."""
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")[:60]


def _tail_tokens(text: str, n_tokens: int = 150) -> str:
    """Return the last ~n_tokens worth of characters from text."""
    char_budget = n_tokens * 4
    return text[-char_budget:] if len(text) > char_budget else text


def _build_markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    col_w = [max(len(h), max((len(r[i]) for r in rows), default=0)) for i, h in enumerate(headers)]
    sep = "|" + "|".join("-" * (w + 2) for w in col_w) + "|"
    head = "| " + " | ".join(h.ljust(col_w[i]) for i, h in enumerate(headers)) + " |"
    body = "\n".join(
        "| " + " | ".join(c.ljust(col_w[i]) for i, c in enumerate(row)) + " |"
        for row in rows
    )
    return f"{head}\n{sep}\n{body}"


def _build_pipe_row(headers: list[str], rows: list[list[str]]) -> str:
    return " | ".join(f"{h}={r[0]}" if len(headers) == 1 else "" for h, r in zip(headers, rows))


# ---------------------------------------------------------------------------
# PDF extraction — pdfplumber only
# ---------------------------------------------------------------------------

@dataclass
class PageData:
    page_number: int
    text: str
    tables: list[list[list[str | None]]]  # list of pdfplumber table rows


def _extract_all_pages(pdf_path: Path) -> list[PageData]:
    pages: list[PageData] = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            # Use layout=True to preserve column/indentation structure
            text = page.extract_text(layout=True) or ""
            tables = page.extract_tables() or []
            # Replace None cells with empty string
            clean_tables = [
                [[cell or "" for cell in row] for row in table]
                for table in tables
            ]
            pages.append(PageData(page_number=i, text=text, tables=clean_tables))
    return pages


# ---------------------------------------------------------------------------
# Layer 1 — Definition Chunks (Section 1)
# ---------------------------------------------------------------------------

# Matches lines like:  "Eligible Borrower":  or  "Bond":
_DEF_TERM_PATTERN = re.compile(r'^"([A-Z][^"]+)":', re.MULTILINE)


def _find_section1_bounds(pages: list[PageData]) -> tuple[int, int]:
    """
    Return (start_char_offset_in_joined_text, end_char_offset_in_joined_text)
    for Section 1. We detect Section 1 start and stop when Section 2 begins.
    Works on the joined full-document text (with page markers).
    """
    joined = "\n".join(p.text for p in pages)
    sec1_match = re.search(r'Section\s+1[\.\s]', joined, re.IGNORECASE)
    sec2_match = re.search(r'Section\s+2[\.\s]', joined, re.IGNORECASE)
    start = sec1_match.start() if sec1_match else 0
    end = sec2_match.start() if sec2_match else len(joined)
    return start, end


def _parse_definitions(section1_text: str) -> list[tuple[str, str]]:
    """
    Returns list of (term, full_definition_text) pairs.
    A definition spans from its opening quote pattern until the next definition
    term OR end of section text.
    """
    matches = list(_DEF_TERM_PATTERN.finditer(section1_text))
    definitions: list[tuple[str, str]] = []
    for idx, m in enumerate(matches):
        term = m.group(1)
        start = m.start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(section1_text)
        raw = section1_text[start:end].strip()
        definitions.append((term, raw))
        _KNOWN_DEFINED_TERMS.add(term)
    return definitions


def _layer1_definition_chunks(pages: list[PageData]) -> list[Chunk]:
    joined = "\n".join(p.text for p in pages)
    start, end = _find_section1_bounds(pages)
    section1_text = joined[start:end]

    definitions = _parse_definitions(section1_text)
    chunks: list[Chunk] = []

    for term, body in definitions:
        chunk_id = f"def_{_slugify(term)}"
        breadcrumb = f"[IHDA Agreement | Section 1 — Definitions | Term: {term}]"
        page_content = f"{breadcrumb}\n\n{body}"

        chunks.append(Chunk(
            page_content=page_content,
            metadata={
                "chunk_id": chunk_id,
                "chunk_type": "definition",
                "term": term,
                "section": "1",
                "doc_source": "main",
                "retrieval_priority": "critical",
                "token_count": _token_count(page_content),
                "contains_table": False,
            },
        ))

    return chunks


# ---------------------------------------------------------------------------
# Layer 2 — Section/Subsection Chunks (Sections 2–14)
# ---------------------------------------------------------------------------

# Matches:  "7.1. Submission Requirements."  or  "Section 7."
_SUBSEC_PATTERN = re.compile(
    r'^(?:Section\s+)?(\d{1,2})\.(\d{1,2})\.\s+([^\n]+)',
    re.MULTILINE,
)
_SECTION_PATTERN = re.compile(
    r'^Section\s+(\d{1,2})\.\s+([^\n]+)',
    re.MULTILINE,
)

# Obligation keywords → which party
_OBLIGATION_MAP = [
    (re.compile(r'\bLender\s+shall\b', re.IGNORECASE), "Lender"),
    (re.compile(r'\bAuthority\s+shall\b', re.IGNORECASE), "Authority"),
    (re.compile(r'\bBorrower\s+shall\b', re.IGNORECASE), "Borrower"),
]

_WARRANTY_PATTERN = re.compile(
    r'(?:^|\n)\s*\(?\w+\)?\s+(?:represents?|warrant(?:s|ies)|certif(?:ies|y))',
    re.IGNORECASE,
)

MAX_TOKENS = 800
OVERLAP_TOKENS = 150


def _detect_obligation(text: str) -> str:
    for pattern, party in _OBLIGATION_MAP:
        if pattern.search(text):
            return party
    return "General"


def _find_defined_terms_in_text(text: str) -> list[str]:
    found = []
    for term in _KNOWN_DEFINED_TERMS:
        if term in text:
            found.append(term)
    return found


def _get_sections_text(pages: list[PageData]) -> str:
    """Return the full text from Section 2 onwards."""
    joined = "\n".join(p.text for p in pages)
    sec2 = re.search(r'Section\s+2[\.\s]', joined, re.IGNORECASE)
    return joined[sec2.start():] if sec2 else joined


def _split_section_into_subsections(
    section_num: str,
    section_title: str,
    section_body: str,
    page_range: list[int],
    prev_chunk_ids: list[str],
) -> list[Chunk]:
    """
    Split one Section into subsection chunks.
    Handles Section 10.2 warranties by grouping 8-10 items per sub-chunk.
    Returns list of Chunk objects with prev/next linkage.
    """
    # Detect subsections inside section body
    sub_matches = list(_SUBSEC_PATTERN.finditer(section_body))

    # If no subsections, treat the whole section as one chunk
    if not sub_matches:
        chunk_id = f"sec_{section_num}_main"
        breadcrumb = f"[IHDA Agreement | Section {section_num} — {section_title} | Obligations of: {_detect_obligation(section_body)}]"
        page_content = f"{breadcrumb}\n\n{section_body.strip()}"

        # Split if over MAX_TOKENS
        return _maybe_split_oversized(
            page_content, section_num, section_title, chunk_id, page_range,
        )

    chunks: list[Chunk] = []
    prev_tail = ""

    for idx, m in enumerate(sub_matches):
        sub_major = m.group(1)
        sub_minor = m.group(2)
        sub_title = m.group(3).strip()
        subsec_num = f"{sub_major}.{sub_minor}"
        body_start = m.start()
        body_end = sub_matches[idx + 1].start() if idx + 1 < len(sub_matches) else len(section_body)
        raw_body = section_body[body_start:body_end].strip()

        # Special handling: Section 10.2 warranties
        is_warranty_section = section_num == "10" and sub_minor == "2"
        if is_warranty_section:
            warranty_chunks = _split_warranty_chunk(
                section_num, sub_title, raw_body, page_range, subsec_num
            )
            chunks.extend(warranty_chunks)
            continue

        # Add overlap prefix from previous chunk tail
        overlap = f"[...continued from previous]\n{prev_tail}\n\n" if prev_tail else ""
        breadcrumb = (
            f"[IHDA Agreement | Section {subsec_num} — {sub_title} | "
            f"Obligations of: {_detect_obligation(raw_body)}]"
        )
        page_content = f"{breadcrumb}\n\n{overlap}{raw_body}"
        chunk_id = f"sec_{sub_major}_{sub_minor}_{_slugify(sub_title)}"

        chunks.append(Chunk(
            page_content=page_content,
            metadata={
                "chunk_id": chunk_id,
                "chunk_type": "section",
                "section_number": subsec_num,
                "section_title": sub_title,
                "parent_section": section_num,
                "doc_source": "main",
                "page_range": page_range,
                "obligations_of": _detect_obligation(raw_body),
                "def_terms_present": _find_defined_terms_in_text(raw_body),
                "retrieval_priority": "high",
                "token_count": _token_count(page_content),
                "contains_table": False,
                "prev_chunk_id": chunks[-1].metadata["chunk_id"] if chunks else (prev_chunk_ids[-1] if prev_chunk_ids else ""),
                "next_chunk_id": "",  # filled in after all chunks built
            },
        ))
        prev_tail = _tail_tokens(raw_body, OVERLAP_TOKENS)

    # Back-fill next_chunk_id links
    for i in range(len(chunks) - 1):
        chunks[i].metadata["next_chunk_id"] = chunks[i + 1].metadata["chunk_id"]

    return chunks


def _maybe_split_oversized(
    page_content: str,
    section_num: str,
    section_title: str,
    base_chunk_id: str,
    page_range: list[int],
) -> list[Chunk]:
    """If content exceeds MAX_TOKENS, paragraph-split it."""
    if _token_count(page_content) <= MAX_TOKENS:
        obligation = _detect_obligation(page_content)
        return [Chunk(
            page_content=page_content,
            metadata={
                "chunk_id": base_chunk_id,
                "chunk_type": "section",
                "section_number": section_num,
                "section_title": section_title,
                "parent_section": section_num.split(".")[0],
                "doc_source": "main",
                "page_range": page_range,
                "obligations_of": obligation,
                "def_terms_present": _find_defined_terms_in_text(page_content),
                "retrieval_priority": "high",
                "token_count": _token_count(page_content),
                "contains_table": False,
                "prev_chunk_id": "",
                "next_chunk_id": "",
            },
        )]

    # Split at paragraph boundaries, respecting MAX_TOKENS
    paragraphs = re.split(r'\n\s*\n', page_content)
    chunks: list[Chunk] = []
    current_paragraphs: list[str] = []
    part_num = 1

    def flush() -> None:
        nonlocal part_num
        body = "\n\n".join(current_paragraphs)
        cid = f"{base_chunk_id}_part{part_num}"
        obligation = _detect_obligation(body)
        chunks.append(Chunk(
            page_content=body,
            metadata={
                "chunk_id": cid,
                "chunk_type": "section",
                "section_number": section_num,
                "section_title": section_title,
                "parent_section": section_num.split(".")[0],
                "doc_source": "main",
                "page_range": page_range,
                "obligations_of": obligation,
                "def_terms_present": _find_defined_terms_in_text(body),
                "retrieval_priority": "high",
                "token_count": _token_count(body),
                "contains_table": False,
                "prev_chunk_id": chunks[-1].metadata["chunk_id"] if chunks else "",
                "next_chunk_id": "",
            },
        ))
        part_num += 1
        current_paragraphs.clear()

    for para in paragraphs:
        test = "\n\n".join(current_paragraphs + [para])
        if _token_count(test) > MAX_TOKENS and current_paragraphs:
            flush()
        current_paragraphs.append(para)

    if current_paragraphs:
        flush()

    for i in range(len(chunks) - 1):
        chunks[i].metadata["next_chunk_id"] = chunks[i + 1].metadata["chunk_id"]

    return chunks


def _split_warranty_chunk(
    section_num: str,
    section_title: str,
    body: str,
    page_range: list[int],
    subsec_num: str,
) -> list[Chunk]:
    """
    Section 10.2 has 37 warranties. Group 8–10 per chunk, never fewer than 5.
    """
    # Detect warranty items as lettered/numbered list items
    items = re.split(r'\n(?=\s*(?:[A-Z]\.|(?:\([a-z]\))|\d+\.))', body)
    items = [item.strip() for item in items if item.strip()]
    total = len(items)

    GROUP_SIZE = 9  # aim for 9 items per chunk (between 8 and 10)
    chunks: list[Chunk] = []
    idx = 0
    group_num = 1

    while idx < total:
        remaining = total - idx
        # Avoid creating a tail group smaller than 5
        take = GROUP_SIZE
        if remaining - GROUP_SIZE < 5 and remaining <= GROUP_SIZE + 5:
            take = remaining  # absorb into current group

        group = items[idx: idx + take]
        start_item = idx + 1
        end_item = idx + len(group)
        chunk_id = f"sec_{section_num}_2_warranties_{start_item}_{end_item}"
        label = f"warranties #{start_item}–{end_item} of {total}"
        breadcrumb = f"[IHDA Agreement | Section {subsec_num} — {section_title} | {label}]"
        page_content = f"{breadcrumb}\n\n" + "\n\n".join(group)

        chunks.append(Chunk(
            page_content=page_content,
            metadata={
                "chunk_id": chunk_id,
                "chunk_type": "section",
                "section_number": subsec_num,
                "section_title": section_title,
                "parent_section": section_num,
                "doc_source": "main",
                "page_range": page_range,
                "obligations_of": "Lender",
                "def_terms_present": _find_defined_terms_in_text(page_content),
                "retrieval_priority": "high",
                "token_count": _token_count(page_content),
                "contains_table": False,
                "warranty_range": f"{start_item}-{end_item}",
                "warranty_total": total,
                "prev_chunk_id": chunks[-1].metadata["chunk_id"] if chunks else "",
                "next_chunk_id": "",
            },
        ))
        idx += take
        group_num += 1

    for i in range(len(chunks) - 1):
        chunks[i].metadata["next_chunk_id"] = chunks[i + 1].metadata["chunk_id"]

    return chunks


def _approximate_page_range(text_start: int, text_end: int, pages: list[PageData]) -> list[int]:
    """Map a character range in the joined text back to approximate page numbers."""
    offset = 0
    first_page = last_page = 1
    found_first = False
    for p in pages:
        length = len(p.text) + 1  # +1 for the newline separator
        if not found_first and offset + length > text_start:
            first_page = p.page_number
            found_first = True
        if found_first and offset < text_end:
            last_page = p.page_number
        offset += length
    return [first_page, last_page]


def _layer2_section_chunks(pages: list[PageData]) -> list[Chunk]:
    joined = "\n".join(p.text for p in pages)
    sec2_match = re.search(r'Section\s+2[\.\s]', joined, re.IGNORECASE)
    if not sec2_match:
        return []

    sections_text = joined[sec2_match.start():]
    section_matches = list(_SECTION_PATTERN.finditer(sections_text))

    all_chunks: list[Chunk] = []
    prev_chunk_ids: list[str] = []

    for idx, sm in enumerate(section_matches):
        sec_num = sm.group(1)
        sec_title = sm.group(2).strip()
        sec_start = sm.start()
        sec_end = section_matches[idx + 1].start() if idx + 1 < len(section_matches) else len(sections_text)
        sec_body = sections_text[sec_start:sec_end]

        # Approx page range
        abs_start = sec2_match.start() + sec_start
        abs_end = sec2_match.start() + sec_end
        page_range = _approximate_page_range(abs_start, abs_end, pages)

        new_chunks = _split_section_into_subsections(
            sec_num, sec_title, sec_body, page_range, prev_chunk_ids
        )
        all_chunks.extend(new_chunks)
        prev_chunk_ids = [c.metadata["chunk_id"] for c in new_chunks]

    return all_chunks


# ---------------------------------------------------------------------------
# Layer 3 — Table Chunks
# ---------------------------------------------------------------------------

# Recapture table ordinals (pdfplumber will give us raw cell text)
_ORDINAL_TO_YEAR = {
    "first": "Year 1", "second": "Year 2", "third": "Year 3",
    "fourth": "Year 4", "fifth": "Year 5", "sixth": "Year 6",
    "seventh": "Year 7", "eighth": "Year 8", "ninth": "Year 9",
}

_EXAMPLE_PATTERN = re.compile(
    r'Example\s+[A-Z]\s*:', re.IGNORECASE
)


def _is_recapture_table(table: list[list[str]]) -> bool:
    """Detect if a table is the recapture percentage table."""
    flat = " ".join(cell.lower() for row in table for cell in row if cell)
    return "percentage" in flat and any(ord_ in flat for ord_ in _ORDINAL_TO_YEAR)


def _serialize_recapture_table(
    table: list[list[str]],
    examples_text: str,
    page_num: int,
    is_duplicate: bool,
) -> Chunk:
    # Normalise table: strip blanks, ensure 2 columns
    data_rows = [
        [cell.strip() for cell in row]
        for row in table
        if any(cell.strip() for cell in row)
    ]
    if not data_rows:
        data_rows = [["—", "—"]]

    # Try to detect header row
    if len(data_rows) > 1 and any("year" in c.lower() or "percent" in c.lower() for c in data_rows[0]):
        headers = data_rows[0]
        rows = data_rows[1:]
    else:
        headers = ["Year After Purchase", "Percentage"]
        rows = data_rows

    # Pad rows to 2 cols
    rows = [(r + [""] * 2)[:2] for r in rows]

    # --- markdown format
    md = _build_markdown_table(headers, rows)

    # --- pipe format
    pipe_parts = []
    for r in rows:
        year_label = _ORDINAL_TO_YEAR.get(r[0].lower(), r[0])
        pipe_parts.append(f"{year_label}={r[1]}")
    pipe_parts.append("After 9yr=0%")
    pipe_str = "PIPE: " + " | ".join(pipe_parts)

    # --- prose format
    prose_items = []
    for r in rows:
        year_label = _ORDINAL_TO_YEAR.get(r[0].lower(), r[0])
        prose_items.append(f"{year_label} is {r[1]}")
    prose_str = (
        "PROSE: Recapture percentages by year of sale after purchase: "
        + ", ".join(prose_items)
        + ". Maximum recapture is always capped at 50% of gain on sale."
    )

    # --- examples block
    examples_block = f"EXAMPLES:\n{examples_text.strip()}" if examples_text.strip() else ""

    breadcrumb = "[IHDA Agreement | Recapture Percentage Table | Section 4 / Exhibit A]"
    page_content = (
        f"{breadcrumb}\n\n"
        f"MARKDOWN:\n{md}\n\n"
        f"{pipe_str}\n\n"
        f"{prose_str}"
    )
    if examples_block:
        page_content += f"\n\n{examples_block}"

    metadata: dict[str, Any] = {
        "chunk_id": "table_recapture_duplicate" if is_duplicate else "table_recapture_main",
        "chunk_type": "table",
        "section": "4",
        "doc_source": "main",
        "page_range": [page_num, page_num],
        "contains_table": True,
        "table_name": "recapture_percentages",
        "serialization_formats": ["markdown", "pipe", "prose"],
        "has_examples": bool(examples_text.strip()),
        "retrieval_priority": "high",
        "token_count": _token_count(page_content),
        "is_duplicate": is_duplicate,
    }
    if is_duplicate:
        metadata["canonical_chunk_id"] = "table_recapture_main"

    return Chunk(page_content=page_content, metadata=metadata)


def _extract_example_text_near_table(page_text: str) -> str:
    """Pull all 'Example X:' paragraphs following a table on the same page."""
    matches = list(_EXAMPLE_PATTERN.finditer(page_text))
    if not matches:
        return ""
    start = matches[0].start()
    return page_text[start:].strip()


def _layer3_table_chunks(pages: list[PageData]) -> list[Chunk]:
    chunks: list[Chunk] = []
    recapture_seen = False

    for page in pages:
        for table in page.tables:
            if not _is_recapture_table(table):
                continue
            examples = _extract_example_text_near_table(page.text)
            is_dup = recapture_seen  # second occurrence = duplicate
            chunk = _serialize_recapture_table(table, examples, page.page_number, is_dup)
            chunks.append(chunk)
            recapture_seen = True

    return chunks


# ---------------------------------------------------------------------------
# SectionAwareSplitter — the main public interface
# ---------------------------------------------------------------------------

class SectionAwareSplitter:
    """
    Orchestrates all 3 chunking layers from a PDF path.
    Call split(pdf_path) to get back a list[Chunk].
    """

    def split(self, pdf_path: Path) -> list[Chunk]:
        pages = _extract_all_pages(pdf_path)

        layer1 = _layer1_definition_chunks(pages)
        layer2 = _layer2_section_chunks(pages)
        layer3 = _layer3_table_chunks(pages)

        all_chunks = layer1 + layer2 + layer3
        print(
            f"[chunker] Layer1={len(layer1)} defs | "
            f"Layer2={len(layer2)} sections | "
            f"Layer3={len(layer3)} tables | "
            f"Total={len(all_chunks)}"
        )
        return all_chunks
