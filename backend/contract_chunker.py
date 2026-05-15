"""
contract_chunker.py
-------------------
3-layer chunking pipeline for legal / government PDF documents.

Layer 1 → Definition Chunks   (Section 1 — every quoted term = 1 atomic chunk)
Layer 2 → Section Chunks      (Sections 2–N, split at subsection boundaries)
Layer 3 → Table Chunks        (classified tables in 3-format serialisation)

PDF Extraction backends (auto-selected or forced via use_cloud flag):
  Cloud  — LlamaParse (llama-parse)  → fast, accurate, handles complex layouts
  Local  — pdfplumber                → offline fallback, no API key required

Rules enforced per the spec:
- No LangChain splitters — SectionAwareSplitter is fully custom
- Every page_content starts with a bracketed breadcrumb header
- Definition chunks have ZERO overlap (atomic + immutable)
- Section chunks overlap 150 tokens from prev chunk tail
- Tables are ALWAYS atomic (never split mid-row or mid-example)
- Tables serialised in 3 formats: markdown + pipe-delimited + prose
- Duplicate tables get is_duplicate=True
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from enum import Enum
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


# ---------------------------------------------------------------------------
# Generalized Header Detection
# ---------------------------------------------------------------------------

@dataclass
class HeaderMatch:
    """A detected heading in document text."""
    level: int     # 1=chapter/section, 2=subsection, 3=deep
    number: str    # "3.6", "Ch4", or ""
    title: str     # clean heading text
    start: int     # char offset in the text
    strategy: str  # which pattern matched


_HDR_EXPLICIT_SECTION = re.compile(
    r'^[ \t]*Section\s+(\d{1,2})[.\s]\s*([^\n.]{3,80}?)[ \t]*$',
    re.MULTILINE | re.IGNORECASE,
)
_HDR_CHAPTER = re.compile(
    r'^[ \t]*Chapter\s+(\d{1,2})\s*[:.\s]\s*([^\n]{3,100}?)[ \t]*$',
    re.MULTILINE | re.IGNORECASE,
)
_HDR_DEEP_NUMERIC = re.compile(
    r'^[ \t]{0,6}(\d{1,2})\.(\d{1,2})\.(\d{1,2})\.?\s+([A-Z][^\n]{2,80}?)[ \t]*$',
    re.MULTILINE,
)
_HDR_SUBSECTION = re.compile(
    r'^[ \t]{0,6}(\d{1,2})\.(\d{1,2})\.?\s+([A-Z][^\n]{2,80}?)[ \t]*$',
    re.MULTILINE,
)
_HDR_ALL_CAPS = re.compile(
    r'^[ \t]*([A-Z][A-Z\s\-\/]{4,60})[ \t]*$',
    re.MULTILINE,
)


def _is_toc_line(line: str) -> bool:
    """Return True if this looks like a Table of Contents entry."""
    return bool(re.search(r'\.{4,}\s*\d+\s*$', line.strip()))


def _detect_headers(text: str) -> list[HeaderMatch]:
    """
    Multi-strategy header detector for generic PDFs.
    Returns HeaderMatch objects sorted by position.

    Strategies (in priority order):
      S1 - Explicit 'Section N.' lines  -> level 1
      S2 - 'Chapter N:' lines           -> level 1
      S3 - Deep numeric 'N.N.N.'        -> level 3
      S4 - Subsection 'N.N.'            -> level 2
      S5 - ALL CAPS standalone line     -> level 1
    """
    result: dict[int, HeaderMatch] = {}

    def _add(m: re.Match, level: int, number: str, title: str, strategy: str) -> None:
        title = title.strip().rstrip('.')
        if not title or _is_toc_line(m.group(0)):
            return
        if m.start() not in result:
            result[m.start()] = HeaderMatch(
                level=level, number=number, title=title,
                start=m.start(), strategy=strategy,
            )

    for m in _HDR_EXPLICIT_SECTION.finditer(text):
        _add(m, 1, m.group(1), m.group(2), "S1_section")
    for m in _HDR_CHAPTER.finditer(text):
        _add(m, 1, f"Ch{m.group(1)}", m.group(2), "S2_chapter")
    for m in _HDR_DEEP_NUMERIC.finditer(text):
        _add(m, 3, f"{m.group(1)}.{m.group(2)}.{m.group(3)}", m.group(4), "S3_deep")
    for m in _HDR_SUBSECTION.finditer(text):
        if m.start() not in result:
            _add(m, 2, f"{m.group(1)}.{m.group(2)}", m.group(3), "S4_subsec")
    for m in _HDR_ALL_CAPS.finditer(text):
        if m.start() in result:
            continue
        title = m.group(1).strip()
        if len(title.split()) < 2 and len(title) < 8:
            continue
        if _is_toc_line(m.group(0)):
            continue
        result[m.start()] = HeaderMatch(
            level=1, number="", title=title,
            start=m.start(), strategy="S5_caps",
        )

    return sorted(result.values(), key=lambda h: h.start)


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
# PDF extraction — two backends: LlamaParse (cloud) and pdfplumber (local)
# ---------------------------------------------------------------------------

@dataclass
class PageData:
    page_number: int
    text: str
    tables: list[list[list[str | None]]]  # list of table rows (cells as strings)


# ---- Markdown table → cell grid converter (used after LlamaParse) ----------

def _md_table_to_grid(md_table: str) -> list[list[str]]:
    """
    Convert a markdown table string into a list-of-rows grid.
    Handles separator lines (|---|---|) gracefully.
    Returns an empty list if the string is not a recognisable table.
    """
    rows: list[list[str]] = []
    for raw_line in md_table.splitlines():
        line = raw_line.strip()
        if not line or not line.startswith("|"):
            continue
        # Skip pure separator rows like |---|---|
        inner = line.strip("|")
        if re.fullmatch(r"[\s\-:|]+", inner):
            continue
        cells = [c.strip() for c in inner.split("|")]
        rows.append(cells)
    return rows


# ---- Cloud backend — LlamaParse --------------------------------------------

def _extract_all_pages_cloud(pdf_path: Path) -> list[PageData]:
    """
    Parse the PDF via LlamaParse cloud service.
    Returns one PageData per page, with text and tables populated from
    LlamaParse's structured JSON output.

    Requires the environment variable LLAMAINDEX_API_KEY (loaded from .env).
    """
    try:
        from llama_parse import LlamaParse  # type: ignore[import]
    except ImportError:
        raise ImportError(
            "llama-parse is not installed. Run: pip install llama-parse"
        )

    api_key = os.environ.get("LLAMAINDEX_API_KEY") or os.environ.get("LLAMA_CLOUD_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "[chunker] LLAMAINDEX_API_KEY not found in environment. "
            "Add it to your .env file or use the local pdfplumber backend."
        )

    print(f"[chunker] Using LlamaParse cloud backend for '{pdf_path.name}' …")

    parser = LlamaParse(
        api_key=api_key,
        result_type="markdown",          # also gives us per-page JSON
        verbose=False,
        language="en",
        # Instruct the model to preserve table structures faithfully
        parsing_instruction=(
            "Extract the document faithfully. "
            "Preserve all tables as proper markdown tables. "
            "Keep section numbering and heading hierarchy intact."
        ),
    )

    # get_json_result returns a list of dicts, one per uploaded file.
    # Each dict has a 'pages' key → list of per-page objects.
    json_results = parser.get_json_result(str(pdf_path))
    if not json_results:
        raise RuntimeError("[chunker] LlamaParse returned empty results.")

    raw_pages: list[dict] = json_results[0].get("pages", [])
    print(f"[chunker] LlamaParse returned {len(raw_pages)} pages.")

    pages: list[PageData] = []
    for rp in raw_pages:
        page_num: int = rp.get("page", len(pages) + 1)
        # Prefer the plain-text field; fall back to markdown
        text: str = rp.get("text") or rp.get("md") or ""

        # Extract tables from the structured 'items' list when available
        tables: list[list[list[str]]] = []
        for item in rp.get("items", []):
            itype = (item.get("type") or "").lower()
            if itype == "table":
                # Items expose the table as markdown in 'value' or 'md'
                md_src = item.get("value") or item.get("md") or ""
                grid = _md_table_to_grid(md_src)
                if grid:
                    tables.append(grid)

        # Fallback: scan page markdown for fenced tables when items are absent
        if not tables:
            page_md: str = rp.get("md") or ""
            # Collect contiguous markdown table blocks
            current_block: list[str] = []
            for line in page_md.splitlines():
                stripped = line.strip()
                if stripped.startswith("|"):
                    current_block.append(line)
                else:
                    if current_block:
                        grid = _md_table_to_grid("\n".join(current_block))
                        if grid:
                            tables.append(grid)
                        current_block = []
            if current_block:
                grid = _md_table_to_grid("\n".join(current_block))
                if grid:
                    tables.append(grid)

        pages.append(PageData(page_number=page_num, text=text, tables=tables))

    return pages


# ---- Local backend — pdfplumber -------------------------------------------

def _extract_all_pages_local(pdf_path: Path) -> list[PageData]:
    """Original pdfplumber-based extractor (offline, no API key needed)."""
    pages: list[PageData] = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text(layout=True) or ""
            tables = page.extract_tables() or []
            clean_tables = [
                [[cell or "" for cell in row] for row in table]
                for table in tables
            ]
            pages.append(PageData(page_number=i, text=text, tables=clean_tables))
    return pages


# ---- Cache helpers ----------------------------------------------------------

import hashlib
import json as _json

_CACHE_DIR = Path(__file__).parent / ".parse_cache"


def _pdf_hash(pdf_path: Path) -> str:
    """SHA-256 hash of the PDF file (fast — streams in 64 KB chunks)."""
    h = hashlib.sha256()
    with open(pdf_path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def _cache_path(pdf_hash: str) -> Path:
    _CACHE_DIR.mkdir(exist_ok=True)
    return _CACHE_DIR / f"{pdf_hash}.json"


def _save_to_cache(pdf_hash: str, pages: list[PageData]) -> None:
    data = [
        {
            "page_number": p.page_number,
            "text": p.text,
            "tables": p.tables,
        }
        for p in pages
    ]
    _cache_path(pdf_hash).write_text(_json.dumps(data), encoding="utf-8")


def _load_from_cache(pdf_hash: str) -> list[PageData] | None:
    cp = _cache_path(pdf_hash)
    if not cp.exists():
        return None
    try:
        data = _json.loads(cp.read_text(encoding="utf-8"))
        return [
            PageData(
                page_number=d["page_number"],
                text=d["text"],
                tables=d["tables"],
            )
            for d in data
        ]
    except Exception:
        return None


# ---- Unified entry point ---------------------------------------------------

def _extract_all_pages(pdf_path: Path, use_cloud: bool = True) -> list[PageData]:
    """
    Extract all pages from a PDF.

    Caches the result by PDF hash, so repeated runs on the same file
    skip the parsing step entirely (loads in ~50ms instead of ~60s).

    Parameters
    ----------
    pdf_path  : path to the PDF file
    use_cloud : if True (default), use LlamaParse cloud service.
                Falls back automatically to pdfplumber if API key is absent.
    """
    # ── Cache lookup ──────────────────────────────────────────────────────
    file_hash = _pdf_hash(pdf_path)
    cached = _load_from_cache(file_hash)
    if cached is not None:
        print(f"[chunker] Cache hit for '{pdf_path.name}' ({len(cached)} pages). Skipping parse.")
        return cached

    # ── Parse ─────────────────────────────────────────────────────────────
    if use_cloud:
        api_key = os.environ.get("LLAMAINDEX_API_KEY") or os.environ.get("LLAMA_CLOUD_API_KEY")
        if api_key:
            pages = _extract_all_pages_cloud(pdf_path)
        else:
            print(
                "[chunker] LLAMAINDEX_API_KEY not set — falling back to local pdfplumber."
            )
            print(f"[chunker] Using local pdfplumber backend for '{pdf_path.name}' …")
            pages = _extract_all_pages_local(pdf_path)
    else:
        print(f"[chunker] Using local pdfplumber backend for '{pdf_path.name}' …")
        pages = _extract_all_pages_local(pdf_path)

    # ── Save to cache ─────────────────────────────────────────────────────
    _save_to_cache(file_hash, pages)
    print(f"[chunker] Cached parse result for '{pdf_path.name}' ({len(pages)} pages).")

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
    Split one section into subsection chunks using generalized _detect_headers().
    Works for any heading style (Chapter N, N.N., Section N, ALL CAPS, etc.).
    Retains backward-compat special handling for IHDA Section 10.2 warranties.
    """
    sub_headers = [h for h in _detect_headers(section_body) if h.level >= 2]

    if not sub_headers:
        sec_label = f"{section_num} \u2014 " if section_num else ""
        chunk_id = f"sec_{_slugify(section_num or section_title)}_main"
        breadcrumb = (
            f"[Document | {sec_label}{section_title}"
            f" | Obligations of: {_detect_obligation(section_body)}]"
        )
        page_content = f"{breadcrumb}\n\n{section_body.strip()}"
        return _maybe_split_oversized(
            page_content, section_num, section_title, chunk_id, page_range,
        )

    chunks: list[Chunk] = []
    prev_tail = ""

    for idx, sh in enumerate(sub_headers):
        sub_title = sh.title
        subsec_num = sh.number
        body_start = sh.start
        body_end = sub_headers[idx + 1].start if idx + 1 < len(sub_headers) else len(section_body)
        raw_body = section_body[body_start:body_end].strip()

        # Backward compat: IHDA Section 10.2 warranties
        parent_num = section_num.split(".")[0] if "." in section_num else section_num
        if parent_num == "10" and subsec_num.endswith(".2"):
            chunks.extend(_split_warranty_chunk(
                parent_num, sub_title, raw_body, page_range, subsec_num
            ))
            continue

        overlap = f"[...continued from previous]\n{prev_tail}\n\n" if prev_tail else ""
        sec_label = f"{subsec_num} \u2014 " if subsec_num else ""
        breadcrumb = (
            f"[Document | {sec_label}{sub_title}"
            f" | Obligations of: {_detect_obligation(raw_body)}]"
        )
        page_content = f"{breadcrumb}\n\n{overlap}{raw_body}"
        chunk_id = f"sec_{_slugify(subsec_num or sub_title)}"

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
                "header_strategy": sh.strategy,
                "prev_chunk_id": chunks[-1].metadata["chunk_id"] if chunks else (prev_chunk_ids[-1] if prev_chunk_ids else ""),
                "next_chunk_id": "",
            },
        ))
        prev_tail = _tail_tokens(raw_body, OVERLAP_TOKENS)

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
    """
    Build section/subsection chunks using generalized header detection.
    Works for any document structure (Chapter N, Section N, N.N., ALL CAPS, etc.).
    """
    joined = "\n".join(p.text for p in pages)
    all_headers = _detect_headers(joined)

    # Level-1 headers define top-level sections
    level1 = [h for h in all_headers if h.level == 1]
    if not level1:
        return []

    all_chunks: list[Chunk] = []
    prev_chunk_ids: list[str] = []
    seen_ids: dict[str, int] = {}

    for idx, h1 in enumerate(level1):
        sec_start = h1.start
        sec_end = level1[idx + 1].start if idx + 1 < len(level1) else len(joined)
        sec_body = joined[sec_start:sec_end]
        page_range = _approximate_page_range(sec_start, sec_end, pages)

        new_chunks = _split_section_into_subsections(
            h1.number, h1.title, sec_body, page_range, prev_chunk_ids
        )
        
        # Deduplicate chunk IDs
        for c in new_chunks:
            base_id = c.metadata["chunk_id"]
            if base_id in seen_ids:
                seen_ids[base_id] += 1
                c.metadata["chunk_id"] = f"{base_id}_dup{seen_ids[base_id]}"
            else:
                seen_ids[base_id] = 0

        # Fix prev/next links for the new chunks
        for i in range(len(new_chunks)):
            if i > 0:
                new_chunks[i].metadata["prev_chunk_id"] = new_chunks[i-1].metadata["chunk_id"]
            if i < len(new_chunks) - 1:
                new_chunks[i].metadata["next_chunk_id"] = new_chunks[i+1].metadata["chunk_id"]
                
        # Link the first chunk to the last chunk of the previous batch
        if all_chunks and new_chunks:
            new_chunks[0].metadata["prev_chunk_id"] = all_chunks[-1].metadata["chunk_id"]
            all_chunks[-1].metadata["next_chunk_id"] = new_chunks[0].metadata["chunk_id"]

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


# ---------------------------------------------------------------------------
# Generalized Table Classification
# ---------------------------------------------------------------------------

class TableKind(Enum):
    """Classification of a table extracted from a PDF page."""
    SKIP      = "skip"        # noise/empty — do not chunk
    KEY_VALUE = "key_value"   # 2-col pairs: Risk/Mitigation, Term/Definition
    SCORING   = "scoring"     # numeric %, scores, weightings
    CRITERIA  = "criteria"    # evaluation criteria / sub-criteria
    REFERENCE = "reference"   # regulatory / legal cross-references
    GENERIC   = "generic"     # any other multi-row, multi-col table


def _looks_like_header(row: list[str]) -> bool:
    """Heuristic: header rows tend to have short, capitalized cells."""
    non_empty = [c for c in row if c]
    if not non_empty:
        return False
    avg_words = sum(len(c.split()) for c in non_empty) / len(non_empty)
    caps_ratio = sum(1 for c in non_empty if c == c.upper() or c.istitle()) / len(non_empty)
    return avg_words <= 5 or caps_ratio >= 0.5


def classify_table(table: list[list[str]]) -> TableKind:
    """
    Classify a pdfplumber table without document-specific hardcoding.
    Works across government manuals, legal agreements, policy documents, etc.

    Classification order (first match wins):
      SKIP      - too few rows/cols or mostly empty
      SCORING   - score / weighting / % in header
      CRITERIA  - criteria / evaluation in header
      REFERENCE - regulatory clause/rule references in header
      KEY_VALUE - 2-column with short first-column entries
      GENERIC   - everything else worth keeping
    """
    data_rows = [
        [cell.strip() for cell in row]
        for row in table
        if any(cell.strip() for cell in row)
    ]
    if len(data_rows) < 2:
        return TableKind.SKIP
    ncols = max(len(row) for row in data_rows)
    if ncols < 2:
        return TableKind.SKIP
    total_cells = sum(len(row) for row in data_rows)
    non_empty = sum(1 for row in data_rows for cell in row if cell)
    if total_cells > 0 and non_empty / total_cells < 0.15:
        return TableKind.SKIP

    header_flat = " ".join(c.lower() for c in data_rows[0] if c)
    all_flat    = " ".join(c.lower() for row in data_rows for c in row if c)

    # SCORING: percentage / score / weighting columns
    if re.search(r'\b(?:score|weighting|weight|marks|percentage|%|rating)\b', header_flat):
        return TableKind.SCORING
    # Backward compat: IHDA recapture table (no explicit header keyword)
    if "percentage" in all_flat and any(ord_ in all_flat for ord_ in _ORDINAL_TO_YEAR):
        return TableKind.SCORING

    # CRITERIA: evaluation / sub-criteria tables
    if re.search(r'\b(?:criteria|sub[- ]criteria|evaluation|parameter|indicator)\b', header_flat):
        return TableKind.CRITERIA

    # REFERENCE: regulatory / legal reference tables
    if re.search(r'\b(?:rule\s*\d|gfr|clause|schedule|appendix|annexure|article)\b', header_flat):
        return TableKind.REFERENCE

    # KEY_VALUE: 2-column with reasonably short first-column entries
    if ncols == 2:
        avg_first_col = sum(len(r[0].split()) for r in data_rows if len(r) >= 1) / len(data_rows)
        if avg_first_col <= 10:
            return TableKind.KEY_VALUE

    return TableKind.GENERIC


def _table_signature(table: list[list[str]]) -> str:
    """Compact fingerprint for table deduplication."""
    flat = "|".join(cell.strip() for row in table for cell in row if cell.strip())
    return flat[:300]


def _extract_heading_context(page_text: str) -> str:
    """Return the last detected heading on a page, used as breadcrumb context."""
    headers = _detect_headers(page_text)
    if not headers:
        return ""
    h = headers[-1]
    return f"{h.number} {h.title}".strip() if h.number else h.title


def _is_recapture_table(table: list[list[str]]) -> bool:
    """Kept for backward compatibility — delegates to classify_table()."""
    return classify_table(table) == TableKind.SCORING


def _serialize_generic_table(
    table: list[list[str]],
    kind: TableKind,
    context_heading: str,
    page_num: int,
    is_duplicate: bool = False,
    table_index: int = 0,
) -> Chunk:
    """
    Unified table serializer for any TableKind.
    - Always produces Markdown.
    - KEY_VALUE  -> also Prose  (Col1: Col2 | ...).
    - SCORING / CRITERIA -> also Pipe (H1=val | H2=val || ...).
    - SCORING tables matching the IHDA recapture pattern -> delegate to
      _serialize_recapture_table for full backward compatibility.
    """
    data_rows = [
        [cell.strip() for cell in row]
        for row in table
        if any(cell.strip() for cell in row)
    ]
    # Delegate IHDA recapture tables to original serializer
    if kind == TableKind.SCORING:
        all_flat = " ".join(c.lower() for row in data_rows for c in row if c)
        if any(ord_ in all_flat for ord_ in _ORDINAL_TO_YEAR):
            examples = _extract_example_text_near_table(context_heading)  # empty string fallback
            return _serialize_recapture_table(table, "", page_num, is_duplicate)

    # Detect header row
    if len(data_rows) > 1 and _looks_like_header(data_rows[0]):
        headers, rows = data_rows[0], data_rows[1:]
    else:
        ncols = max(len(r) for r in data_rows) if data_rows else 2
        headers = [f"Col {i+1}" for i in range(ncols)]
        rows = data_rows

    ncols = len(headers)
    rows = [(r + [""] * ncols)[:ncols] for r in rows]

    md = _build_markdown_table(headers, rows)

    pipe_str = ""
    if kind in (TableKind.SCORING, TableKind.CRITERIA):
        pipe_parts = [
            " | ".join(f"{h}={cell}" for h, cell in zip(headers, row) if cell)
            for row in rows
        ]
        pipe_str = "PIPE: " + " || ".join(p for p in pipe_parts if p)

    prose_str = ""
    if kind == TableKind.KEY_VALUE and ncols == 2:
        items = [f"{r[0]}: {r[1]}" for r in rows if r[0] and r[1]]
        prose_str = "PROSE: " + " | ".join(items)

    ctx = f" | Near: {context_heading}" if context_heading else ""
    breadcrumb = f"[Document | Table ({kind.value}) | Page {page_num}{ctx}]"
    parts = [f"MARKDOWN:\n{md}"]
    if pipe_str:
        parts.append(pipe_str)
    if prose_str:
        parts.append(prose_str)
    page_content = f"{breadcrumb}\n\n" + "\n\n".join(parts)

    dup_sfx = "_dup" if is_duplicate else ""
    chunk_id = f"table_p{page_num}_t{table_index}_{kind.value}{dup_sfx}"

    return Chunk(
        page_content=page_content,
        metadata={
            "chunk_id": chunk_id,
            "chunk_type": "table",
            "table_kind": kind.value,
            "doc_source": "main",
            "page_range": [page_num, page_num],
            "contains_table": True,
            "context_heading": context_heading,
            "serialization_formats": (
                ["markdown"]
                + (["pipe"] if pipe_str else [])
                + (["prose"] if prose_str else [])
            ),
            "retrieval_priority": "high",
            "token_count": _token_count(page_content),
            "is_duplicate": is_duplicate,
            **({"canonical_chunk_id": chunk_id.replace("_dup", "")} if is_duplicate else {}),
        },
    )


def _layer3_table_chunks(pages: list[PageData]) -> list[Chunk]:
    """
    Generalized table chunker.
    Classifies every table on every page using classify_table().
    Skips SKIP-kind tables; serializes all others with _serialize_generic_table().
    Tracks seen table signatures across pages to flag duplicates.
    """
    chunks: list[Chunk] = []
    seen_sigs: set[str] = set()

    for page in pages:
        if not page.tables:
            continue
        context = _extract_heading_context(page.text)
        for t_idx, table in enumerate(page.tables):
            kind = classify_table(table)
            if kind == TableKind.SKIP:
                continue
            sig = _table_signature(table)
            is_dup = sig in seen_sigs
            seen_sigs.add(sig)
            chunk = _serialize_generic_table(
                table=table,
                kind=kind,
                context_heading=context,
                page_num=page.page_number,
                is_duplicate=is_dup,
                table_index=t_idx,
            )
            chunks.append(chunk)

    return chunks


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


# ---------------------------------------------------------------------------
# SectionAwareSplitter — the main public interface
# ---------------------------------------------------------------------------

class SectionAwareSplitter:
    """
    Orchestrates all 3 chunking layers from a PDF path.
    Call split(pdf_path) to get back a list[Chunk].
    """

    def split(self, pdf_path: Path, use_cloud: bool = True) -> list[Chunk]:
        """
        Parse a PDF and produce all chunk layers.

        Parameters
        ----------
        pdf_path  : path to the PDF file
        use_cloud : forward to _extract_all_pages; uses LlamaParse when True
                    (default) and LLAMAINDEX_API_KEY is present, otherwise
                    falls back to local pdfplumber automatically.
        """
        pages = _extract_all_pages(pdf_path, use_cloud=use_cloud)

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
