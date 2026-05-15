"""
contract_feature_extractor.py
------------------------------
Heuristic feature extractor — zero API calls, instant processing.

Extracts structured metadata from contract text chunks using regex patterns,
keyword matching, and simple NLP heuristics. Processes 423 chunks in <2 seconds
on any machine, with no API keys, rate limits, or network calls required.

Extracted fields:
  section_id, chapter, section_title, content_type, obligation,
  responsible_party, action_type, timeline, document_referenced,
  risk_level, keywords, summary
"""

from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any

from contract_chunker import Chunk

# ---------------------------------------------------------------------------
# Stop words for keyword extraction
# ---------------------------------------------------------------------------

_STOP_WORDS = frozenset(
    "a an the is are was were be been being have has had do does did will "
    "would shall should may might can could of in to for on with at by from "
    "as into through during before after above below between under and but or "
    "nor not so yet both either neither each every all any few more most other "
    "some such no only own same than too very just about also back even still "
    "well how what which who whom this that these those it its he she they we "
    "you his her their our your my me him us them i ii iii iv v vi vii viii ix x "
    "ie eg etc per se re vs".split()
)

# ---------------------------------------------------------------------------
# Section ID detection
# ---------------------------------------------------------------------------

_RE_SECTION_ID = re.compile(
    r'(?:^|\n)\s*(\d{1,2}\.\d{1,2}(?:\.\d{1,2})?)\s',
)


def _extract_section_id(text: str) -> str | None:
    m = _RE_SECTION_ID.search(text)
    return m.group(1) if m else None


# ---------------------------------------------------------------------------
# Timeline detection
# ---------------------------------------------------------------------------

_RE_TIMELINE = re.compile(
    r'(\d+)\s*(days?|weeks?|months?|years?|hours?|business\s+days?|calendar\s+days?'
    r'|working\s+days?)',
    re.IGNORECASE,
)


def _extract_timeline(text: str) -> str | None:
    m = _RE_TIMELINE.search(text)
    return f"{m.group(1)} {m.group(2).lower()}" if m else None


# ---------------------------------------------------------------------------
# Obligation detection
# ---------------------------------------------------------------------------

_MANDATORY_KW   = re.compile(r'\b(shall|must|is required to|are required to|obligat)\b', re.I)
_OPTIONAL_KW    = re.compile(r'\b(may|can|is permitted to|at its discretion|optionally)\b', re.I)
_RECOMMENDED_KW = re.compile(r'\b(should|is recommended|it is advisable|ought to)\b', re.I)


def _extract_obligation(text: str) -> str:
    if _MANDATORY_KW.search(text):
        return "Mandatory"
    if _RECOMMENDED_KW.search(text):
        return "Recommended"
    if _OPTIONAL_KW.search(text):
        return "Optional"
    return "Informational"


# ---------------------------------------------------------------------------
# Responsible party detection
# ---------------------------------------------------------------------------

_PARTIES = {
    "Consultant":       re.compile(r'\b(consultant|consultancy firm|consulting firm)\b', re.I),
    "Employer":         re.compile(r'\b(employer|client|procuring entity|contracting authority)\b', re.I),
    "Project Manager":  re.compile(r'\b(project manager|project director|team leader)\b', re.I),
    "Authority":        re.compile(r'\b(authority|government|ministry|department|agency)\b', re.I),
    "Committee":        re.compile(r'\b(committee|evaluation committee|selection committee|panel)\b', re.I),
}


def _extract_responsible_party(text: str) -> list[str]:
    return [name for name, pat in _PARTIES.items() if pat.search(text)]


# ---------------------------------------------------------------------------
# Action type detection
# ---------------------------------------------------------------------------

_ACTIONS = {
    "submit":    re.compile(r'\b(submit|submitting|submission|deliver|furnish)\b', re.I),
    "approve":   re.compile(r'\b(approv|accept|endorse|sanction|consent)\b', re.I),
    "pay":       re.compile(r'\b(pay|payment|remunerat|compensat|reimburse|disburse)\b', re.I),
    "terminate": re.compile(r'\b(terminat|cancel|rescind|revoke)\b', re.I),
    "notify":    re.compile(r'\b(notify|notification|notice|inform|advise)\b', re.I),
    "review":    re.compile(r'\b(review|evaluat|assess|examin|inspect|audit)\b', re.I),
    "prepare":   re.compile(r'\b(prepar|draft|develop|formulate|design)\b', re.I),
    "issue":     re.compile(r'\b(issue|issu(?:ing|ance)|publish|release)\b', re.I),
}


def _extract_action_type(text: str) -> list[str]:
    return [act for act, pat in _ACTIONS.items() if pat.search(text)]


# ---------------------------------------------------------------------------
# Content type detection
# ---------------------------------------------------------------------------

_RE_DEFINITION = re.compile(r'\bmeans\b.*[:;]|\bdefinition\b|\bdefine[ds]?\b', re.I)
_RE_PROCEDURE  = re.compile(r'\bprocedure\b|\bprocess\b|\bstep\s+\d|\bstage\s+\d', re.I)
_RE_HEADING    = re.compile(r'^[A-Z][A-Z\s\-]{4,60}$', re.M)


def _extract_content_type(text: str, metadata: dict) -> str:
    chunk_type = metadata.get("chunk_type", "")
    if chunk_type == "definition":
        return "Definition"
    if chunk_type == "table":
        return "Procedure"  # tables are often procedures or criteria lists
    if _RE_DEFINITION.search(text):
        return "Definition"
    if _RE_PROCEDURE.search(text):
        return "Procedure"
    if len(text) < 120 and _RE_HEADING.match(text.strip()):
        return "Heading"
    return "Clause"


# ---------------------------------------------------------------------------
# Risk level detection
# ---------------------------------------------------------------------------

_HIGH_RISK_KW   = re.compile(
    r'\b(penalt|terminat|liquidated damages|breach|default|forfeiture|disqualif'
    r'|blacklist|debar|suspend|void|invalid)\b', re.I,
)
_MEDIUM_RISK_KW = re.compile(
    r'\b(compliance|compliant|comply|regulation|regulatory|audit|report|deadline'
    r'|liable|liability|warranty|guarantee|insurance|indemnit)\b', re.I,
)


def _extract_risk_level(text: str, content_type: str) -> str:
    if content_type in ("Heading", "Definition"):
        return "None"
    if _HIGH_RISK_KW.search(text):
        return "High"
    if _MEDIUM_RISK_KW.search(text):
        return "Medium"
    return "Low"


# ---------------------------------------------------------------------------
# Document reference detection
# ---------------------------------------------------------------------------

_RE_DOC_REF = re.compile(
    r'\b(Annex|Appendix|Schedule|Attachment|Exhibit|Section|Clause|Article'
    r'|Form|Table|ToR|Terms of Reference|RFP|EOI|LoI|Contract Agreement'
    r'|General Conditions|Special Conditions|Bid Document)\s*[A-Z0-9\-]*\b',
    re.I,
)


def _extract_document_referenced(text: str) -> list[str]:
    matches = _RE_DOC_REF.findall(text)
    # De-duplicate while preserving order
    seen: set[str] = set()
    result: list[str] = []
    for m in matches:
        m_clean = m.strip()
        if m_clean.lower() not in seen:
            seen.add(m_clean.lower())
            result.append(m_clean)
    return result[:5]


# ---------------------------------------------------------------------------
# Keyword extraction (TF-based, no external dependencies)
# ---------------------------------------------------------------------------

def _extract_keywords(text: str, top_n: int = 5) -> list[str]:
    """Extract top-N keywords by term frequency, filtering stop words."""
    words = re.findall(r'[a-z]{3,}', text.lower())
    filtered = [w for w in words if w not in _STOP_WORDS]
    if not filtered:
        return []
    counts = Counter(filtered)
    return [word for word, _ in counts.most_common(top_n)]


# ---------------------------------------------------------------------------
# Summary extraction (first meaningful sentence)
# ---------------------------------------------------------------------------

_RE_SENTENCE = re.compile(r'[A-Z][^.!?\n]{15,200}[.!?]')


def _extract_summary(text: str) -> str:
    """Return the first meaningful sentence as a summary."""
    # Skip bracketed headers like [Section 3.1 | ...]
    clean = re.sub(r'^\[.*?\]\s*', '', text.strip())
    m = _RE_SENTENCE.search(clean)
    if m:
        return m.group(0).strip()
    # Fallback: first 120 chars
    return clean[:120].strip() + "…" if len(clean) > 120 else clean.strip()


# ---------------------------------------------------------------------------
# Chapter detection (from chunk metadata or text heuristics)
# ---------------------------------------------------------------------------

_RE_CHAPTER = re.compile(
    r'(?:Chapter|CHAPTER)\s+(\d+|[IVXLC]+)\s*[:\-–—.\s]+\s*(.+)',
    re.I,
)


def _extract_chapter(text: str, metadata: dict) -> str | None:
    # Check metadata first (from chunk_id like "sec_ch1_...")
    chunk_id = metadata.get("chunk_id", "")
    ch_match = re.search(r'ch(\d+)', chunk_id)
    if ch_match:
        return f"Chapter {ch_match.group(1)}"
    # Check section numbering
    sec_id = metadata.get("section_number") or _extract_section_id(text)
    if sec_id:
        top = sec_id.split(".")[0]
        return f"Chapter {top}"
    # Check text for explicit chapter references
    m = _RE_CHAPTER.search(text)
    if m:
        return f"Chapter {m.group(1)}: {m.group(2).strip()}"
    return None


# ---------------------------------------------------------------------------
# Section title detection
# ---------------------------------------------------------------------------

_RE_TITLE = re.compile(
    r'(?:^|\n)\s*(?:\d+\.)+\d*\s+([A-Z][A-Za-z\s,\-&/]{3,80}?)(?:\n|$)',
)


def _extract_section_title(text: str) -> str | None:
    m = _RE_TITLE.search(text)
    return m.group(1).strip() if m else None


# ---------------------------------------------------------------------------
# Single-chunk extractor (pure heuristic, ~0.5ms per chunk)
# ---------------------------------------------------------------------------

def _extract_features_for_chunk(chunk: Chunk) -> dict[str, Any]:
    text = chunk.page_content
    meta = chunk.metadata

    section_id   = meta.get("section_number") or _extract_section_id(text)
    content_type = _extract_content_type(text, meta)
    obligation   = _extract_obligation(text)
    risk_level   = _extract_risk_level(text, content_type)

    return {
        "section_id":          section_id,
        "chapter":             _extract_chapter(text, meta),
        "section_title":       _extract_section_title(text),
        "content_type":        content_type,
        "obligation":          obligation,
        "responsible_party":   _extract_responsible_party(text),
        "action_type":         _extract_action_type(text),
        "timeline":            _extract_timeline(text),
        "document_referenced": _extract_document_referenced(text),
        "risk_level":          risk_level,
        "keywords":            _extract_keywords(text),
        "summary":             _extract_summary(text),
    }


# ---------------------------------------------------------------------------
# Feature merger
# ---------------------------------------------------------------------------

def _merge_features(chunk: Chunk, features: dict[str, Any]) -> None:
    """Write extracted feature fields into chunk.metadata."""
    if not features:
        return
    chunk.metadata["features_json"] = json.dumps(features)
    for key in ("content_type", "obligation", "risk_level", "timeline", "summary"):
        val = features.get(key)
        if val is not None:
            chunk.metadata[f"feat_{key}"] = str(val)
    for key in ("responsible_party", "action_type", "document_referenced", "keywords"):
        val = features.get(key)
        if val:
            chunk.metadata[f"feat_{key}"] = json.dumps(val)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def enrich_chunks_with_features(chunks: list[Chunk]) -> None:
    """
    Enrich all chunks with structured features using fast heuristics.
    Zero API calls. Processes 423 chunks in under 2 seconds.

    Mutates chunks in-place by adding 'feat_*' fields to metadata.
    """
    import time

    total = len(chunks)
    print(f"[extractor] Extracting features for {total} chunks (heuristic mode) …")
    t0 = time.perf_counter()

    enriched = 0
    for chunk in chunks:
        features = _extract_features_for_chunk(chunk)
        if features:
            _merge_features(chunk, features)
            enriched += 1

    elapsed = time.perf_counter() - t0
    print(f"[extractor] Done. {enriched}/{total} chunks enriched in {elapsed:.2f}s")
