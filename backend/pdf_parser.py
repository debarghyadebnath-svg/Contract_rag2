import os
import re
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from llama_cloud import LlamaCloud

load_dotenv(Path(__file__).parent.parent / ".env")


def _ensure_llama_cloud_key() -> None:
    if os.environ.get("LLAMA_CLOUD_API_KEY"):
        return
    llamaindex_key = os.environ.get("LLAMAINDEX_API_KEY")
    if llamaindex_key:
        os.environ["LLAMA_CLOUD_API_KEY"] = llamaindex_key


def clean_text(text: str) -> str:
    """Remove control characters that can break JSON/LLMs."""
    # Removes ASCII control characters except \n, \r, \t
    return re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)


def normalize_policy_name(filename: str) -> str:
    stem = Path(filename).stem.upper()
    tokens = re.findall(r"[A-Z0-9]+", stem)
    stopwords = {
        "BROCHURE",
        "POLICY",
        "PLAN",
        "PROSPECTUS",
        "INSURANCE",
        "PDF",
        "ENG",
        "HIN",
        "HINDI",
        "VERSION",
        "DRAFT",
        "FINAL",
        "COPY",
        "UPDATED",
        "REVISED",
        "INCH",
        "X",
        "S",
    }
    normalized = [token for token in tokens if token not in stopwords and not token.isdigit()]

    if not normalized:
        return stem

    if "LIC" in normalized and "JEEVAN" in normalized and "AROGYA" in normalized:
        return "LIC_JEEVAN_AROGYA"

    return "_".join(normalized)


def _page_markdown(page: Any) -> str:
    if isinstance(page, dict):
        value = page.get("markdown") or page.get("text") or page.get("content")
        return value if isinstance(value, str) else ""

    for attr in ("markdown", "text", "content"):
        value = getattr(page, attr, None)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def _parse_with_llamaparse(pdf_path: Path) -> list[dict[str, Any]]:
    _ensure_llama_cloud_key()
    client = LlamaCloud()
    uploaded_file = client.files.create(file=str(pdf_path), purpose="parse")
    result = client.parsing.parse(
        file_id=uploaded_file.id,
        tier="agentic",
        version="latest",
        expand=["markdown"],
    )

    markdown = getattr(result, "markdown", None)
    pages = getattr(markdown, "pages", None)
    if not pages:
        return []

    parsed_pages: list[dict[str, Any]] = []
    for page_number, page in enumerate(pages, start=1):
        text = clean_text(_page_markdown(page)).strip()
        if text:
            parsed_pages.append({"page_number": page_number, "text": text})
    return parsed_pages


def extract_pages(pdf_path: Path) -> list[dict[str, Any]]:
    """
    Parse text page-by-page from a PDF using LlamaParse.
    Returns a list of dicts: {page_number: int, text: str}.
    Blank pages are skipped.
    """
    return _parse_with_llamaparse(pdf_path)
