import os
from pathlib import Path
from typing import Any

# Naya tareeka (Standard)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore, FastEmbedSparse
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Distance, VectorParams

from pdf_parser import normalize_policy_name

# Using Qwen3-Embedding-0.6B via HuggingFace
EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"
VECTOR_SIZE = 1024
COLLECTION_NAME = "insurance_policies"
MODELS_DIR = Path(__file__).parent / "models"


def _get_qdrant_client() -> QdrantClient:
    url = os.environ["QDRANT_URL"]
    api_key = os.environ.get("QDRANT_API_KEY")
    return QdrantClient(url=url, api_key=api_key)


def _ensure_collection(client: QdrantClient) -> None:
    existing = {c.name for c in client.get_collections().collections}
    if COLLECTION_NAME not in existing:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
            sparse_vectors_config={
                "langchain-sparse": models.SparseVectorParams(
                    index=models.SparseIndexParams(on_disk=False)
                )
            }
        )
    collection_info = client.get_collection(COLLECTION_NAME)
    existing_payload_schema = getattr(collection_info, "payload_schema", {}) or {}

    if "metadata.manual_id" not in existing_payload_schema:
        client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name="metadata.manual_id",
            field_schema=models.PayloadSchemaType.INTEGER,
        )
    if "metadata.policy_name" not in existing_payload_schema:
        client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name="metadata.policy_name",
            field_schema=models.PayloadSchemaType.KEYWORD,
        )


def index_pdf_pages(
    pages: list[dict[str, Any]],
    manual_id: int,
    insurer: str,
    category: str,
    filename: str,
    policy_name: str | None = None,
) -> None:
    """
    Chunk the extracted pages, embed with Qwen3-Embedding-0.6B,
    and upsert into Qdrant with metadata for citation.
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    documents: list[Document] = []
    resolved_policy_name = policy_name or normalize_policy_name(filename)

    for page in pages:
        chunks = splitter.split_text(page["text"])
        for chunk in chunks:
            documents.append(Document(
                page_content=chunk,
                metadata={
                    "manual_id": manual_id,
                    "page_number": page["page_number"],
                    "insurer": insurer,
                    "category": category,
                    "filename": filename,
                    "policy_name": resolved_policy_name,
                },
            ))
    if not documents:
        return

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        cache_folder=str(MODELS_DIR),
        model_kwargs={"trust_remote_code": True}
    )
    sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")
    
    client = _get_qdrant_client()
    _ensure_collection(client)

    vector_store = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings,
        sparse_embedding=sparse_embeddings,
    )
    vector_store.add_documents(documents)


def delete_manual_vectors(manual_id: int) -> None:
    """Remove all Qdrant points that belong to the given manual."""
    client = _get_qdrant_client()
    client.delete(
        collection_name=COLLECTION_NAME,
        points_selector=models.FilterSelector(
            filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="metadata.manual_id",
                        match=models.MatchValue(value=manual_id),
                    )
                ]
            )
        ),
    )
