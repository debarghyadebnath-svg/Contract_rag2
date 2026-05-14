import os
import re
from typing import Any

from langchain_nomic.embeddings import NomicEmbeddings
from langchain_qdrant import QdrantVectorStore, FastEmbedSparse, RetrievalMode
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from qdrant_client import QdrantClient, models

from embedder import EMBEDDING_MODEL, COLLECTION_NAME

RAG_SYSTEM_PROMPT = """\
You are a professional Insurance Analysis Assistant helping Indian users understand and compare their insurance policies. Your goal is to translate technical data into functional "jobs" that the policy performs for the user, summarizing policy details with high clarity and precision.

Follow these strict formatting, logic, and style rules when formulating your response:

1. **Visual Hierarchy:**
   - Use `##` headings to separate different policies or comparison categories.
   - Use `---` horizontal rules to separate major sections.

2. **Highlight Benefits & Constraints:**
   - Use **bold text** to emphasize the core benefit or the most critical constraint within every sentence. This ensures the user can scan the response quickly.

3. **Action-Oriented Language & Clarity:**
   - Describe policy features using active verbs and the present tense (e.g., "The policy **pays ₹X per day**" instead of "There is a benefit of ₹X").
   - Avoid complex jargon. Translate technical terms into professional, easy-to-understand English.

4. **Define the "Job" & Use Comparison Levers:**
   - Categorize the policy based on its core function (e.g., a "Shield" for broad hospital stay coverage vs. a "Specialized Helmet" for specific high-value protection).
   - Point out "friction points" like waiting periods and highlight differences in payout mechanisms.

5. **Address Gaps (Handling Missing Data):**
   - If data is missing (e.g., premium rates or age limits), **clearly state the missing information** and guide the user on how to obtain it (e.g., "Verify the **entry age limit** with an agent to see if you qualify").
   - Do NOT simply say "Information is not available."

6. **The Conclusion:**
   - Always end EVERY response with a `## Bottom Line` section.
   - This section must summarize the primary takeaway using common sense and provide a **direct next step** for the user.

Use ONLY the policy excerpts provided below to form your analysis. If the excerpts are completely unrelated to the question, guide the user on what to look for instead.

Policy Excerpts:
{context}
"""

RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", RAG_SYSTEM_PROMPT),
    ("human", "{question}"),
])


def _build_context(docs: list[Any]) -> str:
    parts = []
    for i, doc in enumerate(docs, 1):
        m = doc.metadata
        header = (
            f"[Source {i} | {m.get('policy_name','?')} | "
            f"{m.get('insurer','?')} | Page {m.get('page_number','?')}]"
        )
        parts.append(f"{header}\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def retrieve_documents(query: str, manual_ids: list[int] | None = None) -> list[Any]:
    """Retrieve the top matching chunks for a question."""
    embeddings = NomicEmbeddings(model=EMBEDDING_MODEL, nomic_api_key=os.environ.get("NOMIC_API_KEY"))
    sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")
    
    client = QdrantClient(
        url=os.environ["QDRANT_URL"],
        api_key=os.environ.get("QDRANT_API_KEY"),
    )
    existing = {c.name for c in client.get_collections().collections}
    if COLLECTION_NAME not in existing:
        return []

    vector_store = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings,
        sparse_embedding=sparse_embeddings,
        retrieval_mode=RetrievalMode.HYBRID,
    )

    search_kwargs: dict[str, Any] = {"k": 15}
    if manual_ids:
        search_kwargs["filter"] = models.Filter(
            should=[
                models.FieldCondition(
                    key="metadata.manual_id",
                    match=models.MatchValue(value=mid),
                )
                for mid in manual_ids
            ]
        )

    retriever = vector_store.as_retriever(search_kwargs=search_kwargs)
    return retriever.invoke(query)


def answer_query(
    query: str,
    manual_ids: list[int] | None = None,
) -> dict[str, Any]:
    """
    Retrieve top-5 chunks from Qdrant and generate an answer with Groq.
    Optionally filter to specific manuals by their SQLite id.
    """
    docs = retrieve_documents(query, manual_ids)
    if not docs:
        return {
            "answer": "I could not find relevant policy content yet. Please confirm your manuals are indexed and in active status, then try again.",
            "sources": [],
            "documents": [],
        }

    # Clean the retrieved documents to prevent JSON parsing errors
    for doc in docs:
        doc.page_content = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', doc.page_content)

    llm = ChatGroq(
        api_key=os.environ["GROQ_API_KEY"],
        model_name="meta-llama/llama-4-scout-17b-16e-instruct",
        temperature=0.5,
    )
    context = _build_context(docs)
    chain = RAG_PROMPT | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "question": query})

    sources = [
        {
            "manual_id": doc.metadata.get("manual_id"),
            "page_number": doc.metadata.get("page_number"),
            "policy_name": doc.metadata.get("policy_name"),
            "insurer": doc.metadata.get("insurer"),
            "category": doc.metadata.get("category"),
            "filename": doc.metadata.get("filename"),
            "snippet": doc.page_content[:400],
        }
        for doc in docs
    ]
    documents = [
        {
            "page_content": doc.page_content,
            "metadata": dict(doc.metadata),
            "id": getattr(doc, "id", doc.metadata.get("_id")),
        }
        for doc in docs
    ]
    return {"answer": answer, "sources": sources, "documents": documents}
