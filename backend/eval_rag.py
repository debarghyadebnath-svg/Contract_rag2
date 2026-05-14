import argparse
import os
from pathlib import Path
from typing import Any

import re
from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv(Path(__file__).parent.parent / ".env")

if os.environ.get("LANGSMITH_API_KEY") and not os.environ.get("LANGCHAIN_API_KEY"):
    os.environ["LANGCHAIN_API_KEY"] = os.environ["LANGSMITH_API_KEY"]

os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
os.environ.setdefault("LANGCHAIN_PROJECT", "insurance-rag-evals")
os.environ.setdefault("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")

from langchain_groq import ChatGroq
from langsmith import Client

import query_engine


class CorrectnessGrade(BaseModel):
    explanation: str = Field(description="Explain your reasoning for the score")
    correct: str = Field(description="Must be exactly the string 'true' or 'false'.")


class RelevanceGrade(BaseModel):
    explanation: str = Field(description="Explain your reasoning for the score")
    relevant: str = Field(description="Must be exactly the string 'true' or 'false'.")


class GroundedGrade(BaseModel):
    explanation: str = Field(description="Explain your reasoning for the score")
    grounded: str = Field(description="Must be exactly the string 'true' or 'false'.")


class RetrievalRelevanceGrade(BaseModel):
    explanation: str = Field(description="Explain your reasoning for the score")
    relevant: str = Field(description="Must be exactly the string 'true' or 'false'.")


def _documents_to_text(documents: list[dict[str, Any]]) -> str:
    text = "\n\n".join(document["snippet"] for document in documents)
    return re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)


def _build_judge(model_name: str, temperature: float = 0.0) -> ChatGroq:
    return ChatGroq(
        api_key=os.environ["GROQ_API_KEY"],
        model_name=model_name,
        temperature=temperature,
    )


def correctness(inputs: dict[str, Any], outputs: dict[str, Any], reference_outputs: dict[str, Any]) -> bool:
    """Grade answer accuracy against a reference answer."""
    judge = _build_judge("meta-llama/llama-4-scout-17b-16e-instruct").with_structured_output(
        CorrectnessGrade
    )
    if "answer" not in outputs:
        return False
        
    prompt = (
        f"QUESTION: {inputs['input']}\n"
        f"GROUND TRUTH ANSWER: {reference_outputs['expected']}\n"
        f"STUDENT ANSWER: {outputs['answer']}"
    )
    grade = judge.invoke(
        [
            {
                "role": "system",
                "content": (
                    "You are a teacher grading a quiz. Grade the student answer based only on "
                    "its factual accuracy relative to the ground truth answer. It is okay if the "
                    "student answer contains more information than the ground truth answer, as long "
                    "as it remains factually accurate. You must output the boolean values true or false without quotes."
                ),
            },
            {"role": "user", "content": prompt},
        ]
    )
    return grade.correct.lower() == "true"


def relevance(inputs: dict[str, Any], outputs: dict[str, Any]) -> bool:
    """Grade whether the answer addresses the user question."""
    judge = _build_judge("meta-llama/llama-4-scout-17b-16e-instruct").with_structured_output(
        RelevanceGrade
    )
    if "answer" not in outputs:
        return False
        
    prompt = f"QUESTION: {inputs['input']}\nSTUDENT ANSWER: {outputs['answer']}"
    grade = judge.invoke(
        [
            {
                "role": "system",
                "content": (
                    "You are a teacher grading a quiz. Ensure the student answer is concise and "
                    "relevant to the question, and helps answer the question. You must output the boolean values true or false without quotes."
                ),
            },
            {"role": "user", "content": prompt},
        ]
    )
    return grade.relevant.lower() == "true"


def groundedness(inputs: dict[str, Any], outputs: dict[str, Any]) -> bool:
    """Grade whether the answer is supported by the retrieved context."""
    judge = _build_judge("meta-llama/llama-4-scout-17b-16e-instruct").with_structured_output(
        GroundedGrade
    )
    if "answer" not in outputs or "sources" not in outputs:
        return False
        
    facts = _documents_to_text(outputs["sources"])
    prompt = f"FACTS: {facts}\nSTUDENT ANSWER: {outputs['answer']}"
    grade = judge.invoke(
        [
            {
                "role": "system",
                "content": (
                    "You are a teacher grading a quiz. Ensure the student answer is grounded in "
                    "the facts and does not include hallucinated information outside the facts. You must output the boolean values true or false without quotes."
                ),
            },


            
            {"role": "user", "content": prompt},
        ]
    )
    return grade.grounded.lower() == "true"


def retrieval_relevance(inputs: dict[str, Any], outputs: dict[str, Any]) -> bool:
    """Grade whether the retrieved documents are relevant to the question."""
    judge = _build_judge("meta-llama/llama-4-scout-17b-16e-instruct").with_structured_output(
        RetrievalRelevanceGrade
    )
    if "sources" not in outputs:
        return False
        
    facts = _documents_to_text(outputs["sources"])
    prompt = f"FACTS: {facts}\nQUESTION: {inputs['input']}"
    grade = judge.invoke(
        [
            {
                "role": "system",
                "content": (
                    "You are a teacher grading a quiz. Decide whether the retrieved facts are "
                    "relevant to the question, even if they contain some unrelated information. You must output the boolean values true or false without quotes."
                ),
            },
            {"role": "user", "content": prompt},
        ]
    )
    return grade.relevant.lower() == "true"

def retrieval_recall(inputs: dict[str, Any], outputs: dict[str, Any], reference_outputs: dict[str, Any]) -> bool:
    """Grade whether the exact expected chunk ID was retrieved."""
    if "documents" not in outputs:
        return False
        
    expected_chunk_id = reference_outputs.get("expected_chunk_id")
    if not expected_chunk_id:
        return True # Skip if this dataset doesn't use chunk IDs
        
    retrieved_chunk_ids = [doc.get("id") for doc in outputs["documents"]]
    return expected_chunk_id in retrieved_chunk_ids


def mrr_at_k(inputs: dict[str, Any], outputs: dict[str, Any], reference_outputs: dict[str, Any]) -> float:
    """Calculate Mean Reciprocal Rank (MRR) for retrieved documents."""
    if "documents" not in outputs:
        return 0.0
        
    expected_chunk_id = reference_outputs.get("expected_chunk_id")
    if not expected_chunk_id:
        return 1.0 # Skip if this dataset doesn't use chunk IDs
        
    retrieved_chunk_ids = [doc.get("id") for doc in outputs["documents"]]
    
    try:
        # Index is 0-based, rank is 1-based
        rank = retrieved_chunk_ids.index(expected_chunk_id) + 1
        return 1.0 / rank
    except ValueError:
        return 0.0


def target(inputs: dict[str, Any]) -> dict[str, Any]:
    return query_engine.answer_query(inputs["input"], inputs.get("manual_ids"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LangSmith evaluations for the insurance RAG app.")
    parser.add_argument("--dataset-name", required=True, help="LangSmith dataset name to evaluate.")
    parser.add_argument(
        "--experiment-prefix",
        default="insurance-rag-eval",
        help="Prefix for the LangSmith experiment name.",
    )
    parser.add_argument(
        "--include-correctness",
        action="store_true",
        help="Include reference-answer correctness if the dataset has reference outputs.",
    )
    args = parser.parse_args()

    client = Client()
    evaluators = [relevance, groundedness, retrieval_relevance, retrieval_recall, mrr_at_k]
    if args.include_correctness:
        evaluators = [correctness, *evaluators]

    results = client.evaluate(
        target,
        data=args.dataset_name,
        evaluators=evaluators,
        experiment_prefix=args.experiment_prefix,
        metadata={
            "app": "insurance-rag",
            "judge_model": "meta-llama/llama-4-scout-17b-16e-instruct",
        },
    )

    print(results)


if __name__ == "__main__":
    main()