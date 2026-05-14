import json
import os
from pathlib import Path
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient

load_dotenv(Path(__file__).parent.parent / ".env")

# 1. Define what we want the LLM to output
class SyntheticQA(BaseModel):
    question: str = Field(description="A realistic question a user might ask that is answered by the context.")
    answer: str = Field(description="A brief answer to the question based on the context.")

def generate_synthetic_dataset(sample_size=10, output_file="synthetic_eval_data.jsonl"):
    client = QdrantClient(url=os.environ["QDRANT_URL"], api_key=os.environ.get("QDRANT_API_KEY"))
    
    # Scroll/fetch a batch of existing chunks from your database
    records, _ = client.scroll(
        collection_name="insurance_policies",
        limit=sample_size,
        with_payload=True
    )
    
    llm = ChatGroq(
        api_key=os.environ["GROQ_API_KEY"],
        model_name="meta-llama/llama-4-scout-17b-16e-instruct",
        temperature=0.3
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert at generating evaluation datasets for an Insurance RAG system. "
                   "Given a chunk of policy text, generate exactly one realistic user question that is "
                   "fully answered by the chunk. Output valid JSON matching the required schema.\n\n"
                   "Format instructions: {format_instructions}"),
        ("human", "Context:\n{context}")
    ])
    
    parser = JsonOutputParser(pydantic_object=SyntheticQA)
    chain = prompt | llm | parser

    dataset = []
    print(f"Generating {len(records)} synthetic queries...")
    
    for record in records:
        context = record.payload.get("page_content", "")
        # Use the point ID (chunk ID) as our ground truth identifier
        chunk_id = record.id 
        metadata = record.payload.get("metadata", {})
        
        try:
            # Generate the QA pair
            result = chain.invoke({
                "context": context, 
                "format_instructions": parser.get_format_instructions()
            })
            
            dataset.append({
                "input": result["question"],
                "expected": result["answer"],
                "expected_chunk_id": chunk_id,
                "manual_id": metadata.get("manual_id"),
                "filename": metadata.get("filename")
            })
            print(f"Generated: {result['question']}")
            
        except Exception as e:
            print(f"Failed to generate for chunk {chunk_id}: {e}")

    # Save to a JSONL file
    with open(output_file, "w") as f:
        for item in dataset:
            f.write(json.dumps(item) + "\n")
            
    print(f"\nSaved {len(dataset)} examples to {output_file}")

if __name__ == "__main__":
    generate_synthetic_dataset(sample_size=15)
