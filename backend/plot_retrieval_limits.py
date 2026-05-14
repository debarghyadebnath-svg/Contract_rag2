import argparse
import os
import time
import matplotlib.pyplot as plt
from ranx import evaluate, Run

from langchain_nomic.embeddings import NomicEmbeddings
from langchain_cohere import CohereRerank
from langchain_core.documents import Document

from eval_lancedb import setup_lancedb, load_data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-file", default="aligned_eval_data.jsonl")
    args = parser.parse_args()

    print("Setting up LanceDB...")
    table = setup_lancedb()
    
    print("Loading queries...")
    queries_dict, qrels = load_data(args.data_file)
    
    # Initialize Models
    embeddings = NomicEmbeddings(
        model="nomic-embed-text-v1.5", 
        nomic_api_key=os.environ.get("NOMIC_API_KEY"),
       
    )
    reranker = CohereRerank(
        cohere_api_key=os.environ.get("COHERE_API_KEY"), 
        model="rerank-english-v3.0"
    )
    
    
    # The limits we want to test (number of chunks fetched before reranking)
    limits = [10, 50, 100]
    ks = [1, 3, 5, 10, 15, 20, 25]
    
    results = {
        limit: {"mrr": [], "recall": []} for limit in limits
    }
    
    print("Starting Two-Stage Retrieval Evaluation...")
    for limit in limits:
        print(f"\n--- Evaluating Initial Retrieval Limit = {limit} ---")
        run_dict = {}  
        
        for i, (q_id, query) in enumerate(queries_dict.items()):
            print(f"  Processing query {i+1}/{len(queries_dict)}")
            # 1. Base Dense Retrieval
            query_vector = embeddings.embed_query(query)
            base_results = table.search(query_vector).limit(limit).to_list()
            
            if not base_results:
                run_dict[q_id] = {}
                continue
                
            # 2. Reranking Stage
            docs = [Document(page_content=r["text"], metadata={"id": r["id"]}) for r in base_results]
            
            success = False
            retries = 0
            while not success and retries < 5:
                try:
                    reranked_docs = reranker.compress_documents(docs, query)
                    success = True
                except Exception as e:
                    if "429" in str(e) or "TooManyRequests" in str(e.__class__.__name__):
                        print(f"    [Cohere Rate Limit] Sleeping for 60s... (Retry {retries+1})")
                        time.sleep(60)
                        retries += 1
                    else:
                        raise e
            
            if not success:
                raise Exception("Failed to rerank due to rate limits.")
                
            # Assign scores based on new reranked order
            scores = {}
            for j, doc in enumerate(reranked_docs):
                scores[doc.metadata["id"]] = doc.metadata.get("relevance_score", 1.0 - (j*0.001))
                
            run_dict[q_id] = scores
            
        # Evaluate Run
        run = Run(run_dict, name=f"Limit-{limit}")
        for k in ks:
            results[limit]["mrr"].append(evaluate(qrels, run, f"mrr@{k}"))
            results[limit]["recall"].append(evaluate(qrels, run, f"recall@{k}"))
            
    print("\nGenerating Plots...")
    
    # Plot MRR
    plt.figure(figsize=(10, 6))
    for limit in limits:
        plt.plot(ks, results[limit]["mrr"], marker='o', label=f"Retrieval Limit = {limit}", linewidth=2)
    plt.title("MRR@k for Different Retrieval Limits (with Cohere Reranking)")
    plt.xlabel("k (Final Cutoff)")
    plt.ylabel("MRR")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.xticks(ks)
    plt.ylim(0, 1.05)
    plt.savefig("mrr_limits_plot.png", dpi=300, bbox_inches='tight')
    
    # Plot Recall
    plt.figure(figsize=(10, 6))
    for limit in limits:
        plt.plot(ks, results[limit]["recall"], marker='o', label=f"Retrieval Limit = {limit}", linewidth=2)
    plt.title("Recall@k for Different Retrieval Limits (with Cohere Reranking)")
    plt.xlabel("k (Final Cutoff)")
    plt.ylabel("Recall")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.xticks(ks)
    plt.ylim(0, 1.05)
    plt.savefig("recall_limits_plot.png", dpi=300, bbox_inches='tight')
    
    print("Success! Saved mrr_limits_plot.png and recall_limits_plot.png")

if __name__ == "__main__":
    main()
