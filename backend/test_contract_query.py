import sys
import os
from pathlib import Path
from dotenv import load_dotenv

# Load env from parent directory
load_dotenv(Path(__file__).parent.parent / ".env")

# Ensure backend is in path
sys.path.append(str(Path(__file__).parent))

from contract_query import answer_contract_query

def main():
    # Allow passing query as command line argument
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        query = "What are the recapture percentages?"

    print(f"\n[bold cyan]Querying Contract RAG:[/bold cyan] {query}\n")
    
    try:
        result = answer_contract_query(query)
        
        print("\n--- [bold green]ASSISTANT ANSWER[/bold green] ---\n")
        print(result["answer"])
        
        if result.get("sources"):
            print("\n--- [bold yellow]SOURCES[/bold yellow] ---\n")
            for i, src in enumerate(result["sources"], 1):
                chunk_id = src.get("chunk_id", "unknown")
                section = src.get("section", "?")
                print(f"{i}. [Section {section}] (ID: {chunk_id})")
                # Optional: print snippet
                # print(f"   Snippet: {src.get('snippet', '')}...")
                
    except Exception as e:
        print(f"\n[bold red]Error during query:[/bold red] {e}")

if __name__ == "__main__":
    # Check if rich is installed for pretty printing, otherwise fallback to plain print
    try:
        from rich import print
    except ImportError:
        pass
        
    main()
