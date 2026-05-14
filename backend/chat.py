import os
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

# Install rich if not present
try:
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.panel import Panel
except ImportError:
    print("Please install 'rich' to run this CLI: pip install rich")
    sys.exit(1)

import query_engine

def main():
    console = Console()
    console.print(Panel("[bold cyan]Welcome to the Insurance RAG CLI[/bold cyan]\nType 'exit' or 'quit' to stop.", title="Insurance AI"))
    
    while True:
        try:
            query = console.input("\n[bold green]You:[/bold green] ")
            if query.lower() in ["exit", "quit"]:
                break
            if not query.strip():
                continue
                
            with console.status("[bold yellow]Analyzing policies...[/bold yellow]", spinner="dots"):
                result = query_engine.answer_query(query)
                
            answer_md = Markdown(result["answer"])
            
            console.print("\n")
            console.print(Panel(answer_md, title="[bold blue]Assistant[/bold blue]", border_style="blue"))
            
            if result["sources"]:
                sources_text = ""
                for i, src in enumerate(result["sources"], 1):
                    sources_text += f"- **{src.get('filename', 'Unknown')}** (Page {src.get('page_number', '?')})\n"
                
                console.print(Panel(Markdown(sources_text), title="[bold magenta]Sources[/bold magenta]", border_style="magenta"))
                
        except KeyboardInterrupt:
            break
        except Exception as e:
            console.print(f"[bold red]Error:[/bold red] {e}")

if __name__ == "__main__":
    main()
