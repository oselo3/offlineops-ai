"""
scripts/ingest.py

Ingest documentation into the OfflineOps AI vector store.

Usage:
  python scripts/ingest.py --source docs/runbooks/
  python scripts/ingest.py --source docs/runbooks/ --glob "**/*.md"
  python scripts/ingest.py --file docs/runbooks/disk-management.md
"""

import typer
from pathlib import Path
from rich.console import Console

from core.rag.pipeline import RAGPipeline

app = typer.Typer()
console = Console()


@app.command()
def run(
    source: Path = typer.Option(None, help="Directory to ingest recursively"),
    file: Path = typer.Option(None, help="Single file to ingest"),
    glob: str = typer.Option("**/*.md", help="Glob pattern for directory ingestion"),
    ollama_url: str = typer.Option("http://localhost:11434"),
    qdrant_url: str = typer.Option("http://localhost:6333"),
    collection: str = typer.Option("infra-docs"),
):
    if not source and not file:
        console.print("[red]Error:[/red] Provide --source (directory) or --file.")
        raise typer.Exit(1)

    console.print(f"\n[bold]OfflineOps AI — Document Ingestion[/bold]")

    rag = RAGPipeline(
        qdrant_url=qdrant_url,
        collection=collection,
        ollama_url=ollama_url,
    )

    if file:
        console.print(f"Ingesting file: [cyan]{file}[/cyan]")
        rag.ingest_file(file)
    else:
        console.print(f"Ingesting directory: [cyan]{source}[/cyan] (glob: {glob})")
        rag.ingest_directory(source, glob=glob)

    info = rag.collection_info()
    console.print(f"\n[green]Done.[/green] Collection '{info['collection']}' now has {info['points_count']} vectors.")


if __name__ == "__main__":
    app()
