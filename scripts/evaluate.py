"""
scripts/evaluate.py

Run the evaluation harness from the command line.

Usage:
  python scripts/evaluate.py --model llama3.1:8b
  python scripts/evaluate.py --model mistral:7b --eval-set datasets/eval-set-v1.jsonl --no-agent
"""

import typer
from pathlib import Path
from rich.console import Console
from rich.table import Table

from core.inference.ollama_client import OllamaClient, Message
from core.rag.pipeline import RAGPipeline
from core.agents.engine import AgentEngine
from core.agents.tools import build_default_registry
from core.eval.harness import EvalHarness
import os

app = typer.Typer()
console = Console()


@app.command()
def run(
    model: str = typer.Option("llama3.1:8b", help="Ollama model name"),
    eval_set: Path = typer.Option(Path("datasets/eval-set-v1.jsonl"), help="Path to JSONL eval set"),
    ollama_url: str = typer.Option("http://localhost:11434"),
    qdrant_url: str = typer.Option("http://localhost:6333"),
    use_agent: bool = typer.Option(True, help="Use agent (with tools) vs plain RAG"),
    output: Path = typer.Option(Path("docs/eval-results-latest.json"), help="Where to save the report"),
    verbose: bool = typer.Option(True),
):
    console.print(f"\n[bold]OfflineOps AI — Evaluation Run[/bold]")
    console.print(f"Model: [cyan]{model}[/cyan] | Eval set: [cyan]{eval_set}[/cyan]\n")

    ollama = OllamaClient(base_url=ollama_url)
    rag = RAGPipeline(qdrant_url=qdrant_url, ollama_url=ollama_url)
    registry = build_default_registry()
    agent = AgentEngine(ollama=ollama, registry=registry, model=model)

    harness = EvalHarness()
    items = harness.load_eval_set(eval_set)
    console.print(f"Loaded {len(items)} eval items.\n")

    def eval_fn(question: str) -> str:
        chunks = rag.retrieve(question, top_k=5)
        context = rag.format_context(chunks)
        if use_agent:
            result = agent.run(query=question, context=context)
            return result.answer
        else:
            response = ollama.chat(
                messages=[Message(role="user", content=f"Context:\n{context}\n\nQuestion: {question}")],
                model=model,
                system="You are an infrastructure assistant. Answer concisely using the provided context.",
            )
            return response.content

    report = harness.evaluate(
        eval_fn=eval_fn,
        eval_set=items,
        model=model,
        eval_set_name=str(eval_set),
        verbose=verbose,
    )

    # Print summary table
    table = Table(title="\nEvaluation Summary", show_header=True)
    table.add_column("Metric", style="bold")
    table.add_column("Value", style="cyan")
    table.add_row("Model", report.model)
    table.add_row("Total questions", str(report.total))
    table.add_row("Passed", str(report.passed))
    table.add_row("Pass rate", f"{report.pass_rate:.1%}")
    table.add_row("Avg ROUGE-L", f"{report.avg_rouge_l:.4f}")
    table.add_row("Avg keyword coverage", f"{report.avg_keyword_coverage:.2%}")
    table.add_row("Avg latency", f"{report.avg_latency_ms:.0f}ms")
    console.print(table)

    harness.save_report(report, output)
    ollama.close()


if __name__ == "__main__":
    app()
