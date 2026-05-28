"""
scripts/evaluate.py

Run the evaluation harness from the command line.

Usage:
  python scripts/evaluate.py --model llama3.1:8b
  python scripts/evaluate.py --model mistral:7b --no-agent
"""

import argparse
from pathlib import Path

from core.inference.ollama_client import OllamaClient, Message
from core.rag.pipeline import RAGPipeline
from core.agents.engine import AgentEngine
from core.agents.tools import build_default_registry
from core.eval.harness import EvalHarness


def main():
    parser = argparse.ArgumentParser(description="Run OfflineOps AI evaluation")
    parser.add_argument("--model", type=str, default="llama3.1:8b")
    parser.add_argument("--eval-set", type=str, default="datasets/eval-set-v1.jsonl")
    parser.add_argument("--ollama-url", type=str, default="http://localhost:11434")
    parser.add_argument("--qdrant-url", type=str, default="http://localhost:6333")
    parser.add_argument("--no-agent", action="store_true", help="Use plain RAG without tool-calling")
    parser.add_argument("--output", type=str, default="docs/eval-results-latest.json")
    args = parser.parse_args()

    use_agent = not args.no_agent

    print(f"\nOfflineOps AI — Evaluation Run")
    print(f"Model: {args.model} | Eval set: {args.eval_set} | Agent: {use_agent}\n")

    ollama = OllamaClient(base_url=args.ollama_url)
    rag = RAGPipeline(qdrant_url=args.qdrant_url, ollama_url=args.ollama_url)
    registry = build_default_registry()
    agent = AgentEngine(ollama=ollama, registry=registry, model=args.model)

    harness = EvalHarness()
    items = harness.load_eval_set(args.eval_set)
    print(f"Loaded {len(items)} eval items.\n")

    def eval_fn(question: str) -> str:
        chunks = rag.retrieve(question, top_k=5)
        context = rag.format_context(chunks)
        if use_agent:
            result = agent.run(query=question, context=context)
            return result.answer
        else:
            response = ollama.chat(
                messages=[Message(role="user", content=f"Context:\n{context}\n\nQuestion: {question}")],
                model=args.model,
                system="You are an infrastructure assistant. Answer concisely using the provided context.",
            )
            return response.content

    report = harness.evaluate(
        eval_fn=eval_fn,
        eval_set=items,
        model=args.model,
        eval_set_name=args.eval_set,
        verbose=True,
    )

    print(f"\n--- Evaluation Summary ---")
    print(f"Model:               {report.model}")
    print(f"Total questions:     {report.total}")
    print(f"Passed:              {report.passed}")
    print(f"Pass rate:           {report.pass_rate:.1%}")
    print(f"Avg ROUGE-L:         {report.avg_rouge_l:.4f}")
    print(f"Avg keyword coverage:{report.avg_keyword_coverage:.2%}")
    print(f"Avg latency:         {report.avg_latency_ms:.0f}ms")

    harness.save_report(report, args.output)
    ollama.close()


if __name__ == "__main__":
    main()
