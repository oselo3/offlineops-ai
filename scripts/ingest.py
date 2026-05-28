"""
scripts/ingest.py

Ingest documentation into the OfflineOps AI vector store.

Usage:
  python scripts/ingest.py --file datasets/eval-set-v1.jsonl
  python scripts/ingest.py --source docs/runbooks/
"""

import argparse
from pathlib import Path
from core.rag.pipeline import RAGPipeline


def main():
    parser = argparse.ArgumentParser(description="Ingest documents into OfflineOps AI")
    parser.add_argument("--file", type=str, help="Single file to ingest")
    parser.add_argument("--source", type=str, help="Directory to ingest recursively")
    parser.add_argument("--glob", type=str, default="**/*.md")
    parser.add_argument("--ollama-url", type=str, default="http://localhost:11434")
    parser.add_argument("--qdrant-url", type=str, default="http://localhost:6333")
    parser.add_argument("--collection", type=str, default="infra-docs")
    args = parser.parse_args()

    if not args.file and not args.source:
        print("Error: provide --file or --source")
        exit(1)

    rag = RAGPipeline(
        qdrant_url=args.qdrant_url,
        collection=args.collection,
        ollama_url=args.ollama_url,
    )

    if args.file:
        print(f"Ingesting file: {args.file}")
        rag.ingest_file(Path(args.file))
    else:
        print(f"Ingesting directory: {args.source}")
        rag.ingest_directory(Path(args.source), glob=args.glob)

    info = rag.collection_info()
    print(f"Done. Collection '{info['collection']}' now has {info['points_count']} vectors.")


if __name__ == "__main__":
    main()
