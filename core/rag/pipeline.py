"""
core/rag/pipeline.py

Document ingestion and retrieval pipeline.

Flow:
  ingest: raw text/markdown files → chunks → embeddings → Qdrant
  retrieve: query → embedding → Qdrant search → ranked chunks
"""

import uuid
from pathlib import Path
from dataclasses import dataclass, field

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter

from core.inference.ollama_client import OllamaClient


# ------------------------------------------------------------------ #
# Data models                                                          #
# ------------------------------------------------------------------ #

@dataclass
class Document:
    text: str
    source: str          # file path or identifier
    metadata: dict = field(default_factory=dict)


@dataclass
class RetrievedChunk:
    text: str
    source: str
    score: float
    metadata: dict = field(default_factory=dict)


# ------------------------------------------------------------------ #
# Pipeline                                                             #
# ------------------------------------------------------------------ #

class RAGPipeline:
    """
    Handles ingestion and retrieval for OfflineOps AI.

    Uses:
      - Qdrant as the local vector store
      - nomic-embed-text via Ollama for embeddings
      - RecursiveCharacterTextSplitter for chunking
    """

    VECTOR_DIM = 768  # nomic-embed-text output dimension

    def __init__(
        self,
        qdrant_url: str = "http://localhost:6333",
        collection: str = "infra-docs",
        ollama_url: str = "http://localhost:11434",
        embed_model: str = "nomic-embed-text",
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        top_k: int = 5,
    ):
        self.collection = collection
        self.embed_model = embed_model
        self.top_k = top_k

        self.qdrant = QdrantClient(url=qdrant_url)
        self.ollama = OllamaClient(base_url=ollama_url)

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        self._ensure_collection()

    # ---------------------------------------------------------------- #
    # Collection management                                             #
    # ---------------------------------------------------------------- #

    def _ensure_collection(self):
        """Create the Qdrant collection if it doesn't exist."""
        existing = [c.name for c in self.qdrant.get_collections().collections]
        if self.collection not in existing:
            self.qdrant.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(
                    size=self.VECTOR_DIM,
                    distance=Distance.COSINE,
                ),
            )

    # ---------------------------------------------------------------- #
    # Ingestion                                                          #
    # ---------------------------------------------------------------- #

    def ingest_text(self, text: str, source: str, metadata: dict | None = None):
        """Chunk, embed, and store a single document."""
        chunks = self.splitter.split_text(text)
        self._upsert_chunks(chunks, source, metadata or {})

    def ingest_file(self, path: str | Path):
        """Read a markdown or text file and ingest it."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        text = path.read_text(encoding="utf-8")
        self.ingest_text(text, source=str(path), metadata={"filename": path.name})

    def ingest_directory(self, directory: str | Path, glob: str = "**/*.md"):
        """Recursively ingest all matching files from a directory."""
        directory = Path(directory)
        files = list(directory.glob(glob))
        print(f"Ingesting {len(files)} files from {directory}")
        for f in files:
            print(f"  → {f.name}")
            self.ingest_file(f)

    def _upsert_chunks(self, chunks: list[str], source: str, metadata: dict):
        """Embed chunks and upsert into Qdrant."""
        vectors = self.ollama.embed_batch(chunks, model=self.embed_model)
        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={
                    "text": chunk,
                    "source": source,
                    **metadata,
                },
            )
            for chunk, vector in zip(chunks, vectors)
        ]
        self.qdrant.upsert(collection_name=self.collection, points=points)

    # ---------------------------------------------------------------- #
    # Retrieval                                                          #
    # ---------------------------------------------------------------- #

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        """
        Embed a query and return the top-k most relevant chunks.
        """
        k = top_k or self.top_k
        query_vector = self.ollama.embed(query, model=self.embed_model)

        results = self.qdrant.search(
            collection_name=self.collection,
            query_vector=query_vector,
            limit=k,
            with_payload=True,
        )

        return [
            RetrievedChunk(
                text=r.payload["text"],
                source=r.payload.get("source", "unknown"),
                score=round(r.score, 4),
                metadata={k: v for k, v in r.payload.items() if k not in ("text", "source")},
            )
            for r in results
        ]

    def format_context(self, chunks: list[RetrievedChunk]) -> str:
        """Format retrieved chunks into a prompt-ready context block."""
        sections = []
        for i, chunk in enumerate(chunks, 1):
            sections.append(
                f"[Source {i}: {chunk.source} | score={chunk.score}]\n{chunk.text}"
            )
        return "\n\n---\n\n".join(sections)

    # ---------------------------------------------------------------- #
    # Stats                                                              #
    # ---------------------------------------------------------------- #

    def collection_info(self) -> dict:
        info = self.qdrant.get_collection(self.collection)
        return {
            "collection": self.collection,
            "vectors_count": info.vectors_count,
            "points_count": info.points_count,
        }
