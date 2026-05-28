"""
api/main.py

FastAPI application for OfflineOps AI.

Endpoints:
  POST /chat        — send a query, get an answer with sources
  GET  /health      — system health check (Ollama + Qdrant)
  GET  /models      — list available Ollama models
  POST /ingest      — ingest a document into the RAG pipeline
  GET  /collection  — get stats on the vector store
"""

import os
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from core.inference.ollama_client import OllamaClient
from core.rag.pipeline import RAGPipeline
from core.agents.engine import AgentEngine
from core.agents.tools import build_default_registry


# ------------------------------------------------------------------ #
# Config from environment                                              #
# ------------------------------------------------------------------ #

OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_DEFAULT_MODEL", "llama3.1:8b")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "infra-docs")


# ------------------------------------------------------------------ #
# App lifecycle                                                        #
# ------------------------------------------------------------------ #

ollama: OllamaClient
rag: RAGPipeline
agent: AgentEngine


@asynccontextmanager
async def lifespan(app: FastAPI):
    global ollama, rag, agent
    ollama = OllamaClient(base_url=OLLAMA_URL)
    rag = RAGPipeline(
        qdrant_url=QDRANT_URL,
        collection=QDRANT_COLLECTION,
        ollama_url=OLLAMA_URL,
    )
    registry = build_default_registry()
    agent = AgentEngine(ollama=ollama, registry=registry, model=OLLAMA_MODEL)
    yield
    ollama.close()


app = FastAPI(
    title="OfflineOps AI",
    description="AI-powered infrastructure operations assistant — fully offline",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ------------------------------------------------------------------ #
# Request / Response models                                            #
# ------------------------------------------------------------------ #

class ChatRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=2000)
    model: Optional[str] = None
    use_agent: bool = True       # False = RAG-only, no tool calls
    top_k: int = Field(5, ge=1, le=20)


class SourceDoc(BaseModel):
    source: str
    score: float
    preview: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceDoc]
    model: str
    latency_ms: float
    steps_count: int = 0


class IngestRequest(BaseModel):
    text: str
    source: str
    metadata: dict = {}


# ------------------------------------------------------------------ #
# Routes                                                               #
# ------------------------------------------------------------------ #

@app.get("/health")
async def health():
    ollama_ok = ollama.is_available()
    try:
        info = rag.collection_info()
        qdrant_ok = True
    except Exception:
        qdrant_ok = False
        info = {}

    status = "ok" if (ollama_ok and qdrant_ok) else "degraded"
    return {
        "status": status,
        "ollama": ollama_ok,
        "qdrant": qdrant_ok,
        "collection": info,
    }


@app.get("/models")
async def list_models():
    try:
        models = ollama.list_models()
        return {"models": models}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Ollama unavailable: {e}")


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    model = req.model or OLLAMA_MODEL

    # 1. Retrieve context
    chunks = rag.retrieve(req.query, top_k=req.top_k)
    context = rag.format_context(chunks)

    # 2. Run agent or plain RAG
    if req.use_agent:
        result = agent.run(query=req.query, context=context)
        answer = result.answer
        latency = result.total_latency_ms
        steps_count = len(result.steps)
    else:
        from core.inference.ollama_client import Message
        system = (
            "You are OfflineOps AI, an infrastructure operations assistant. "
            "Answer based only on the provided documentation context. "
            "If the context does not contain enough information, say so clearly."
        )
        user_msg = f"Context:\n{context}\n\nQuestion: {req.query}"
        response = ollama.chat(
            messages=[Message(role="user", content=user_msg)],
            model=model,
            system=system,
        )
        answer = response.content
        latency = response.latency_ms
        steps_count = 0

    sources = [
        SourceDoc(
            source=c.source,
            score=c.score,
            preview=c.text[:200] + "..." if len(c.text) > 200 else c.text,
        )
        for c in chunks
    ]

    return ChatResponse(
        answer=answer,
        sources=sources,
        model=model,
        latency_ms=latency,
        steps_count=steps_count,
    )


@app.post("/ingest", status_code=201)
async def ingest(req: IngestRequest):
    try:
        rag.ingest_text(req.text, source=req.source, metadata=req.metadata)
        return {"status": "ingested", "source": req.source}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/collection")
async def collection_info():
    try:
        return rag.collection_info()
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))
