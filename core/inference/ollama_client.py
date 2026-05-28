"""
core/inference/ollama_client.py

Wrapper around the Ollama HTTP API.
Handles chat completions, embeddings, and model listing.
No streaming implementation yet — added in a later iteration.
"""

import time
from typing import Optional
import httpx
from pydantic import BaseModel


class Message(BaseModel):
    role: str  # "system" | "user" | "assistant"
    content: str


class ChatResponse(BaseModel):
    content: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float


class OllamaClient:
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url.rstrip("/")
        self._client = httpx.Client(timeout=120.0)

    # ------------------------------------------------------------------ #
    # Chat                                                                 #
    # ------------------------------------------------------------------ #

    def chat(
        self,
        messages: list[Message],
        model: str = "llama3.1:8b",
        temperature: float = 0.0,
        system: Optional[str] = None,
    ) -> ChatResponse:
        """
        Send a list of messages to Ollama and return a structured response.

        temperature=0.0 by default — important for reproducible eval runs.
        """
        payload_messages = []

        if system:
            payload_messages.append({"role": "system", "content": system})

        payload_messages.extend(
            [{"role": m.role, "content": m.content} for m in messages]
        )

        payload = {
            "model": model,
            "messages": payload_messages,
            "stream": False,
            "options": {"temperature": temperature},
        }

        t0 = time.perf_counter()
        response = self._client.post(
            f"{self.base_url}/api/chat",
            json=payload,
        )
        latency_ms = (time.perf_counter() - t0) * 1000

        response.raise_for_status()
        data = response.json()

        return ChatResponse(
            content=data["message"]["content"],
            model=data["model"],
            prompt_tokens=data.get("prompt_eval_count", 0),
            completion_tokens=data.get("eval_count", 0),
            latency_ms=round(latency_ms, 2),
        )

    # ------------------------------------------------------------------ #
    # Embeddings                                                           #
    # ------------------------------------------------------------------ #

    def embed(self, text: str, model: str = "nomic-embed-text") -> list[float]:
        """Return a vector embedding for the given text."""
        response = self._client.post(
            f"{self.base_url}/api/embeddings",
            json={"model": model, "prompt": text},
        )
        response.raise_for_status()
        return response.json()["embedding"]

    def embed_batch(
        self, texts: list[str], model: str = "nomic-embed-text"
    ) -> list[list[float]]:
        """Embed a list of texts. Sequential — Ollama doesn't batch natively."""
        return [self.embed(text, model=model) for text in texts]

    # ------------------------------------------------------------------ #
    # Model management                                                     #
    # ------------------------------------------------------------------ #

    def list_models(self) -> list[str]:
        """Return names of models currently available in Ollama."""
        response = self._client.get(f"{self.base_url}/api/tags")
        response.raise_for_status()
        return [m["name"] for m in response.json().get("models", [])]

    def is_available(self) -> bool:
        """Health check — returns True if Ollama is reachable."""
        try:
            self._client.get(f"{self.base_url}/api/tags", timeout=3.0)
            return True
        except httpx.RequestError:
            return False

    def close(self):
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
