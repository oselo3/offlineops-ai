# OfflineOps AI

**Production-grade AI engineering for air-gapped infrastructure — no internet required.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![Ollama](https://img.shields.io/badge/inference-Ollama-black.svg)](https://ollama.ai)
[![Docker](https://img.shields.io/badge/deployment-Docker-2496ED.svg)](https://www.docker.com/)

---

## What is this?

OfflineOps AI is an open-source AI-powered infrastructure operations assistant designed to run **entirely without internet access**. It answers infrastructure questions, diagnoses issues, and executes safe remediation actions — using only locally-hosted language models and your own documentation.

It is built for environments where cloud connectivity is unavailable, unreliable, or not permitted: large-scale on-premise deployments, government facilities, schools, hospitals, and enterprise data centres in emerging markets.

This project is also a deliberate, end-to-end implementation of the AI Engineering stack described in Chip Huyen's *AI Engineering* (O'Reilly, 2025) — covering evaluation, RAG, agents, inference optimization, and production architecture — applied to a real, hard problem.

> **Related project:** [InfraAgent](https://github.com/oselo3/infraagent) applies these same offline-first principles to a specific production problem — autonomous, risk-gated incident response tested end-to-end on live national-scale exam infrastructure. OfflineOps AI is the general toolkit; InfraAgent is a worked deployment.

---

## Why offline-first AI?

Most AI tooling assumes a stable, fast internet connection. Much of the world — and many high-security or large-scale on-premise deployments globally — does not have this. Existing solutions offer no practical path to:

- Running LLM inference on local hardware
- Retrieval over private internal documentation
- Tool-calling agents that act on real infrastructure
- Systematic evaluation without cloud eval APIs

OfflineOps AI fills this gap.

---

## Core capabilities

| Capability | Description |
|---|---|
| **Ask** | Natural language Q&A over your own runbooks, logs, and documentation |
| **Diagnose** | Root-cause analysis of infrastructure incidents using retrieved context |
| **Act** | Tool-calling agent that executes safe, scoped commands (disk check, service status, log tail) |
| **Evaluate** | Built-in evaluation harness to benchmark model accuracy on infra tasks |
| **Benchmark** | Side-by-side model comparison (Llama, Mistral, Phi, Gemma) on your eval set |

---

## Architecture overview

```
User Query
    │
    ▼
┌─────────────────────────────────────────┐
│              FastAPI Backend             │
│                                         │
│  ┌──────────┐    ┌──────────────────┐   │
│  │  Router  │───▶│  RAG Pipeline    │   │
│  └──────────┘    │  (Qdrant + embed)│   │
│       │          └──────────────────┘   │
│       ▼                   │             │
│  ┌──────────┐             ▼             │
│  │  Agent   │◀──── Retrieved Context   │
│  │  Engine  │                          │
│  └──────────┘                          │
│       │                                │
│       ▼                                │
│  ┌──────────────────────────────────┐  │
│  │         Ollama (local LLM)       │  │
│  │   Llama 3.1 / Mistral / Phi-3    │  │
│  └──────────────────────────────────┘  │
│       │                                │
│       ▼                                │
│  Tool Calls (optional)                 │
│  ├── check_disk_usage                  │
│  ├── get_service_status                │
│  ├── tail_log_file                     │
│  ├── ping_host                         │
│  └── list_network_interfaces           │
└─────────────────────────────────────────┘
    │
    ▼
Response + Sources + Actions Taken
```

---

## Tech stack

| Layer | Technology |
|---|---|
| LLM inference | [Ollama](https://ollama.ai) — local, quantized models (default `llama3.1:8b`) |
| Vector database | [Qdrant](https://qdrant.tech) — local instance via Docker |
| Embeddings | `nomic-embed-text` via Ollama |
| API | FastAPI + Uvicorn |
| Agent framework | Custom tool-calling loop (no LangChain dependency) |
| Evaluation | Custom harness — `datasets/eval-set-v1.jsonl` |
| Deployment | Docker Compose |
| UI | Lightweight web chat — *planned (see roadmap)* |

**No cloud. No API keys. No telemetry.**

---

## Getting started

### Prerequisites

- Docker and Docker Compose
- [Ollama](https://ollama.ai/download) installed and running
- Python 3.11 (for development)
- 16GB RAM recommended (benchmarks were run CPU-only on 16GB)

### 1. Clone and configure

```bash
git clone https://github.com/oselo3/offlineops-ai.git
cd offlineops-ai
cp .env.example .env
# Edit .env to set your model preferences
```

### 2. Pull models

```bash
ollama pull llama3.1:8b
ollama pull nomic-embed-text
```

### 3. Start services

```bash
docker compose up -d
```

### 4. Ingest your documentation

```bash
# Point --source at a directory of your own runbooks/docs (Markdown)
python scripts/ingest.py --source /path/to/your/runbooks --collection infra-docs
```

### 5. Query the assistant

The API is live at `http://localhost:8000`. Send a question to the `/chat` endpoint:

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "How do I find what is using disk space?"}'
```

Interactive API docs are available at `http://localhost:8000/docs`. A web chat UI is on the roadmap.

---

## Project structure

```
offlineops-ai/
├── core/
│   ├── rag/          # Document ingestion, chunking, retrieval
│   ├── agents/       # Tool-calling agent loop and tool definitions
│   ├── eval/         # Evaluation harness and metrics
│   └── inference/    # Ollama client, model benchmarking
├── api/              # FastAPI application and routes
├── datasets/         # Public eval set (eval-set-v1.jsonl)
├── docs/             # Benchmarks and results
├── scripts/          # CLI utilities (ingest, evaluate)
├── tests/            # Test suite (in progress)
├── docker-compose.yml
└── .env.example
```

---

## Evaluation & benchmarks

A core contribution of this project is a **public evaluation dataset** for infrastructure operations Q&A — `datasets/eval-set-v1.jsonl` (15 tasks across storage, network, services, performance, security, and logging; easy/medium/hard).

Headline results on `eval-set-v1`, **CPU-only (Intel i7 Ultra, 16GB RAM)**:

| Mode | Model | Pass rate | Avg ROUGE-L | Avg keyword coverage |
|---|---|---|---|---|
| RAG-only | `llama3.1:8b` | 100% | 0.957 | 92% |
| Agent | `llama3.1:8b` | 60% | 0.497 | 55% |

RAG-only achieves perfect task success. Agent mode currently drops to 60% due to inconsistent `Answer:` extraction in the tool-calling loop — a known, isolated issue in the agent engine; the retrieval layer itself is functioning correctly.

Run evaluations yourself:

```bash
python scripts/evaluate.py --model llama3.1:8b --eval-set datasets/eval-set-v1.jsonl
```

See [docs/BENCHMARKS.md](docs/BENCHMARKS.md) for the full breakdown.

---

## Roadmap

- [x] Repo structure and documentation
- [x] RAG pipeline (ingestion + retrieval)
- [x] Ollama inference wrapper with model switching
- [x] Tool-calling agent with safe Linux tools
- [x] Evaluation harness + eval dataset v1
- [x] FastAPI backend
- [x] Model benchmark results — `llama3.1:8b` (multi-model comparison in progress)
- [ ] Fix agent-loop `Answer:` extraction to close the RAG/agent gap
- [ ] Web UI (chat interface)
- [ ] Expand test suite
- [ ] LoRA finetuning experiment on infra Q&A dataset
- [ ] Example runbook library (public domain)

---

## Contributing

Contributions welcome. Please read [CONTRIBUTING.md](CONTRIBUTING.md) before submitting a pull request. Areas where help is most valuable:

- Additional runbook examples for the public dataset
- Tool definitions for other infrastructure tasks
- Evaluation question contributions
- Testing on different hardware configurations

---

## License

MIT — see [LICENSE](LICENSE).

---

## Acknowledgements

Built on the AI Engineering framework described in Chip Huyen's *AI Engineering: Building Applications with Foundation Models* (O'Reilly, 2025). Motivated by the real operational challenges of large-scale, offline infrastructure deployments.
