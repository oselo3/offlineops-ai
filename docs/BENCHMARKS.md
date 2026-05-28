## Results

| Mode | Model | Pass Rate | Avg ROUGE-L | Avg KW Coverage | Avg Latency | Hardware |
|---|---|---|---|---|---|---|
| RAG-only | llama3.1:8b | 100.0% | 0.9572 | 92.00% | ~118s | i7 Ultra, 16GB RAM, CPU-only |
| Agent | llama3.1:8b | 60.0% | 0.4971 | 54.89% | ~168s | i7 Ultra, 16GB RAM, CPU-only |

## Key finding
RAG-only mode achieves perfect scores on eval-set-v1. Agent mode drops to 60% due to
inconsistent `Answer:` extraction in the tool-calling loop — a known issue being addressed
in the agent engine. The RAG retrieval layer is functioning correctly.
