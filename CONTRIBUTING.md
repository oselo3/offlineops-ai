# Contributing to OfflineOps AI

Thank you for your interest. Contributions of all kinds are welcome.

## Ways to contribute

### 1. Evaluation questions
Add questions to `datasets/eval-set-v1.jsonl`. Each question should:
- Cover a real infrastructure problem with a deterministic, verifiable answer
- Include 3–6 keywords that must appear in a correct response
- Specify a category: `storage | network | services | performance | security | logging`
- Specify a difficulty: `easy | medium | hard`

### 2. Runbook documents
Add markdown runbooks to `datasets/runbooks/`. These are ingested into the RAG pipeline and serve as the knowledge base. Topics needed:
- Backup and restore procedures
- Database administration (PostgreSQL, MySQL)
- Container troubleshooting (Docker)
- Certificate management (TLS/SSL)
- Cron job debugging

### 3. Tool definitions
Add new safe, read-only infrastructure tools in `core/agents/tools.py`. All tools must:
- Return a plain string
- Enforce a timeout
- Validate inputs before passing to subprocess
- Be documented with a clear description and parameter schema

### 4. Bug reports and feature requests
Open an issue. Provide:
- What you expected to happen
- What actually happened
- Steps to reproduce
- Your OS, hardware, and Ollama version

## Development setup

```bash
git clone https://github.com/YOUR_USERNAME/offlineops-ai.git
cd offlineops-ai
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Run tests:
```bash
pytest tests/
```

## Code style

- Python 3.10+ type hints throughout
- Docstrings on all public functions and classes
- No LangChain or LlamaIndex as core dependencies
- All subprocess calls must be explicit, bounded, and sanitized

## Pull request checklist

- [ ] Code runs without errors locally
- [ ] Tests added or updated where applicable
- [ ] README updated if new features added
- [ ] No proprietary or sensitive data included in datasets
