"""
core/eval/harness.py

Evaluation harness for OfflineOps AI.

Loads an eval set (JSONL), runs each question through the full pipeline,
and scores responses using exact match, keyword coverage, and ROUGE-L.

Eval set format (one JSON object per line):
{
  "id": "disk-001",
  "question": "How do I check which partition is consuming the most disk space?",
  "expected_answer": "Use df -h to see disk usage by mount point, then du -sh /* to find large directories.",
  "keywords": ["df", "du", "disk", "partition"],
  "category": "storage",
  "difficulty": "easy"
}
"""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from rouge_score import rouge_scorer


# ------------------------------------------------------------------ #
# Data models                                                          #
# ------------------------------------------------------------------ #

@dataclass
class EvalItem:
    id: str
    question: str
    expected_answer: str
    keywords: list[str] = field(default_factory=list)
    category: str = "general"
    difficulty: str = "medium"


@dataclass
class EvalResult:
    id: str
    question: str
    expected_answer: str
    actual_answer: str
    rouge_l: float
    keyword_coverage: float
    latency_ms: float
    category: str
    difficulty: str
    passed: bool    # True if rouge_l >= threshold AND keyword_coverage >= threshold


@dataclass
class EvalReport:
    model: str
    eval_set: str
    total: int
    passed: int
    pass_rate: float
    avg_rouge_l: float
    avg_keyword_coverage: float
    avg_latency_ms: float
    results: list[EvalResult]
    by_category: dict[str, dict]
    by_difficulty: dict[str, dict]


# ------------------------------------------------------------------ #
# Scoring functions                                                    #
# ------------------------------------------------------------------ #

def score_rouge_l(reference: str, hypothesis: str) -> float:
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = scorer.score(reference, hypothesis)
    return round(scores["rougeL"].fmeasure, 4)


def score_keyword_coverage(keywords: list[str], answer: str) -> float:
    if not keywords:
        return 1.0
    answer_lower = answer.lower()
    hits = sum(1 for kw in keywords if kw.lower() in answer_lower)
    return round(hits / len(keywords), 4)


# ------------------------------------------------------------------ #
# Harness                                                              #
# ------------------------------------------------------------------ #

class EvalHarness:
    def __init__(
        self,
        rouge_threshold: float = 0.3,
        keyword_threshold: float = 0.5,
    ):
        self.rouge_threshold = rouge_threshold
        self.keyword_threshold = keyword_threshold

    def load_eval_set(self, path: str | Path) -> list[EvalItem]:
        path = Path(path)
        items = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                items.append(EvalItem(**data))
        return items

    def evaluate(
        self,
        eval_fn,               # callable(question: str) -> str
        eval_set: list[EvalItem],
        model: str = "unknown",
        eval_set_name: str = "unknown",
        verbose: bool = True,
    ) -> EvalReport:
        """
        Run evaluation.

        eval_fn: a callable that takes a question string and returns an answer string.
        This decouples the harness from the specific pipeline implementation.
        """
        results = []

        for item in eval_set:
            if verbose:
                print(f"  [{item.id}] {item.question[:60]}...")

            t0 = time.perf_counter()
            actual = eval_fn(item.question)
            latency = (time.perf_counter() - t0) * 1000

            rouge_l = score_rouge_l(item.expected_answer, actual)
            kw_cov = score_keyword_coverage(item.keywords, actual)
            passed = rouge_l >= self.rouge_threshold and kw_cov >= self.keyword_threshold

            results.append(EvalResult(
                id=item.id,
                question=item.question,
                expected_answer=item.expected_answer,
                actual_answer=actual,
                rouge_l=rouge_l,
                keyword_coverage=kw_cov,
                latency_ms=round(latency, 2),
                category=item.category,
                difficulty=item.difficulty,
                passed=passed,
            ))

            if verbose:
                status = "✓" if passed else "✗"
                print(f"    {status} ROUGE-L={rouge_l:.3f} | KW={kw_cov:.2f} | {latency:.0f}ms")

        return self._build_report(results, model, eval_set_name)

    def _build_report(
        self,
        results: list[EvalResult],
        model: str,
        eval_set_name: str,
    ) -> EvalReport:
        total = len(results)
        passed = sum(1 for r in results if r.passed)

        def avg(values):
            return round(sum(values) / len(values), 4) if values else 0.0

        # Aggregate by category
        categories: dict[str, list[EvalResult]] = {}
        for r in results:
            categories.setdefault(r.category, []).append(r)

        by_category = {
            cat: {
                "total": len(rs),
                "passed": sum(1 for r in rs if r.passed),
                "pass_rate": round(sum(1 for r in rs if r.passed) / len(rs), 4),
                "avg_rouge_l": avg([r.rouge_l for r in rs]),
            }
            for cat, rs in categories.items()
        }

        # Aggregate by difficulty
        difficulties: dict[str, list[EvalResult]] = {}
        for r in results:
            difficulties.setdefault(r.difficulty, []).append(r)

        by_difficulty = {
            diff: {
                "total": len(rs),
                "pass_rate": round(sum(1 for r in rs if r.passed) / len(rs), 4),
            }
            for diff, rs in difficulties.items()
        }

        return EvalReport(
            model=model,
            eval_set=eval_set_name,
            total=total,
            passed=passed,
            pass_rate=round(passed / total, 4) if total else 0.0,
            avg_rouge_l=avg([r.rouge_l for r in results]),
            avg_keyword_coverage=avg([r.keyword_coverage for r in results]),
            avg_latency_ms=avg([r.latency_ms for r in results]),
            results=results,
            by_category=by_category,
            by_difficulty=by_difficulty,
        )

    def save_report(self, report: EvalReport, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump({
                "model": report.model,
                "eval_set": report.eval_set,
                "summary": {
                    "total": report.total,
                    "passed": report.passed,
                    "pass_rate": report.pass_rate,
                    "avg_rouge_l": report.avg_rouge_l,
                    "avg_keyword_coverage": report.avg_keyword_coverage,
                    "avg_latency_ms": report.avg_latency_ms,
                },
                "by_category": report.by_category,
                "by_difficulty": report.by_difficulty,
                "results": [
                    {
                        "id": r.id,
                        "passed": r.passed,
                        "rouge_l": r.rouge_l,
                        "keyword_coverage": r.keyword_coverage,
                        "latency_ms": r.latency_ms,
                        "actual_answer": r.actual_answer,
                    }
                    for r in report.results
                ],
            }, f, indent=2)
        print(f"Report saved → {path}")
