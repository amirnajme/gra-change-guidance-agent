"""CLI: run the labeled test set through the live pipeline, judge each
result with the independent eval subagent, and print/save an aggregate
report. Run manually: `python eval/run_eval.py`.
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from graph import run_query  # noqa: E402

from eval_agent import judge_answer

TESTSET_PATH = Path(__file__).resolve().parent / "testset.jsonl"


def load_testset():
    with open(TESTSET_PATH) as f:
        return [json.loads(line) for line in f if line.strip()]


def run_evaluation(on_case=None) -> dict:
    """Run the full testset through the live pipeline and the independent
    judge, save a timestamped report, and return it. `on_case`, if given, is
    called with each row dict as soon as it's ready (used by the UI to show
    progress)."""
    cases = load_testset()
    rows = []

    for case in cases:
        generated = run_query(case["question"])
        verdict = judge_answer(case["question"], case, generated)
        row = {
            "id": case["id"],
            "question": case["question"],
            "answerable": case["answerable"],
            "answer": generated.get("answer"),
            "confidence": generated.get("final_confidence"),
            "hit": verdict.hit,
            "groundedness": verdict.groundedness,
            "relevance_score": verdict.relevance_score,
            "reason": verdict.reason,
        }
        rows.append(row)
        if on_case:
            on_case(row)

    n = len(rows)
    hit_rate = sum(r["hit"] for r in rows) / n
    groundedness_rate = sum(r["groundedness"] for r in rows) / n
    avg_relevance = sum(r["relevance_score"] for r in rows) / n

    report = {
        "cases": rows,
        "hit_rate": hit_rate,
        "groundedness_rate": groundedness_rate,
        "avg_relevance": avg_relevance,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    out_path = Path(__file__).resolve().parent / f"results_{datetime.now(timezone.utc):%Y%m%dT%H%M%S}.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    report["path"] = str(out_path)
    return report


def main():
    report = run_evaluation(
        on_case=lambda r: print(
            f"[{r['id']}] hit={r['hit']} grounded={r['groundedness']} "
            f"relevance={r['relevance_score']} confidence={r['confidence']}"
        )
    )
    print("\n--- Aggregate report ---")
    print(f"Cases: {len(report['cases'])}")
    print(f"Retrieval hit-rate: {report['hit_rate']:.0%}")
    print(f"Groundedness rate: {report['groundedness_rate']:.0%}")
    print(f"Average relevance score: {report['avg_relevance']:.1f}")
    print(f"\nSaved detailed report to {report['path']}")


if __name__ == "__main__":
    main()
