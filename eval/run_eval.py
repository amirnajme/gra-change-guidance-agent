"""CLI: run the labeled test set through the live extraction pipeline,
judge each result with the independent eval subagent, and print/save an
aggregate report. Run manually: `python eval/run_eval.py`.
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # so `graph`/`llm`/etc. resolve
sys.path.insert(0, str(Path(__file__).resolve().parent))  # so `eval_agent` resolves even when
# this module is imported as `eval.run_eval` (e.g. from app.py) rather than run standalone.
from config import BASE_DIR  # noqa: E402
from graph import run_extraction  # noqa: E402
from ingest import ingest_document  # noqa: E402
from variable_schema import VARIABLES_BY_KEY  # noqa: E402

from eval_agent import judge_extraction  # noqa: E402

TESTSET_PATH = Path(__file__).resolve().parent / "testset.jsonl"


def load_testset():
    with open(TESTSET_PATH) as f:
        return [json.loads(line) for line in f if line.strip()]


def _resolve_doc_id(doc_filename: str) -> str:
    pdf_path = BASE_DIR / doc_filename
    with open(pdf_path, "rb") as f:
        file_bytes = f.read()
    return ingest_document(file_bytes, doc_filename)


def run_evaluation(on_case=None) -> dict:
    """Run the full testset through the live pipeline and the independent
    judge, save a timestamped report, and return it. `on_case`, if given, is
    called with each row dict as soon as it's ready (used by the UI to show
    progress)."""
    cases = load_testset()
    rows = []
    doc_ids = {}

    for case in cases:
        doc_id = doc_ids.setdefault(case["doc_filename"], _resolve_doc_id(case["doc_filename"]))
        variable = VARIABLES_BY_KEY[case["variable_key"]]
        generated = run_extraction(doc_id, variable)
        verdict = judge_extraction(variable, case, generated)
        row = {
            "id": case["id"],
            "variable_key": case["variable_key"],
            "variable_label": variable.label,
            "value": generated.get("value"),
            "confidence": generated.get("final_confidence"),
            "hit": verdict.hit,
            "groundedness": verdict.groundedness,
            "value_correctness": verdict.value_correctness,
            "reason": verdict.reason,
        }
        rows.append(row)
        if on_case:
            on_case(row)

    n = len(rows)
    hit_rate = sum(r["hit"] for r in rows) / n
    groundedness_rate = sum(r["groundedness"] for r in rows) / n
    avg_value_correctness = sum(r["value_correctness"] for r in rows) / n

    report = {
        "cases": rows,
        "hit_rate": hit_rate,
        "groundedness_rate": groundedness_rate,
        "avg_value_correctness": avg_value_correctness,
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
            f"value_correctness={r['value_correctness']} confidence={r['confidence']}"
        )
    )
    print("\n--- Aggregate report ---")
    print(f"Cases: {len(report['cases'])}")
    print(f"Retrieval hit-rate: {report['hit_rate']:.0%}")
    print(f"Groundedness rate: {report['groundedness_rate']:.0%}")
    print(f"Average value correctness: {report['avg_value_correctness']:.1f}")
    print(f"\nSaved detailed report to {report['path']}")


if __name__ == "__main__":
    main()
