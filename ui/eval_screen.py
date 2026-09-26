"""Evaluation Metrics screen: an automatic, always-on quality dashboard
over every extraction the live pipeline has actually run (groundedness,
found-in-document rate, confidence) — captured as soon as a document is
processed, with no button required. A separate, clearly optional section
still offers the labeled-testset judge benchmark (eval/run_eval.py) for
developers, since that needs ground truth that only exists for the
checked-in sample document.
"""
import json
from pathlib import Path

import pandas as pd
import streamlit as st

import db
from config import BASE_DIR

EVAL_DIR = BASE_DIR / "eval"


def _live_quality_dashboard():
    st.subheader("Live Extraction Quality")
    st.caption(
        "Computed automatically from every variable the pipeline has extracted so far — "
        "no test set, no button, always up to date."
    )

    rows = db.get_all_variables_with_doc()
    if not rows:
        st.info("No documents have been processed yet. Upload one on the Home screen.")
        return

    df = pd.DataFrame(rows)
    df["found_in_document"] = df["found_in_document"].astype(bool)
    df["grounded_bool"] = df["grounded"].apply(lambda g: bool(g) if g is not None else None)

    total = len(df)
    found_rate = df["found_in_document"].mean()
    grounded_rate = df["grounded_bool"].dropna().mean() if df["grounded_bool"].notna().any() else None
    avg_confidence = df["final_confidence"].dropna().mean() if df["final_confidence"].notna().any() else None

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Variables extracted", total)
    m2.metric("Found-in-document rate", f"{found_rate:.0%}")
    m3.metric("Groundedness rate", f"{grounded_rate:.0%}" if grounded_rate is not None else "—")
    m4.metric("Avg confidence", f"{avg_confidence:.1f}" if avg_confidence is not None else "—")

    st.markdown("**Per-document breakdown**")
    per_doc = (
        df.groupby("original_filename")
        .agg(
            variables=("variable_key", "count"),
            found_rate=("found_in_document", "mean"),
            grounded_rate=("grounded_bool", "mean"),
            avg_confidence=("final_confidence", "mean"),
        )
        .reset_index()
    )
    per_doc["found_rate"] = (per_doc["found_rate"] * 100).round(0).astype(int).astype(str) + "%"
    per_doc["grounded_rate"] = per_doc["grounded_rate"].apply(
        lambda x: f"{x * 100:.0f}%" if pd.notna(x) else "—"
    )
    per_doc["avg_confidence"] = per_doc["avg_confidence"].round(1)
    st.dataframe(per_doc, use_container_width=True, hide_index=True)

    doc_id = st.session_state.get("doc_id")
    if doc_id:
        current = df[df["doc_id"] == doc_id]
        if not current.empty:
            st.markdown(f"**Current document: '{current['original_filename'].iloc[0]}'**")
            display = current[
                ["label", "current_value", "final_confidence", "found_in_document", "grounded_bool", "guardrail_reason"]
            ].rename(
                columns={
                    "label": "Variable",
                    "current_value": "Value",
                    "final_confidence": "Confidence",
                    "found_in_document": "Found",
                    "grounded_bool": "Grounded",
                    "guardrail_reason": "Guardrail Reason",
                }
            )
            st.dataframe(display, use_container_width=True, hide_index=True)


def list_eval_reports() -> list[Path]:
    return sorted(EVAL_DIR.glob("results_*.json"), reverse=True)


def _labeled_benchmark_section():
    from eval.run_eval import run_evaluation

    st.caption(
        "Runs the labeled test set (eval/testset.jsonl) through the live pipeline and an "
        "independent LLM-judge subagent — a separate call from the one that produced the "
        "extraction, so the model never grades its own work. Useful for regression-testing "
        "prompt/retrieval changes against known-good answers."
    )

    reports = list_eval_reports()

    run_col, _ = st.columns([1, 3])
    with run_col:
        if st.button("Run new evaluation", type="primary"):
            progress = st.empty()
            results_box = st.container()
            rows_so_far = []

            def on_case(row):
                rows_so_far.append(row)
                progress.text(f"Evaluated {len(rows_so_far)} case(s)... last: {row['id']}")

            with st.spinner("Running test set through the live pipeline and the judge subagent..."):
                run_evaluation(on_case=on_case)
            progress.empty()
            results_box.success("Evaluation run complete.")
            st.rerun()

    if not reports:
        st.info(
            "No evaluation results yet. Click \"Run new evaluation\" above, or run "
            "`python eval/run_eval.py` from the command line."
        )
        return

    labels = [p.stem.replace("results_", "") for p in reports]
    selected_idx = st.selectbox(
        "Run", options=range(len(reports)), format_func=lambda i: f"{labels[i]} (latest)" if i == 0 else labels[i]
    )
    report = json.loads(reports[selected_idx].read_text())

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Cases", len(report["cases"]))
    m2.metric("Retrieval hit-rate", f"{report['hit_rate']:.0%}")
    m3.metric("Groundedness rate", f"{report['groundedness_rate']:.0%}")
    m4.metric("Avg value correctness", f"{report['avg_value_correctness']:.1f}")

    st.markdown("**Per-case results**")
    for row in report["cases"]:
        passed = row["hit"] and row["groundedness"]
        icon = "✅" if passed else "❌"
        with st.expander(f"{icon} [{row['id']}] {row['variable_label']}"):
            st.write(f"**Extracted value:** {row['value']}")
            c1, c2, c3, c4 = st.columns(4)
            c1.write(f"Confidence: {row['confidence']}")
            c2.write(f"Hit: {row['hit']}")
            c3.write(f"Grounded: {row['groundedness']}")
            c4.write(f"Value correctness: {row['value_correctness']}")
            st.caption(f"Judge reason: {row['reason']}")


def render_eval_screen():
    st.title("Evaluation Metrics")

    _live_quality_dashboard()

    st.divider()
    with st.expander("Labeled test-set benchmark (optional)"):
        _labeled_benchmark_section()
