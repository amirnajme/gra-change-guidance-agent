"""Streamlit UI with two screens, selected from the sidebar:

- Assistant: chat input/history on top; below it, a Response box and a
  Source box (yellow-highlighted passage in the original PDF) side by
  side, each in its own bordered container. Asking a question — or
  clicking "Show this answer below" on an older message — updates both
  boxes via st.session_state, without navigating away from the chat.
- Evaluation Metrics: an observability screen showing the latest (or a
  past) run of the independent evaluation subagent against the labeled
  test set, plus a button to trigger a fresh run on demand.
"""
import json
import sys
from pathlib import Path
from typing import Optional

import pymupdf
import streamlit as st

from config import PDF_PATH
from graph import run_query

EVAL_DIR = Path(__file__).resolve().parent / "eval"
sys.path.insert(0, str(EVAL_DIR))
from run_eval import run_evaluation  # noqa: E402

st.set_page_config(page_title="GRA Change Guidance Agent", layout="wide")


@st.cache_resource
def get_pdf_document():
    return pymupdf.open(PDF_PATH)


def render_highlighted_page(page_num: int, bboxes: list) -> bytes:
    doc = get_pdf_document()
    page = doc.load_page(page_num)
    pix_page = page  # annotate a fresh copy each render, then revert
    added = []
    for bbox in bboxes:
        rect = pymupdf.Rect(*bbox)
        annot = pix_page.add_highlight_annot(rect)
        annot.set_colors(stroke=(1, 1, 0))
        annot.update()
        added.append(annot)
    pix = pix_page.get_pixmap(matrix=pymupdf.Matrix(1.6, 1.6))
    png_bytes = pix.tobytes("png")
    for annot in added:
        pix_page.delete_annot(annot)
    return png_bytes


def confidence_badge(confidence: int) -> str:
    if confidence >= 70:
        return f":green[**{confidence}% confidence**]"
    if confidence >= 40:
        return f":orange[**{confidence}% confidence**]"
    return f":red[**{confidence}% confidence**]"


def render_source_box(source: Optional[dict]):
    if not source:
        st.info("No source to display yet.")
        return
    st.caption(f"Page {source['page'] + 1} — {source['section']}")
    collection_meta = None
    try:
        from config import get_chroma_collection

        item = get_chroma_collection().get(ids=[source["id"]], include=["metadatas"])
        collection_meta = item["metadatas"][0] if item["metadatas"] else None
    except Exception:
        collection_meta = None

    bboxes = json.loads(collection_meta["bboxes_json"]) if collection_meta else []
    png_bytes = render_highlighted_page(source["page"], bboxes)
    st.image(png_bytes, use_container_width=True)
    with st.expander("Excerpt text used for the answer"):
        st.write(source["text"])


def render_assistant_screen():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "selected_msg_idx" not in st.session_state:
        st.session_state.selected_msg_idx = None
    if "selected_source" not in st.session_state:
        st.session_state.selected_source = None

    st.title("GRA Change Guidance Agent")
    st.caption("Answers are grounded strictly in the loaded guidance PDF.")

    st.subheader("Chat")
    chat_history = st.container(height=280)
    with chat_history:
        for idx, msg in enumerate(st.session_state.messages):
            with st.chat_message(msg["role"]):
                st.write(msg["content"])
                if msg["role"] == "assistant":
                    st.caption(confidence_badge(msg["confidence"]))
                    if st.button("Show this answer below", key=f"select_{idx}"):
                        st.session_state.selected_msg_idx = idx
                        citations = msg.get("citation_chunks", [])
                        st.session_state.selected_source = citations[0] if citations else None
                        st.rerun()

    question = st.chat_input("Ask a question about the guidance document...")
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.spinner("Retrieving and generating answer..."):
            result = run_query(question)

        citation_chunks = [c for c in result.get("retrieved", []) if c["id"] in result.get("citations", [])]
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": result.get("answer"),
                "confidence": result.get("final_confidence", 0),
                "citation_chunks": citation_chunks,
            }
        )
        st.session_state.selected_msg_idx = len(st.session_state.messages) - 1
        st.session_state.selected_source = citation_chunks[0] if citation_chunks else None
        st.rerun()

    st.divider()

    selected_msg = None
    if st.session_state.selected_msg_idx is not None:
        selected_msg = st.session_state.messages[st.session_state.selected_msg_idx]

    answer_col, source_col = st.columns(2)

    with answer_col:
        with st.container(border=True, height=460):
            st.subheader("Response")
            if not selected_msg:
                st.info("Ask a question above to see the answer here.")
            else:
                st.write(selected_msg["content"])
                st.caption(confidence_badge(selected_msg["confidence"]))
                citations = selected_msg.get("citation_chunks", [])
                if len(citations) > 1:
                    options = {f"page {c['page'] + 1} — {c['section']}": c for c in citations}
                    choice = st.radio(
                        "Cited passages",
                        list(options.keys()),
                        horizontal=True,
                        key=f"citation_choice_{st.session_state.selected_msg_idx}",
                    )
                    st.session_state.selected_source = options[choice]

    with source_col:
        with st.container(border=True, height=460):
            st.subheader("Source")
            render_source_box(st.session_state.selected_source)


def list_eval_reports() -> list[Path]:
    return sorted(EVAL_DIR.glob("results_*.json"), reverse=True)


def render_eval_screen():
    st.title("Evaluation Metrics")
    st.caption(
        "Independent evaluation-subagent runs against the labeled test set "
        "(eval/testset.jsonl) — retrieval hit-rate, groundedness, and answer "
        "relevance, judged by a separate LLM call from the one that generated "
        "the answers."
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
    m4.metric("Avg relevance", f"{report['avg_relevance']:.1f}")

    st.subheader("Per-case results")
    for row in report["cases"]:
        passed = row["hit"] and row["groundedness"]
        icon = "✅" if passed else "❌"
        with st.expander(f"{icon} [{row['id']}] {row['question']}"):
            st.write(f"**Answer:** {row['answer']}")
            c1, c2, c3, c4 = st.columns(4)
            c1.write(f"Confidence: {row['confidence']}")
            c2.write(f"Hit: {row['hit']}")
            c3.write(f"Grounded: {row['groundedness']}")
            c4.write(f"Relevance: {row['relevance_score']}")
            st.caption(f"Judge reason: {row['reason']}")


st.sidebar.title("GRA Change Guidance Agent")
screen = st.sidebar.radio("Screen", ["Assistant", "Evaluation Metrics"])

if screen == "Assistant":
    render_assistant_screen()
else:
    render_eval_screen()
