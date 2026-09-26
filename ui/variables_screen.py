"""Variables screen: consolidates the old Table + Validation screens into
one page. A single table shows every extracted variable at a glance (plus
the final Submit action); clicking "Edit" on a row opens an editable
value/rationale panel with the highlighted PDF source side-by-side,
directly below the table, without navigating away.
"""
import json

import pandas as pd
import streamlit as st

import db
from ui.pdf_view import render_highlighted_page
from vectorstore import get_chroma_collection
from variable_schema import CRITICAL_VARIABLES


def _status_badge(row: dict) -> str:
    if row["validated"]:
        return "✅ Validated"
    if not row["found_in_document"]:
        return "⚪ Not found"
    if row["grounded"] == 0:
        return "🔴 Failed groundedness"
    return "🟡 Pending review"


def confidence_badge(confidence: int) -> str:
    if confidence is None:
        confidence = 0
    if confidence >= 70:
        return f":green[**{confidence}% confidence**]"
    if confidence >= 40:
        return f":orange[**{confidence}% confidence**]"
    return f":red[**{confidence}% confidence**]"


def _render_source(doc_id: str, pdf_path: str, row: dict) -> None:
    cited_ids = json.loads(row["cited_chunk_ids_json"] or "[]")
    if not cited_ids or row["page"] is None:
        st.info("No source passage to display for this variable.")
        return

    st.caption(f"Page {row['page'] + 1} — {row['section']}")
    try:
        item = get_chroma_collection(doc_id).get(ids=[cited_ids[0]], include=["documents", "metadatas"])
        meta = item["metadatas"][0] if item["metadatas"] else None
        chunk_text = item["documents"][0] if item["documents"] else ""
    except Exception:
        meta, chunk_text = None, ""

    bboxes = json.loads(meta["bboxes_json"]) if meta else []
    png_bytes = render_highlighted_page(row["page"], bboxes, pdf_path)
    st.image(png_bytes, use_container_width=True)
    with st.expander("Excerpt text used for extraction"):
        st.write(chunk_text)


def _render_edit_panel(doc_id: str, pdf_path: str, selected_key: str) -> None:
    row = db.get_variable(doc_id, selected_key)
    if not row:
        st.info("This variable hasn't been extracted yet.")
        return

    st.divider()
    keys_in_order = [v.key for v in CRITICAL_VARIABLES]
    idx = keys_in_order.index(selected_key)
    nav_prev, nav_title, nav_next = st.columns([1, 4, 1])
    with nav_prev:
        if st.button("Previous", disabled=idx == 0):
            st.session_state.selected_variable_key = keys_in_order[idx - 1]
            st.rerun()
    with nav_title:
        st.markdown(f"**Editing: {row['label']}**  ({idx + 1}/{len(keys_in_order)})")
    with nav_next:
        if st.button("Next", disabled=idx == len(keys_in_order) - 1):
            st.session_state.selected_variable_key = keys_in_order[idx + 1]
            st.rerun()

    value_col, source_col = st.columns(2)

    with value_col:
        with st.container(border=True, height=520):
            st.subheader("Extracted Value")
            st.caption(confidence_badge(row["final_confidence"]))
            if not row["found_in_document"]:
                st.warning("Not found in document.")
            elif row["grounded"] == 0:
                st.error(f"Failed groundedness check: {row['guardrail_reason']}")
            new_value = st.text_area("Current value", value=row["current_value"] or "", key=f"value_{selected_key}")
            st.caption("Rationale")
            st.write(row["rationale"] or "—")
            if row["validated"]:
                st.caption(f":green[Validated by {row['validated_by']} at {row['validated_at']}]")

            if st.button("Save & Mark Validated", type="primary", key=f"save_{selected_key}"):
                user = st.session_state.get("user_name", "Reviewer")
                db.save_variable_edit(doc_id, selected_key, new_value, user)
                st.success("Saved.")
                st.rerun()

    with source_col:
        with st.container(border=True, height=520):
            st.subheader("Source in Document")
            _render_source(doc_id, pdf_path, row)


def render_variables_screen():
    doc_id = st.session_state.get("doc_id")
    pdf_path = st.session_state.get("pdf_path")
    if not doc_id:
        st.info("Upload a document on the Home screen first.")
        return

    doc = db.get_document(doc_id)
    st.title("Extracted Variables")
    st.caption(f"Document: '{doc['original_filename']}'")

    variables = db.get_variables(doc_id)
    if not variables:
        st.info("No variables extracted yet.")
        return

    df = pd.DataFrame(
        [
            {
                "Variable": row["label"],
                "Value": row["current_value"] or "—",
                "Confidence": row["final_confidence"],
                "Found in Document": bool(row["found_in_document"]),
                "Grounded": bool(row["grounded"]) if row["grounded"] is not None else None,
                "Status": _status_badge(row),
                "Page": (row["page"] + 1) if row["page"] is not None else None,
            }
            for row in variables
        ]
    )
    st.dataframe(df, use_container_width=True, hide_index=True)

    validated_count = sum(1 for row in variables if row["validated"])
    st.caption(f"{validated_count}/{len(variables)} variables validated by a human reviewer.")

    st.subheader("Review & Edit")
    keys_in_order = [v.key for v in CRITICAL_VARIABLES]
    variables_by_key = {row["variable_key"]: row for row in variables}
    edit_cols = st.columns(3)
    for i, key in enumerate(keys_in_order):
        row = variables_by_key.get(key)
        if not row:
            continue
        with edit_cols[i % 3]:
            if st.button(f"Edit: {row['label']}", key=f"edit_btn_{key}", use_container_width=True):
                st.session_state.selected_variable_key = key
                st.rerun()

    selected_key = st.session_state.get("selected_variable_key")
    if selected_key:
        _render_edit_panel(doc_id, pdf_path, selected_key)

    st.divider()
    st.subheader("Submit Final Values")
    user = st.session_state.get("user_name", "Reviewer")
    st.caption(f"Submitting will record the current values above as final, attributed to '{user}'.")
    if st.button("Submit All Values", type="primary"):
        result = db.submit_document(doc_id, submitted_by=user)
        st.success(
            f"Submitted {result['variable_count']} variables for '{doc['original_filename']}' at {result['submitted_at']}."
        )
