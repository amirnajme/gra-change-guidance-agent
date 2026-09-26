"""Home screen: upload a guidance PDF and submit it for processing. Shows
running commentary (ingestion stages, then one line per variable) while the
pipeline works, using the same st.empty()-placeholder pattern already used
by the Evaluation Metrics screen's "Run new evaluation" progress display.
"""
import streamlit as st

import db
from graph import run_all_extractions
from ingest import ingest_document
from variable_schema import CRITICAL_VARIABLES


def _go_to(screen: str) -> None:
    st.session_state.screen = screen


def render_home_screen():
    st.title("Upload Guidance Document")
    st.caption(
        f"Upload a pharma regulatory guidance PDF. The system will extract "
        f"{len(CRITICAL_VARIABLES)} critical regulatory variables for review and validation."
    )

    uploaded_file = st.file_uploader("Guidance PDF", type=["pdf"])
    submit = st.button("Submit for Processing", type="primary", disabled=uploaded_file is None)

    if submit and uploaded_file is not None:
        file_bytes = uploaded_file.getvalue()
        progress = st.empty()

        with st.status("Processing document...", expanded=True) as status:
            def on_ingest_progress(msg: str) -> None:
                progress.info(msg)
                status.write(msg)

            doc_id = ingest_document(file_bytes, uploaded_file.name, on_progress=on_ingest_progress)

            def on_variable(index: int, total: int, variable, result) -> None:
                msg = f"Extracting variable {index}/{total}: {variable.label}..."
                progress.info(msg)
                status.write(msg)

            run_all_extractions(doc_id, on_variable=on_variable)
            db.set_document_status(doc_id, "extracted")
            status.update(label="Processing complete.", state="complete", expanded=False)

        progress.empty()
        st.session_state.doc_id = doc_id
        st.session_state.pdf_path = db.get_document(doc_id)["stored_path"]
        st.success(f"Extracted {len(CRITICAL_VARIABLES)} variables from '{uploaded_file.name}'.")
        if st.button("Review Key Variables"):
            _go_to("Key Variables")
            st.rerun()

    previous = db.list_documents()
    if previous:
        st.divider()
        st.subheader("Previously processed documents")
        options = {f"{d['original_filename']} ({d['doc_id']})": d for d in previous}
        choice = st.selectbox("Reopen a document", ["-- select --"] + list(options.keys()))
        if choice != "-- select --" and st.button("Open"):
            doc = options[choice]
            st.session_state.doc_id = doc["doc_id"]
            st.session_state.pdf_path = doc["stored_path"]
            _go_to("Key Variables")
            st.rerun()
