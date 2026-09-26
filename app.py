"""Streamlit entry point. Sidebar switches between four screens:

- Home: upload a guidance PDF and submit it for processing (running
  commentary shown while ingestion + variable extraction run).
- Variables: single-table view of all extracted variables plus the final
  Submit action, with an inline edit panel (side-by-side value/rationale
  and highlighted PDF source) for human-in-the-loop validation.
- Database: raw browser over every SQLite table, across all documents.
- Evaluation Metrics: automatic live extraction-quality dashboard, plus an
  optional labeled-testset judge benchmark.
"""
import streamlit as st

import db
from ui.database_screen import render_database_screen
from ui.eval_screen import render_eval_screen
from ui.home_screen import render_home_screen
from ui.variables_screen import render_variables_screen

st.set_page_config(page_title="GRA Change Guidance Agent", layout="wide")
db.init_db()

st.markdown(
    """
    <div style="display:flex;align-items:baseline;gap:0.6rem;margin-bottom:0.5rem;">
        <span style="font-size:1.6rem;font-weight:800;color:#D04A02;letter-spacing:-0.02em;">pwc</span>
        <span style="font-size:1.1rem;font-weight:600;color:#1A1A1A;">GRA Change Guidance Agent</span>
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.markdown("### GRA Change Guidance Agent")
st.session_state.setdefault("user_name", "Reviewer")
st.sidebar.text_input("Your name (for the audit trail)", key="user_name")

SCREENS = ["Home", "Variables", "Database", "Evaluation Metrics"]
st.session_state.setdefault("screen", "Home")
screen = st.sidebar.radio("Screen", SCREENS, key="screen")

if st.session_state.get("doc_id"):
    doc = db.get_document(st.session_state["doc_id"])
    if doc:
        st.sidebar.caption(f"Open document: {doc['original_filename']}")

if screen == "Home":
    render_home_screen()
elif screen == "Variables":
    render_variables_screen()
elif screen == "Database":
    render_database_screen()
else:
    render_eval_screen()
