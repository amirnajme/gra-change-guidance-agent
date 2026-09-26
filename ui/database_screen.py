"""Database screen: a raw browser over every table in the SQLite store
(documents, variables, variable_audit, submissions, logs), across all
uploaded documents — for full auditability without a separate sqlite3
session.
"""
import pandas as pd
import streamlit as st

import db

TABLES = ["documents", "variables", "variable_audit", "submissions", "logs"]


def render_database_screen():
    st.title("Database")
    st.caption("Raw view of the structured SQLite store (./data/gra_agent.db), across all documents.")

    table = st.selectbox("Table", TABLES)
    rows = db.get_all_rows(table)

    if not rows:
        st.info(f"'{table}' is empty.")
        return

    st.caption(f"{len(rows)} row(s)")
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
