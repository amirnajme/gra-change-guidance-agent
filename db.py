"""Structured SQLite storage: uploaded-document registry, per-variable
extraction results, the human-in-the-loop audit trail, final submissions,
and the observability event log — all in one local file (config.DB_PATH),
replacing the old JSONL log and giving the extracted/validated data a
queryable, auditable home instead of living only in st.session_state.
"""
import json
import sqlite3
from datetime import datetime, timezone
from functools import lru_cache

from config import DB_PATH

_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id TEXT PRIMARY KEY,
    original_filename TEXT NOT NULL,
    stored_path TEXT NOT NULL,
    page_count INTEGER,
    chunk_count INTEGER,
    status TEXT NOT NULL DEFAULT 'uploaded',
    error_message TEXT,
    uploaded_at TEXT NOT NULL,
    ingested_at TEXT
);

CREATE TABLE IF NOT EXISTS variables (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id TEXT NOT NULL REFERENCES documents(doc_id),
    variable_key TEXT NOT NULL,
    label TEXT NOT NULL,
    value TEXT,
    normalized_value TEXT,
    rationale TEXT,
    cited_chunk_ids_json TEXT,
    page INTEGER,
    section TEXT,
    self_confidence INTEGER,
    final_confidence INTEGER,
    found_in_document INTEGER NOT NULL DEFAULT 0,
    grounded INTEGER,
    guardrail_reason TEXT,
    run_id TEXT,
    extracted_at TEXT,
    current_value TEXT,
    validated INTEGER NOT NULL DEFAULT 0,
    validated_by TEXT,
    validated_at TEXT,
    UNIQUE(doc_id, variable_key)
);

CREATE TABLE IF NOT EXISTS variable_audit (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id TEXT NOT NULL,
    variable_key TEXT NOT NULL,
    changed_from TEXT,
    changed_to TEXT,
    change_type TEXT NOT NULL,
    user TEXT NOT NULL,
    changed_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS submissions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id TEXT NOT NULL REFERENCES documents(doc_id),
    submitted_by TEXT NOT NULL,
    submitted_at TEXT NOT NULL,
    variable_count INTEGER NOT NULL,
    snapshot_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    run_id TEXT,
    doc_id TEXT,
    event TEXT NOT NULL,
    fields_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_logs_run_id ON logs(run_id);
CREATE INDEX IF NOT EXISTS idx_variables_doc ON variables(doc_id);
CREATE INDEX IF NOT EXISTS idx_audit_doc ON variable_audit(doc_id);
"""


@lru_cache(maxsize=1)
def get_connection() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    conn = get_connection()
    conn.executescript(_SCHEMA)
    conn.commit()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# --- documents -------------------------------------------------------------

def upsert_document(doc_id: str, original_filename: str, stored_path: str, status: str = "uploaded") -> None:
    conn = get_connection()
    conn.execute(
        """INSERT INTO documents (doc_id, original_filename, stored_path, status, uploaded_at)
           VALUES (?, ?, ?, ?, ?)
           ON CONFLICT(doc_id) DO UPDATE SET status=excluded.status""",
        (doc_id, original_filename, stored_path, status, _now()),
    )
    conn.commit()


def get_document(doc_id: str) -> dict | None:
    conn = get_connection()
    row = conn.execute("SELECT * FROM documents WHERE doc_id = ?", (doc_id,)).fetchone()
    return dict(row) if row else None


def list_documents() -> list[dict]:
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM documents WHERE status IN ('ingested', 'extracted') ORDER BY uploaded_at DESC"
    ).fetchall()
    return [dict(r) for r in rows]


def set_document_status(doc_id: str, status: str, error_message: str | None = None) -> None:
    conn = get_connection()
    conn.execute(
        "UPDATE documents SET status = ?, error_message = ? WHERE doc_id = ?",
        (status, error_message, doc_id),
    )
    conn.commit()


def mark_ingested(doc_id: str, page_count: int, chunk_count: int) -> None:
    conn = get_connection()
    conn.execute(
        """UPDATE documents SET status = 'ingested', page_count = ?, chunk_count = ?, ingested_at = ?
           WHERE doc_id = ?""",
        (page_count, chunk_count, _now(), doc_id),
    )
    conn.commit()


# --- variables --------------------------------------------------------------

def save_variable_result(doc_id: str, variable_key: str, label: str, **fields) -> None:
    """Upsert one variable's freshly-extracted result. `current_value`
    resets to the new extraction (a re-run replaces prior human edits with
    the new machine draft, which the user then re-validates)."""
    conn = get_connection()
    payload = {
        "doc_id": doc_id,
        "variable_key": variable_key,
        "label": label,
        "value": fields.get("value"),
        "normalized_value": fields.get("normalized_value"),
        "rationale": fields.get("rationale"),
        "cited_chunk_ids_json": json.dumps(fields.get("cited_chunk_ids", [])),
        "page": fields.get("page"),
        "section": fields.get("section"),
        "self_confidence": fields.get("self_confidence"),
        "final_confidence": fields.get("final_confidence"),
        "found_in_document": int(bool(fields.get("found_in_document"))),
        "grounded": int(bool(fields.get("grounded"))) if fields.get("grounded") is not None else None,
        "guardrail_reason": fields.get("guardrail_reason"),
        "run_id": fields.get("run_id"),
        "extracted_at": _now(),
        "current_value": fields.get("value"),
    }
    conn.execute(
        """INSERT INTO variables (doc_id, variable_key, label, value, normalized_value, rationale,
               cited_chunk_ids_json, page, section, self_confidence, final_confidence,
               found_in_document, grounded, guardrail_reason, run_id, extracted_at, current_value)
           VALUES (:doc_id, :variable_key, :label, :value, :normalized_value, :rationale,
               :cited_chunk_ids_json, :page, :section, :self_confidence, :final_confidence,
               :found_in_document, :grounded, :guardrail_reason, :run_id, :extracted_at, :current_value)
           ON CONFLICT(doc_id, variable_key) DO UPDATE SET
               label=excluded.label, value=excluded.value, normalized_value=excluded.normalized_value,
               rationale=excluded.rationale, cited_chunk_ids_json=excluded.cited_chunk_ids_json,
               page=excluded.page, section=excluded.section, self_confidence=excluded.self_confidence,
               final_confidence=excluded.final_confidence, found_in_document=excluded.found_in_document,
               grounded=excluded.grounded, guardrail_reason=excluded.guardrail_reason,
               run_id=excluded.run_id, extracted_at=excluded.extracted_at,
               current_value=excluded.current_value, validated=0, validated_by=NULL, validated_at=NULL""",
        payload,
    )
    conn.commit()


def get_variables(doc_id: str) -> list[dict]:
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM variables WHERE doc_id = ? ORDER BY id", (doc_id,)
    ).fetchall()
    return [dict(r) for r in rows]


def get_variable(doc_id: str, variable_key: str) -> dict | None:
    conn = get_connection()
    row = conn.execute(
        "SELECT * FROM variables WHERE doc_id = ? AND variable_key = ?", (doc_id, variable_key)
    ).fetchone()
    return dict(row) if row else None


def save_variable_edit(doc_id: str, variable_key: str, new_value: str, user: str) -> None:
    """Human-in-the-loop update: writes the new current_value, marks the
    variable validated, and appends one audit row recording whether the
    value actually changed ('edit') or was accepted as-is ('validate')."""
    conn = get_connection()
    existing = get_variable(doc_id, variable_key)
    old_value = existing["current_value"] if existing else None
    change_type = "edit" if (old_value or "") != (new_value or "") else "validate"
    now = _now()
    conn.execute(
        """UPDATE variables SET current_value = ?, validated = 1, validated_by = ?, validated_at = ?
           WHERE doc_id = ? AND variable_key = ?""",
        (new_value, user, now, doc_id, variable_key),
    )
    conn.execute(
        """INSERT INTO variable_audit (doc_id, variable_key, changed_from, changed_to, change_type, user, changed_at)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (doc_id, variable_key, old_value, new_value, change_type, user, now),
    )
    conn.commit()


def get_audit_trail(doc_id: str) -> list[dict]:
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM variable_audit WHERE doc_id = ? ORDER BY changed_at", (doc_id,)
    ).fetchall()
    return [dict(r) for r in rows]


# --- submissions -------------------------------------------------------------

def submit_document(doc_id: str, submitted_by: str) -> dict:
    conn = get_connection()
    variables = get_variables(doc_id)
    snapshot = {v["variable_key"]: v["current_value"] for v in variables}
    now = _now()
    conn.execute(
        """INSERT INTO submissions (doc_id, submitted_by, submitted_at, variable_count, snapshot_json)
           VALUES (?, ?, ?, ?, ?)""",
        (doc_id, submitted_by, now, len(variables), json.dumps(snapshot)),
    )
    for v in variables:
        conn.execute(
            """INSERT INTO variable_audit (doc_id, variable_key, changed_from, changed_to, change_type, user, changed_at)
               VALUES (?, ?, ?, ?, 'submit', ?, ?)""",
            (doc_id, v["variable_key"], v["current_value"], v["current_value"], submitted_by, now),
        )
    conn.commit()
    return {"doc_id": doc_id, "submitted_at": now, "variable_count": len(variables)}


def get_submissions(doc_id: str) -> list[dict]:
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM submissions WHERE doc_id = ? ORDER BY submitted_at DESC", (doc_id,)
    ).fetchall()
    return [dict(r) for r in rows]


# --- logs --------------------------------------------------------------------

def log_event(event: str, run_id: str = "", doc_id: str = "", **fields) -> None:
    conn = get_connection()
    conn.execute(
        "INSERT INTO logs (timestamp, run_id, doc_id, event, fields_json) VALUES (?, ?, ?, ?, ?)",
        (_now(), run_id, doc_id, event, json.dumps(fields, default=str)),
    )
    conn.commit()


def get_logs(run_id: str | None = None, limit: int = 500) -> list[dict]:
    conn = get_connection()
    if run_id:
        rows = conn.execute(
            "SELECT * FROM logs WHERE run_id = ? ORDER BY id", (run_id,)
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM logs ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]


# --- raw DB browser / cross-document quality dashboard -----------------------

_ALLOWED_TABLES = {"documents", "variables", "variable_audit", "submissions", "logs"}


def get_all_rows(table: str, limit: int = 1000) -> list[dict]:
    """Raw read of an entire table, for the Database screen. `table` is
    checked against a whitelist before being spliced into SQL, since
    sqlite3 can't parameterize identifiers."""
    if table not in _ALLOWED_TABLES:
        raise ValueError(f"Unknown table: {table}")
    conn = get_connection()
    rows = conn.execute(f"SELECT * FROM {table} ORDER BY rowid DESC LIMIT ?", (limit,)).fetchall()
    return [dict(r) for r in rows]


def get_all_variables_with_doc() -> list[dict]:
    """variables joined with documents.original_filename, for the automatic
    cross-document extraction-quality dashboard on the Evaluation Metrics
    screen (no test-set/ground-truth dependency)."""
    conn = get_connection()
    rows = conn.execute(
        """SELECT v.*, d.original_filename FROM variables v
           JOIN documents d ON d.doc_id = v.doc_id
           ORDER BY v.doc_id, v.id"""
    ).fetchall()
    return [dict(r) for r in rows]
