# GRA Change Guidance Agent

A regulatory-operations tool that ingests an uploaded pharma guidance PDF
and extracts a standardized list of critical GRA (Global Regulatory
Affairs) submission variables from it — storage conditions, shelf-life,
change notification timelines, dossier format, and more. A human reviewer
validates or edits each extracted value side-by-side with its highlighted
source passage before submitting the final values into a structured,
audited SQLite database. Extractions are grounded strictly in the
document — no outside knowledge, no hallucination — and every value comes
with a confidence score and a traceable source.

See [CLAUDE.md](CLAUDE.md) for the full spec this implements.

## Features

- **Dynamic upload** — any pharma guidance PDF can be uploaded at runtime;
  documents are identified by content hash, so re-uploading the same file
  reuses prior extraction work instead of re-ingesting.
- **Standardized variable extraction** — a fixed list of ~14 critical
  regulatory-submission variables (`variable_schema.py`) is extracted per
  document, one retrieval + extraction pass per variable; if a variable
  isn't discussed in the document, the agent says so instead of guessing.
- **Confidence score** on every extracted variable, blending retrieval
  similarity with the model's self-assessed groundedness.
- **Source highlighting** — the exact passage used to extract a variable
  is rendered from the original PDF page with a yellow highlight,
  side-by-side with the extracted value.
- **Human-in-the-loop validation** — reviewers can edit any extracted
  value or mark it validated as-is; every change is appended to an audit
  trail (prior value, new value, who, when).
- **Structured, audited storage** — final values, per-variable audit
  history, and submission snapshots are all persisted to a single local
  SQLite database (`./data/gra_agent.db`).
- **Running commentary** — ingestion and per-variable extraction progress
  is shown step-by-step in the UI, not behind a silent spinner.
- **Hybrid retrieval** — vector search (ChromaDB + sentence-transformers)
  merged with a BM25 lexical index via Reciprocal Rank Fusion, plus
  unit-aware numeric matching (e.g. a query for "15km" matches a chunk
  stating "1500m").
- **Anti-hallucination guardrail** — a second, independent LLM call checks
  that every extracted value is supported by the cited excerpts before
  it's persisted.
- **Observability** — every pipeline step (query, retrieved chunks,
  prompt, response, latency, token usage) is logged into the same SQLite
  database (`logs` table).
- **Automatic live quality dashboard** — found-in-document rate,
  groundedness rate, and average confidence render immediately on the
  Evaluation Metrics screen, computed straight from every extraction the
  pipeline has actually run — no button, no dependency on a fixed test set.
- **Evaluation subagent (optional benchmark)** — an independent LLM-as-judge
  scores extractions against a labeled test set (retrieval hit-rate,
  groundedness, value correctness), runnable from the CLI or from a
  secondary section of the UI.
- **Raw database browser** — a Database screen in the UI lets you inspect
  every SQLite table (`documents`, `variables`, `variable_audit`,
  `submissions`, `logs`) across all uploaded documents, without a separate
  `sqlite3` session.

## Architecture

```
Upload PDF ─▶ Ingestion (parse, chunk, embed) ─▶ ChromaDB (per-doc_id collection)
                                                       │
For each critical variable ─▶ Retrieve top-k chunks ─▶ LLM (Claude) extracts value
                                                       │
                                             Confidence + guardrail check
                                                       │
                                          SQLite (variables, audit, logs)
                                                       │
                                    Human review/edit ─▶ Submit ─▶ submissions table
```

The extraction pipeline is a [LangGraph](https://github.com/langchain-ai/langgraph)
graph (`graph.py`): `normalize_query` → `retrieve` (hybrid vector + BM25 +
numeric matching, scoped to the document's `doc_id`) → `check_context` →
`extract_variable` (Claude, cites chunks, self-reports confidence) →
`score_confidence` → `guardrail` (groundedness check) → end, with
on-error fallbacks at each stage so a failure never crashes the app. One
pass through the graph extracts one variable; `run_all_extractions` loops
it over the full `CRITICAL_VARIABLES` list for a document.

## Project layout

```
config.py            # paths, model/retrieval constants, similarity_to_confidence_cap()
db.py                # SQLite layer: documents, variables, audit trail, submissions, logs
variable_schema.py   # the fixed list of critical GRA regulatory variables
llm.py               # Claude client (cached) + shared structured-output retry wrapper
vectorstore.py       # per-doc_id Chroma collection handle + cached chunk read
units.py             # numeric + unit extraction/normalization (km/m, mg/g, months/years, %, degC)
observability.py     # SQLite-backed event logger + LangGraph callback hooks
ingest.py            # upload -> chunks (with bboxes) -> embed -> ChromaDB (+ CLI wrapper)
lexical.py           # per-doc_id BM25 index used for hybrid retrieval
retrieval.py         # hybrid vector + BM25 + numeric-match retrieval algorithm (RRF)
extraction.py        # per-variable extraction LLM call (hardened prompt)
generation.py        # groundedness-guardrail LLM call (hardened prompt)
graph.py             # LangGraph pipeline (nodes, state, routing) + run_extraction()/run_all_extractions()
app.py               # Streamlit entry point: sidebar screen switch + PwC theming
ui/
  pdf_view.py                # PDF page rendering with yellow highlight
  home_screen.py              # upload + submit-for-processing, running commentary
  variables_screen.py         # table of all variables + inline edit/validate panel + Submit action
  database_screen.py           # raw browser over every SQLite table, all documents
  eval_screen.py                # automatic live quality dashboard + optional labeled-benchmark section
eval/
  testset.jsonl        # labeled variable-extraction test set
  eval_agent.py         # independent LLM-as-judge subagent
  run_eval.py           # CLI / library entry point that runs the test set and judges it
  archive/             # superseded Q&A-era eval reports
chroma_db/           # persisted vector store (created on ingest, gitignored)
data/                # SQLite database (gitignored)
uploads/             # saved copies of uploaded PDFs, keyed by doc_id (gitignored)
```

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Set Anthropic credentials in the environment:

```bash
export ANTHROPIC_AUTH_TOKEN=your_api_key   # or ANTHROPIC_API_KEY
export ANTHROPIC_BASE_URL=https://your-proxy   # optional, if not using api.anthropic.com directly
```

## Usage

**Run the app:**

```bash
streamlit run app.py
```

The UI has four screens, selected from the sidebar:

- **Home** — upload a guidance PDF and click "Submit for Processing";
  running commentary shows ingestion and per-variable extraction progress.
  Previously processed documents can be reopened from here too.
- **Variables** — every extracted variable in a single table (value,
  confidence, found/grounded/validated status, page) plus an "Edit" button
  per row. Clicking Edit opens an editable value/rationale panel directly
  below, side-by-side with the highlighted PDF source passage; edit the
  value and click "Save & Mark Validated", or step through variables with
  Previous/Next — no navigation away from the table. "Submit All Values" at
  the bottom snapshots current values into the `submissions` table.
- **Database** — a raw browser over every SQLite table (`documents`,
  `variables`, `variable_audit`, `submissions`, `logs`), across all
  uploaded documents.
- **Evaluation Metrics** — an automatic live quality dashboard (found rate,
  groundedness rate, average confidence, overall and per-document) that
  renders immediately from real pipeline data, no button required. An
  optional "Labeled test-set benchmark" expander still runs the independent
  evaluation subagent against `eval/testset.jsonl` on demand.

**Ingest a PDF from the CLI (optional, e.g. for scripting/eval setup):**

```bash
python ingest.py "Guidance for IndustryBiologicals.pdf"
```

**Run evaluation from the CLI (optional):**

```bash
python eval/run_eval.py
```

Runs the labeled test set (`eval/testset.jsonl`) through the live
extraction pipeline, judges each result with the independent eval
subagent, prints a summary, and saves a timestamped report to
`eval/results_*.json`.

## Tech stack

Python · LangGraph · ChromaDB · SQLite · Claude (Anthropic API) · PyMuPDF ·
sentence-transformers · Streamlit · pandas
