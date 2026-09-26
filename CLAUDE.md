# GRA Change Guidance Agent

## Purpose
A regulatory-operations tool that ingests an uploaded pharma guidance PDF
(country-specific pharma guidance document, e.g.
`Guidance for IndustryBiologicals.pdf`) and extracts a fixed set of
critical GRA (Global Regulatory Affairs) submission variables from it, with
a human reviewer validating/editing each extracted value side-by-side with
its highlighted source passage before the final values are pushed into a
structured, audited database. Extracted values must be grounded strictly in
the document — no hallucination, no outside knowledge.

## Core Requirements

1. **Input**: A pharma guidance PDF uploaded at runtime through the UI (not
   a static, hardcoded file). Each uploaded document is identified by a
   content hash (`doc_id`) so re-uploading the same file reuses prior work
   instead of re-ingesting.
2. **Variable extraction**: Extract a standardized list of critical GRA
   regulatory-submission variables (see `variable_schema.py`) from the
   document — one retrieval + extraction pass per variable, not a single
   free-form Q&A pass. There is no open-ended chat mode.
3. **Anti-hallucination**: If a variable isn't discussed in the document,
   say so explicitly ("not found in document") rather than guessing. Every
   extracted value must be traceable to a specific retrieved chunk/section.
4. **Confidence score**: Every extracted variable includes a confidence
   score (0–100) reflecting retrieval relevance + extraction groundedness.
5. **Source highlighting**: The exact section/passage used to extract a
   variable must be shown to the user, highlighted (yellow) in the original
   document page, side-by-side with the extracted value.
6. **Human-in-the-loop validation**: The reviewer can accept or edit each
   extracted value. Edits and validations are appended to an audit trail
   (original LLM value vs. current value, who changed it, when).
7. **Structured, persistent storage**: Final (possibly human-edited) values
   are pushed into a structured local database on submit, together with the
   variable metadata, the audit trail of user modifications, and
   timestamps.
8. **Running commentary**: While a document is being ingested and its
   variables extracted, the UI shows step-by-step progress (not a silent
   spinner) so the user can see what's happening.
9. **Observability**: Log every step of the pipeline (query, retrieved
   chunks, prompt sent to LLM, raw response, latency, token usage) into the
   structured database for debugging and audit — not just to a file.
10. **Evaluation**: A repeatable way to measure extraction quality
    (groundedness, value correctness, retrieval precision) against a test
    set of variable/expected-value pairs.
11. **Semantic search over literal match**: Retrieval must find relevant
    passages even when the query's wording/units differ from the
    document's (e.g. querying "15km" should retrieve a passage stating
    "15000m" if that's the semantically/numerically relevant value). Don't
    rely on plain keyword or exact-string matching alone.

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

### Orchestration: LangGraph

The extraction pipeline is a LangGraph graph, not a linear script, so each
step is independently observable/testable and can loop/retry. One pass
through the graph extracts **one** critical variable from **one** uploaded
document; the app loops it once per variable in the standard list.

- **Nodes**: `normalize_query` (unit/numeric normalization of the
  variable's search query) → `retrieve` (hybrid vector + numeric search,
  scoped to the uploaded document's `doc_id`) → `check_context` (is
  retrieved context sufficient?) → `extract_variable` (Claude, cites
  chunks) → `score_confidence` → `guardrail_check` (hallucination /
  groundedness check against retrieved chunks) → end.
  - `check_context` routes straight to a "not found in document" result
    instead of forcing an extraction attempt when retrieval is too weak.
  - `guardrail_check` demotes the result to "not found" with a capped
    confidence if the groundedness check fails, instead of returning an
    unverified value.
- **State**: `doc_id`, variable definition, normalized query, retrieved
  chunks + scores, extracted value, citations, confidence score, guardrail
  verdict.
- **Required hooks** (LangGraph callbacks, invoked around node execution):
  - **Pre-retrieval hook**: logs the normalized query before search runs.
  - **Post-retrieval hook**: records retrieved chunk IDs, scores, and
    metadata for observability (feeds the logging table below).
  - **Post-generation hook**: runs the groundedness/hallucination check on
    the draft extraction against cited chunks before it's persisted.
  - **On-error hook**: catches node failures (e.g. Claude API error, empty
    retrieval) and routes to a graceful "couldn't extract this variable"
    result instead of crashing the app.
  - All hooks write to the same structured observability log (the `logs`
    table — see Observability below) so a full run is reconstructable
    after the fact.

### Components

- **Ingestion pipeline** (`ingest.py`, triggered by upload, not a
  standalone manual step)
  - Compute `doc_id` from the uploaded file's content hash; short-circuit
    to reuse an already-ingested document instead of re-embedding.
  - Parse PDF into text, preserving page numbers and section headers.
  - Chunk by section/paragraph (not fixed-size only) so highlighted
    sections are meaningful to a reader.
  - Store chunk text + metadata (page number, section title, char offsets,
    bounding boxes for highlighting) alongside embeddings.

- **Vector store**: ChromaDB (local, persistent client).
  - One collection per uploaded document (`doc_{doc_id}`).
  - Metadata fields: `page`, `section`, `start_offset`, `end_offset`,
    `source_file`, `bboxes_json`.

- **Variable schema** (`variable_schema.py`): a fixed, config-driven list
  of ~14 critical GRA submission variables (e.g. regulatory authority,
  storage conditions, shelf-life/retest period, change notification
  timeline, batch size/scale-up limits, stability study duration,
  submission review timeline, dossier format, change classification,
  packaging requirements, labeling requirements, testing requirements,
  applicable fees, effective date). Each entry defines the retrieval query
  and extraction instructions used to pull that one variable out of the
  document; not every document will discuss every variable, and the
  pipeline reports "not found in document" per-variable rather than
  guessing.

- **Retrieval**: Top-k semantic search (k configurable, default 5) per
  variable, scoped to the document's `doc_id`.
  - Embedding-based similarity alone won't reliably equate values across
    units (km vs m, mg vs g, %, etc). Normalize numeric/unit mentions
    during ingestion (e.g. extract value + unit, store a normalized SI
    value in metadata) and normalize each variable's search query the same
    way, so "15 km" can match a chunk containing "1500 m" via a numeric
    comparison, not just vector distance.
  - Combine vector search with this unit-aware numeric matching (hybrid
    retrieval, RRF) rather than depending on one method exclusively.

- **Extraction**: Claude, with a strict system prompt (`extraction.py`)
  that:
  - Instructs the model to extract the variable's value *only* from
    provided context chunks.
  - Requires the model to say "not found in document" when context is
    insufficient.
  - Requires the model to cite which chunk(s)/page(s) it used.
  - Requires the model to self-report a confidence score based on how
    directly the retrieved context supports the extracted value.

- **Confidence scoring**: Combine two signals:
  - Retrieval similarity score(s) of the chunks actually cited.
  - Model's self-assessed groundedness (structured output field).
  - Final score is a blend/rule (e.g. low retrieval similarity caps the
    max reported confidence).

- **Human-in-the-loop validation**: The reviewer can edit any extracted
  value or mark it validated as-is. Every save updates the variable's
  live/current value and appends a row to the audit trail recording the
  prior value, the new value, who made the change, and when.

- **Structured storage**: SQLite (`db.py`, single local file
  `./data/gra_agent.db`) holds documents, extracted variables (LLM value
  vs. current/edited value), the append-only variable audit trail,
  submission snapshots, and the observability log — all in one place.
  Submitting a document snapshots the current values of every variable
  into a `submissions` row, independent of later re-extraction or edits.

- **Observability**:
  - Structured log rows (the `logs` table in the same SQLite database) per
    pipeline step: timestamp, run ID, `doc_id`, event, and JSON fields
    (query, retrieved chunk IDs + scores, prompt, response, confidence,
    latency, token counts).
  - Every run should be inspectable after the fact without re-running.

- **Evaluation**:
  - Maintain a small labeled test set (variable key, expected value/page)
    in a checked-in file (`eval/testset.jsonl`).
  - Run evaluation via a dedicated **evaluation subagent**, separate from
    the main extraction agent, so judging isn't done by the same
    prompt/model call that produced the extraction (avoids the model
    grading its own work).
    - Input: variable definition, generated extraction (value, citations,
      confidence), expected value/section from the test set.
    - Output (structured): retrieval hit-rate (did top-k contain the right
      section?), groundedness verdict (is the value supported by the cited
      text, yes/no + reason), value-correctness score, and a pass/fail per
      test case.
    - Implemented as a distinct module (`eval/eval_agent.py`) using an
      LLM-as-judge prompt, invoked only by `eval/run_eval.py` — kept
      independent of the live extraction graph so eval runs don't affect
      production latency or logs.
  - Run manually or on ingestion/prompt pipeline changes — not required on
    every commit. Aggregate results into a report (pass rate per metric)
    rather than one-off pass/fail prints.

## UI (Streamlit)

A sidebar selector switches between four screens. There is no open-ended
chat/Q&A screen, and no separate "Key Variables" middle-man screen — the
overview and the edit/validate flow live on one screen so editing a value
never requires an extra navigation hop.

1. **Home**: upload a guidance PDF and a "Submit for Processing" button.
   While ingestion and per-variable extraction run, the screen shows
   running commentary (step-by-step progress messages), not a silent
   spinner. Also lists previously processed documents so a reviewer can
   reopen one without re-uploading.
2. **Variables**: a single table of every extracted variable (value,
   confidence, found/grounded/validated status, page) at the top, with an
   "Edit" button per row. Clicking Edit opens an editable value/rationale
   panel directly below the table, side-by-side with the highlighted PDF
   source passage (same highlight mechanism used throughout the app);
   Previous/Next steps through the full variable list without leaving the
   screen. The final "Submit All Values" button lives at the bottom of this
   same screen — it snapshots the current values of every variable into the
   `submissions` table.
3. **Database**: a raw browser over every SQLite table (`documents`,
   `variables`, `variable_audit`, `submissions`, `logs`), across all
   uploaded documents — full auditability without a separate `sqlite3` CLI
   session.
4. **Evaluation Metrics**: an automatic, always-on quality dashboard
   computed directly from every extraction the live pipeline has actually
   run (found-in-document rate, groundedness rate, average confidence,
   overall and per-document) — renders immediately on page load, no button,
   no dependency on a fixed labeled test set. A secondary, clearly-optional
   "Labeled test-set benchmark" section (inside an expander) still runs the
   independent evaluation subagent (`eval/run_eval.py` +
   `eval/eval_agent.py`) against the checked-in `eval/testset.jsonl` on
   demand, for regression-testing prompt/retrieval changes against known
   ground truth.

## Tech Stack

- **Language**: Python
- **Orchestration**: LangGraph (per-variable extraction pipeline graph +
  hooks)
- **Vector DB**: ChromaDB (persistent local instance, one collection per
  uploaded document)
- **Structured storage**: SQLite (`./data/gra_agent.db`) — variables, audit
  trail, submissions, observability logs
- **LLM**: Claude (Anthropic API)
- **PDF parsing**: `pymupdf` (fitz) — preferred for getting page/coordinate
  metadata needed for highlighting; fallback `pdfplumber`.
- **UI**: Streamlit, PwC-styled theme (orange/black color theme + text
  wordmark; no image logo asset)
- **Embeddings**: any Claude-compatible embedding model or a local
  sentence-transformers model — decide based on what's available; keep
  consistent between ingestion and query time.

## Non-Goals / Constraints

- No answers from general world knowledge — document-grounded only.
- No open-ended Q&A/chat mode — extraction is scoped to the fixed critical
  variable list.
- Multi-document support is limited to "one document open at a time in the
  UI, many documents persisted"; no cross-document comparison required.
- Don't over-engineer: no need for multi-tenant auth, streaming
  infrastructure, or production deployment concerns unless requested.

## Working Conventions

- Keep ingestion (`ingest_document`) callable both from the Streamlit
  upload flow and as a standalone CLI (`python ingest.py <path>`) — don't
  force a UI session just to (re-)embed a document.
- Persist ChromaDB to disk (`./chroma_db/`) and SQLite to disk
  (`./data/gra_agent.db`) so the app starts fast and survives restarts.
- Every document/variable/vector-store lookup is keyed by `doc_id`
  (content hash of the uploaded file) — no process-wide singleton state
  tied to one static PDF.
- Treat this CLAUDE.md as the source of truth for scope; update it if
  requirements change.
