# GRA Change Guidance Agent

A Retrieval-Augmented Generation (RAG) agent that answers questions about a
pharma regulatory guidance document (`Guidance for IndustryBiologicals.pdf`,
a CDSCO/India biologicals guidance PDF). Answers are grounded strictly in
the document — no outside knowledge, no hallucination — and every answer
comes with a confidence score and the exact source passage highlighted in
the original PDF.

See [CLAUDE.md](CLAUDE.md) for the full spec this implements.

## Features

- **Document-grounded Q&A** — answers are generated only from retrieved
  excerpts of the PDF; if the document doesn't contain the answer, the
  agent says so instead of guessing.
- **Confidence score** on every answer, blending retrieval similarity with
  the model's self-assessed groundedness.
- **Source highlighting** — the exact passage used to answer a question is
  rendered from the original PDF page with a yellow highlight.
- **Hybrid retrieval** — vector search (ChromaDB + sentence-transformers)
  merged with a BM25 lexical index via Reciprocal Rank Fusion, plus
  unit-aware numeric matching (e.g. a query for "15km" matches a chunk
  stating "1500m").
- **Anti-hallucination guardrail** — a second, independent LLM call checks
  that every claim in the draft answer is supported by the cited excerpts
  before it's shown to the user.
- **Observability** — every pipeline step (query, retrieved chunks,
  prompt, response, latency, token usage) is logged as structured JSON.
- **Evaluation subagent** — an independent LLM-as-judge scores answers
  against a labeled test set (retrieval hit-rate, groundedness, relevance),
  runnable from the CLI or from the UI.

## Architecture

```
PDF ─▶ Ingestion (parse, chunk, embed) ─▶ ChromaDB (vector store)
                                              │
User Query ─▶ Embed query ─▶ Retrieve top-k chunks ─▶ LLM (Claude) ─▶ Answer + Confidence
                                              │
                                    Chunk metadata (page, bboxes) ─▶ Highlight source in UI
```

The query-time pipeline is a [LangGraph](https://github.com/langchain-ai/langgraph)
graph (`graph.py`): `normalize_query` → `retrieve` (hybrid vector + BM25 +
numeric matching) → `check_context` → `generate_answer` (Claude, cites
chunks, self-reports confidence) → `score_confidence` → `guardrail`
(groundedness check) → end, with on-error fallbacks at each stage so a
failure never crashes the app.

## Project layout

```
config.py            # env/model/path config, get_llm(), get_chroma_collection()
units.py             # numeric + unit extraction/normalization (km/m, mg/g, months/years, %, degC)
observability.py     # JSONL event logger + LangGraph callback hooks
ingest.py            # standalone: PDF -> chunks (with bboxes) -> embed -> ChromaDB
lexical.py           # in-process BM25 index used for hybrid retrieval
graph.py             # LangGraph pipeline (nodes, state, routing) + run_query()
app.py               # Streamlit UI (Assistant screen + Evaluation Metrics screen)
eval/
  testset.jsonl        # labeled Q&A test set
  eval_agent.py         # independent LLM-as-judge subagent
  run_eval.py           # CLI / library entry point that runs the test set and judges it
chroma_db/           # persisted vector store (created by ingest.py, gitignored)
logs/                # structured observability logs (gitignored)
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

**1. Ingest the PDF** (run once, or whenever the source PDF changes):

```bash
python ingest.py
```

This parses `Guidance for IndustryBiologicals.pdf` into section-aware
chunks, extracts numeric/unit mentions, embeds them, and persists them to
`./chroma_db`.

**2. Run the app:**

```bash
streamlit run app.py
```

The UI has two screens, selected from the sidebar:

- **Assistant** — chat input/history on top; below it, a side-by-side
  Response box (answer + confidence) and Source box (highlighted PDF
  page) that update together as you ask questions or revisit earlier
  answers.
- **Evaluation Metrics** — an observability view over the evaluation
  subagent's output: aggregate hit-rate/groundedness/relevance and a
  per-case breakdown, with a button to trigger a fresh evaluation run.

**3. Run evaluation from the CLI (optional):**

```bash
python eval/run_eval.py
```

Runs the labeled test set (`eval/testset.jsonl`) through the live
pipeline, judges each answer with the independent eval subagent, prints a
summary, and saves a timestamped report to `eval/results_*.json`.

## Tech stack

Python · LangGraph · ChromaDB · Claude (Anthropic API) · PyMuPDF ·
sentence-transformers · Streamlit
