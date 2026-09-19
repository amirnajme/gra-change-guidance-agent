# GRA Change Guidance Agent

## Purpose
A Retrieval-Augmented Generation (RAG) agent that answers questions about a
pharma regulatory guidance document (country-specific pharma guidance PDF,
e.g. `Guidance for IndustryBiologicals.pdf`). Answers must be grounded
strictly in the document — no hallucination, no outside knowledge.

## Core Requirements

1. **Input**: A single PDF guidance document (pharma-specific, regulatory).
2. **Q&A**: User asks natural-language questions; agent answers using only
   content retrieved from the PDF.
3. **Anti-hallucination**: If the answer isn't in the document, say so
   explicitly rather than guessing. Every answer must be traceable to a
   specific retrieved chunk/section.
4. **Confidence score**: Every response includes a confidence score
   (0–100% or low/medium/high) reflecting retrieval relevance + answer
   groundedness.
5. **Source highlighting**: The exact section/passage used to generate the
   answer must be shown to the user, highlighted (yellow) in the original
   document text.
6. **Observability**: Log every step of the pipeline (query, retrieved
   chunks, prompt sent to LLM, raw response, latency, token usage) for
   debugging and audit.
7. **Evaluation**: A repeatable way to measure answer quality (groundedness,
   relevance, retrieval precision) against a test set of Q&A pairs.
8. **Semantic search over literal match**: Retrieval must find relevant
   passages even when the query's wording/units differ from the document's
   (e.g. querying "15km" should retrieve a passage stating "15000m" if that's
   the semantically/numerically relevant value). Don't rely on plain keyword
   or exact-string matching alone.

## Architecture

```
PDF ─▶ Ingestion (parse, chunk, embed) ─▶ ChromaDB (vector store)
                                              │
User Query ─▶ Embed query ─▶ Retrieve top-k chunks ─▶ LLM (Claude) ─▶ Answer + Confidence
                                              │
                                    Chunk metadata (page, offsets) ─▶ Highlight source in UI
```

### Orchestration: LangGraph

The query-time pipeline is a LangGraph graph, not a linear script, so each
step is independently observable/testable and can loop/retry.

- **Nodes**: `normalize_query` (unit/numeric normalization) →
  `retrieve` (hybrid vector + numeric search) → `check_context` (is
  retrieved context sufficient?) → `generate_answer` (Claude, cites
  chunks) → `score_confidence` → `guardrail_check` (hallucination /
  groundedness check against retrieved chunks) → end.
  - `guardrail_check` can route back to `retrieve` with a widened k, or
    straight to an "insufficient context" response, instead of returning
    an unverified answer.
- **State**: query, normalized query, retrieved chunks + scores, draft
  answer, citations, confidence score, guardrail verdict.
- **Required hooks** (LangGraph callbacks, invoked around node execution):
  - **Pre-retrieval hook**: logs/records the normalized query before
    search runs.
  - **Post-retrieval hook**: records retrieved chunk IDs, scores, and
    metadata for observability (feeds the logging pipeline below).
  - **Post-generation hook**: runs the groundedness/hallucination check
    on the draft answer against cited chunks before it's shown to the
    user.
  - **On-error hook**: catches node failures (e.g. Claude API error,
    empty retrieval) and routes to a graceful "couldn't find an answer"
    response instead of crashing the app.
  - All hooks write to the same structured observability log (see
    Observability below) so a full run is reconstructable from logs.

### Components

- **Ingestion pipeline**
  - Parse PDF into text, preserving page numbers and section headers.
  - Chunk by section/paragraph (not fixed-size only) so highlighted
    sections are meaningful to a reader.
  - Store chunk text + metadata (page number, section title, char offsets)
    alongside embeddings.

- **Vector store**: ChromaDB (local, persistent client).
  - One collection per document.
  - Metadata fields: `page`, `section`, `start_offset`, `end_offset`,
    `source_file`.

- **Retrieval**: Top-k semantic search (k configurable, default 5),
  optionally with a reranking step to improve precision.
  - Embedding-based similarity alone won't reliably equate values across
    units (km vs m, mg vs g, %, etc). Normalize numeric/unit mentions
    during ingestion (e.g. extract value + unit, store a normalized SI
    value in metadata) and normalize the query the same way, so a query
    like "15 km" can match a chunk containing "1500 m" via a numeric
    comparison, not just vector distance.
  - Combine vector search with this unit-aware numeric matching (hybrid
    retrieval) rather than depending on one method exclusively.

- **Generation**: Claude, with a strict system prompt that:
  - Instructs the model to answer *only* from provided context chunks.
  - Requires the model to say "not found in document" when context is
    insufficient.
  - Requires the model to cite which chunk(s)/page(s) it used.
  - Requires the model to self-report a confidence score based on how
    directly the retrieved context supports the answer.

- **Confidence scoring**: Combine two signals:
  - Retrieval similarity score(s) of the chunks actually cited.
  - Model's self-assessed groundedness (structured output field).
  - Final score is a blend/rule (e.g. low retrieval similarity caps the
    max reported confidence).

- **Observability**:
  - Structured logs (JSON) per query: timestamp, query, retrieved chunk
    IDs + scores, prompt, response, confidence, latency, token counts.
  - Prefer a lightweight tracing tool (e.g. LangSmith, Langfuse, or plain
    structured logging to file/SQLite) — pick one and keep it simple.
  - Every run should be inspectable after the fact without re-running.

- **Evaluation**:
  - Maintain a small labeled test set (question, expected answer/section)
    in a checked-in file (e.g. `eval/testset.jsonl`).
  - Run evaluation via a dedicated **evaluation subagent**, separate from
    the main answer-generation agent, so judging isn't done by the same
    prompt/model call that produced the answer (avoids the model grading
    its own work).
    - Input: question, generated answer, cited chunks, expected
      answer/section from the test set.
    - Output (structured): retrieval hit-rate (did top-k contain the
      right section?), groundedness verdict (is the answer supported by
      the cited text, yes/no + reason), answer relevance score, and a
      pass/fail per test case.
    - Implement as a separate LangGraph graph (or a distinct node/branch
      invoked only in eval mode) using an LLM-as-judge prompt — kept
      independent of the live chat graph so eval runs don't affect
      production latency or logs.
  - Run manually or on ingestion/prompt pipeline changes — not required
    on every commit. Aggregate results into a report (pass rate per
    metric) rather than one-off pass/fail prints.

## UI (Streamlit)

A sidebar selector switches between two screens:

1. **Assistant screen**: single screen, chat on top, response + source
   side by side below it:
   - **Chat pane** (top): Standard chatbot UI — message history, input
     box where the user types questions.
   - **Response box** (bottom-left) and **Source box** (bottom-right):
     side-by-side bordered containers below the chat. The response box
     shows the selected answer + confidence score; the source box renders
     the document page with the exact passage used to answer it
     highlighted in yellow. Asking a new question, or picking an older
     answer from the chat history, updates both boxes together, with no
     page/screen navigation.
   - Use `st.session_state` to link the selected chat message to the
     response/source boxes.
2. **Evaluation Metrics screen**: observability view over the evaluation
   subagent's output — aggregate metrics (retrieval hit-rate, groundedness
   rate, average relevance) plus a per-case breakdown (question, answer,
   confidence, hit/groundedness/relevance, judge reason) for the most
   recent (or a past) run, with a button to trigger a fresh evaluation run
   on demand.

## Tech Stack

- **Language**: Python
- **Orchestration**: LangGraph (query-time pipeline graph + hooks)
- **Vector DB**: ChromaDB (persistent local instance)
- **LLM**: Claude (Anthropic API)
- **PDF parsing**: `pymupdf` (fitz) — preferred for getting page/coordinate
  metadata needed for highlighting; fallback `pdfplumber`.
- **UI**: Streamlit
- **Embeddings**: any Claude-compatible embedding model or a local
  sentence-transformers model — decide based on what's available; keep
  consistent between ingestion and query time.

## Non-Goals / Constraints

- No answers from general world knowledge — document-grounded only.
- No multi-document support required (single guidance PDF for now).
- Don't over-engineer: no need for multi-tenant auth, streaming
  infrastructure, or production deployment concerns unless requested.

## Working Conventions

- Keep ingestion (build the ChromaDB collection) as a separate script from
  the Streamlit app — don't re-embed the PDF on every app run.
- Persist ChromaDB to disk (e.g. `./chroma_db/`) so the app starts fast.
- Treat this CLAUDE.md as the source of truth for scope; update it if
  requirements change.
