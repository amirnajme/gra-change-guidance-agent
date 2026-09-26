"""PDF ingestion: parse into section-aware chunks (keeping bounding boxes
for later highlighting), extract numeric/unit mentions, embed, and persist
into a per-document ChromaDB collection. `ingest_document` is called
directly by the Streamlit Home screen on every upload (with a progress
callback for the running-commentary UI); `main()` is a thin CLI wrapper
kept for manual/ad-hoc use.
"""
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Callable, Optional

import pymupdf

import db
from config import UPLOADS_DIR
from units import extract_numeric_mentions
from vectorstore import get_chroma_collection

MAX_CHUNK_CHARS = 1000
HEADER_RE = re.compile(r"^\d+(\.\d+)*\s+[A-Z][A-Za-z /()&,-]{2,80}$")


def _is_header(line: str) -> bool:
    line = line.strip()
    return bool(line) and len(line) <= 90 and bool(HEADER_RE.match(line))


def build_chunks(pdf_path: Path):
    if not Path(pdf_path).exists():
        raise SystemExit(f"PDF not found at {pdf_path}.")
    doc = pymupdf.open(pdf_path)
    chunks = []
    current_section = "Preamble"

    for page_num, page in enumerate(doc):
        blocks = [b for b in page.get_text("blocks") if b[6] == 0 and b[4].strip()]
        blocks.sort(key=lambda b: (round(b[1]), b[0]))

        buf_text, buf_bboxes = [], []

        def flush():
            if not buf_text:
                return
            text = "\n".join(buf_text).strip()
            if not text:
                return
            chunks.append(
                {
                    "id": f"p{page_num}_{len(chunks)}",
                    "text": text,
                    "page": page_num,
                    "section": current_section,
                    "bboxes": [list(b) for b in buf_bboxes],
                    "numeric_mentions": [m.__dict__ for m in extract_numeric_mentions(text)],
                }
            )

        for b in blocks:
            block_text = b[4].strip()
            first_line = block_text.splitlines()[0].strip() if block_text else ""
            if _is_header(first_line):
                flush()
                buf_text, buf_bboxes = [], []
                current_section = first_line
                remainder = "\n".join(block_text.splitlines()[1:]).strip()
                if remainder:
                    buf_text.append(remainder)
                    buf_bboxes.append(b[:4])
                continue

            buf_text.append(block_text)
            buf_bboxes.append(b[:4])

            if sum(len(t) for t in buf_text) >= MAX_CHUNK_CHARS:
                flush()
                buf_text, buf_bboxes = [], []

        flush()

    page_count = doc.page_count
    doc.close()
    return chunks, page_count


def compute_doc_id(file_bytes: bytes) -> str:
    return hashlib.sha256(file_bytes).hexdigest()[:16]


def ingest_document(
    file_bytes: bytes, original_filename: str, on_progress: Optional[Callable[[str], None]] = None
) -> str:
    """Ingests an uploaded PDF's bytes into its own ChromaDB collection and
    registers it in SQLite. Returns the doc_id. If this exact file was
    already ingested, skips straight to reuse instead of re-embedding."""

    def progress(msg: str) -> None:
        if on_progress:
            on_progress(msg)

    doc_id = compute_doc_id(file_bytes)
    existing = db.get_document(doc_id)
    if existing and existing["status"] in ("ingested", "extracted"):
        progress(f"'{existing['original_filename']}' already ingested — reusing existing data.")
        return doc_id

    progress("Saving upload...")
    UPLOADS_DIR.mkdir(exist_ok=True)
    stored_path = UPLOADS_DIR / f"{doc_id}.pdf"
    stored_path.write_bytes(file_bytes)
    db.upsert_document(doc_id, original_filename, str(stored_path), status="ingesting")

    progress("Parsing PDF into section-aware chunks...")
    chunks, page_count = build_chunks(stored_path)

    progress(f"Embedding and storing {len(chunks)} chunks in the vector store...")
    collection = get_chroma_collection(doc_id)
    if collection.count():
        collection.delete(where={"source_file": original_filename})
    collection.add(
        ids=[c["id"] for c in chunks],
        documents=[c["text"] for c in chunks],
        metadatas=[
            {
                "page": c["page"],
                "section": c["section"],
                "bboxes_json": json.dumps(c["bboxes"]),
                "numeric_json": json.dumps(c["numeric_mentions"]),
                "source_file": original_filename,
            }
            for c in chunks
        ],
    )

    db.mark_ingested(doc_id, page_count=page_count, chunk_count=len(chunks))
    progress(f"Ingestion complete: {page_count} pages, {len(chunks)} chunks.")
    return doc_id


def main():
    """CLI entry point for manual/ad-hoc ingestion: `python ingest.py path/to/file.pdf`."""
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python ingest.py <path-to-pdf>")
    pdf_path = Path(sys.argv[1])
    doc_id = ingest_document(pdf_path.read_bytes(), pdf_path.name, on_progress=print)
    print(f"doc_id={doc_id}")


if __name__ == "__main__":
    main()
