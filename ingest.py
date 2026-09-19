"""Standalone ingestion script: parse the guidance PDF into section-aware
chunks (keeping bounding boxes for later highlighting), extract numeric/unit
mentions, embed, and persist into ChromaDB.

Run manually: `python ingest.py`. Not invoked by the Streamlit app.
"""
import json
import re

import pymupdf

from config import COLLECTION_NAME, PDF_PATH, get_chroma_collection
from units import extract_numeric_mentions

MAX_CHUNK_CHARS = 1000
HEADER_RE = re.compile(r"^\d+(\.\d+)*\s+[A-Z][A-Za-z /()&,-]{2,80}$")


def _is_header(line: str) -> bool:
    line = line.strip()
    return bool(line) and len(line) <= 90 and bool(HEADER_RE.match(line))


def build_chunks(pdf_path=PDF_PATH):
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

    doc.close()
    return chunks


def main():
    chunks = build_chunks()
    print(f"Parsed {len(chunks)} chunks from {PDF_PATH.name}")

    collection = get_chroma_collection()
    existing = collection.count()
    if existing:
        print(f"Collection '{COLLECTION_NAME}' already has {existing} items; clearing first.")
        collection.delete(where={"source_file": PDF_PATH.name})

    collection.add(
        ids=[c["id"] for c in chunks],
        documents=[c["text"] for c in chunks],
        metadatas=[
            {
                "page": c["page"],
                "section": c["section"],
                "bboxes_json": json.dumps(c["bboxes"]),
                "numeric_json": json.dumps(c["numeric_mentions"]),
                "source_file": PDF_PATH.name,
            }
            for c in chunks
        ],
    )
    print(f"Embedded and stored {len(chunks)} chunks into ./chroma_db (collection='{COLLECTION_NAME}').")


if __name__ == "__main__":
    main()
