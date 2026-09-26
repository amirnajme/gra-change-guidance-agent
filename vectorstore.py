"""ChromaDB access: a per-document collection handle, and a cached read of
a document's full corpus (documents + metadata + pre-parsed numeric
mentions) used by both the BM25 index and the retrieval module's numeric-
match fallback. Since documents are now uploaded dynamically, both caches
are keyed by doc_id rather than being process-wide singletons, so multiple
uploaded documents can be queried in the same running app without colliding.
"""
import json
from dataclasses import dataclass
from functools import lru_cache

from config import CHROMA_DIR, EMBED_MODEL_NAME
from units import NumericMention


def collection_name(doc_id: str) -> str:
    return f"doc_{doc_id}"


@lru_cache(maxsize=8)
def get_chroma_collection(doc_id: str):
    import chromadb
    from chromadb.utils import embedding_functions

    client = chromadb.PersistentClient(path=CHROMA_DIR)
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL_NAME)
    return client.get_or_create_collection(
        name=collection_name(doc_id),
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"},
    )


@dataclass
class Chunk:
    id: str
    text: str
    page: int
    section: str
    numeric_mentions: list[NumericMention]


@lru_cache(maxsize=8)
def get_all_chunks(doc_id: str) -> tuple[Chunk, ...]:
    """One full-collection read per document per process. Chunk metadata is
    static once ingested, so caching this avoids re-fetching and
    re-JSON-parsing every chunk's numeric_json on every query."""
    collection = get_chroma_collection(doc_id)
    items = collection.get(include=["documents", "metadatas"])
    chunks = []
    for cid, doc, meta in zip(items["ids"], items["documents"], items["metadatas"]):
        mentions = [NumericMention(**m) for m in json.loads(meta.get("numeric_json", "[]"))]
        chunks.append(Chunk(id=cid, text=doc, page=meta["page"], section=meta["section"], numeric_mentions=mentions))
    return tuple(chunks)
