"""Hybrid retrieval: ChromaDB vector search merged with an in-process BM25
index via Reciprocal Rank Fusion, plus a unit-aware numeric-mention match
so a query like "15km" can surface a chunk stating "1500m" even when it
doesn't rank highly (or at all) by embedding similarity or BM25 alone.
"""
from typing import TypedDict

from config import NUMERIC_MATCH_TOLERANCE, TOP_K
from lexical import get_bm25_index
from units import NumericMention, numeric_mentions_match
from vectorstore import get_all_chunks, get_chroma_collection

RRF_K = 60
LEXICAL_ONLY_SIMILARITY = 0.45  # confidence-cap placeholder when a chunk has no vector score


class RetrievedChunk(TypedDict):
    id: str
    text: str
    page: int
    section: str
    similarity: float
    numeric_match: bool
    rrf_score: float


def _cosine_similarity(distance: float) -> float:
    return max(0.0, 1.0 - distance / 2.0)


def retrieve_chunks(doc_id: str, query: str, numeric_mentions: list[dict]) -> list[RetrievedChunk]:
    collection = get_chroma_collection(doc_id)
    vector_result = collection.query(query_texts=[query], n_results=TOP_K)
    bm25_hits = get_bm25_index(doc_id).top_n(query, n=TOP_K * 2)

    vector_rank = {cid: rank for rank, cid in enumerate(vector_result["ids"][0])}
    bm25_rank = {cid: rank for rank, (cid, _score) in enumerate(bm25_hits)}
    candidate_ids = set(vector_rank) | set(bm25_rank)

    similarity_by_id = {
        cid: _cosine_similarity(dist)
        for cid, dist in zip(vector_result["ids"][0], vector_result["distances"][0])
    }

    # Fetch text/metadata for any bm25-only candidates not already returned by the vector query.
    vector_docs = dict(zip(vector_result["ids"][0], vector_result["documents"][0]))
    vector_metas = dict(zip(vector_result["ids"][0], vector_result["metadatas"][0]))
    missing_ids = [cid for cid in candidate_ids if cid not in vector_docs]
    if missing_ids:
        extra = collection.get(ids=missing_ids, include=["documents", "metadatas"])
        vector_docs.update(zip(extra["ids"], extra["documents"]))
        vector_metas.update(zip(extra["ids"], extra["metadatas"]))

    query_mentions = [NumericMention(**m) for m in numeric_mentions]
    mentions_by_id = {c.id: c.numeric_mentions for c in get_all_chunks(doc_id)}
    retrieved: dict[str, RetrievedChunk] = {}

    for cid in candidate_ids:
        meta = vector_metas[cid]
        chunk_mentions = mentions_by_id.get(cid, [])
        rrf_score = 1 / (RRF_K + vector_rank[cid] + 1) if cid in vector_rank else 0.0
        rrf_score += 1 / (RRF_K + bm25_rank[cid] + 1) if cid in bm25_rank else 0.0
        retrieved[cid] = {
            "id": cid,
            "text": vector_docs[cid],
            "page": meta["page"],
            "section": meta["section"],
            "similarity": similarity_by_id.get(cid, LEXICAL_ONLY_SIMILARITY),
            "numeric_match": bool(query_mentions)
            and numeric_mentions_match(query_mentions, chunk_mentions, tolerance=NUMERIC_MATCH_TOLERANCE),
            "rrf_score": rrf_score,
        }

    if query_mentions:
        for chunk in get_all_chunks(doc_id):
            if chunk.id in retrieved:
                continue
            if numeric_mentions_match(query_mentions, chunk.numeric_mentions, tolerance=NUMERIC_MATCH_TOLERANCE):
                retrieved[chunk.id] = {
                    "id": chunk.id,
                    "text": chunk.text,
                    "page": chunk.page,
                    "section": chunk.section,
                    "similarity": LEXICAL_ONLY_SIMILARITY,
                    "numeric_match": True,
                    "rrf_score": 0.0,
                }

    ranked = sorted(retrieved.values(), key=lambda c: (c["numeric_match"], c["rrf_score"]), reverse=True)
    return ranked[: TOP_K + 2]
