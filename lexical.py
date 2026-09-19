"""Minimal in-process BM25 lexical index over the ingested chunks. Pure
vector similarity misses exact-phrase/section-heading queries on this
document (short, template-heavy chunks embed too similarly to each other),
so retrieval merges this lexical signal with the embedding search.
"""
import math
import re
from collections import Counter
from functools import lru_cache

from config import get_chroma_collection

_TOKEN_RE = re.compile(r"[a-z0-9]+")
K1 = 1.5
B = 0.75


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class BM25Index:
    def __init__(self, doc_ids: list[str], doc_tokens: list[list[str]]):
        self.doc_ids = doc_ids
        self.doc_tokens = doc_tokens
        self.doc_len = [len(toks) for toks in doc_tokens]
        self.avgdl = sum(self.doc_len) / len(self.doc_len) if doc_tokens else 0.0
        self.term_freqs = [Counter(toks) for toks in doc_tokens]

        df = Counter()
        for tf in self.term_freqs:
            for term in tf:
                df[term] += 1
        n = len(doc_tokens)
        self.idf = {term: math.log(1 + (n - freq + 0.5) / (freq + 0.5)) for term, freq in df.items()}

    def score_all(self, query: str) -> dict[str, float]:
        q_terms = _tokenize(query)
        scores: dict[str, float] = {}
        for idx, doc_id in enumerate(self.doc_ids):
            tf = self.term_freqs[idx]
            dl = self.doc_len[idx] or 1
            score = 0.0
            for term in q_terms:
                if term not in tf:
                    continue
                idf = self.idf.get(term, 0.0)
                freq = tf[term]
                score += idf * (freq * (K1 + 1)) / (freq + K1 * (1 - B + B * dl / self.avgdl))
            if score > 0:
                scores[doc_id] = score
        return scores

    def top_n(self, query: str, n: int) -> list[tuple[str, float]]:
        scores = self.score_all(query)
        return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:n]


@lru_cache(maxsize=1)
def get_bm25_index() -> BM25Index:
    collection = get_chroma_collection()
    items = collection.get(include=["documents"])
    doc_ids = items["ids"]
    doc_tokens = [_tokenize(doc) for doc in items["documents"]]
    return BM25Index(doc_ids, doc_tokens)
