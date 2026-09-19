"""Shared configuration: paths, model settings, and the Claude/Chroma client helpers."""
import os
from functools import lru_cache
from pathlib import Path

from langchain_anthropic import ChatAnthropic

BASE_DIR = Path(__file__).resolve().parent

PDF_PATH = BASE_DIR / "Guidance for IndustryBiologicals.pdf"
CHROMA_DIR = str(BASE_DIR / "chroma_db")
LOG_DIR = BASE_DIR / "logs"
COLLECTION_NAME = "guidance"

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
CLAUDE_MODEL = os.environ.get("GUIDANCE_AGENT_MODEL", "claude-sonnet-5")

TOP_K = 5
NUMERIC_MATCH_TOLERANCE = 0.02  # 2% relative tolerance when comparing SI values

# Cosine similarity -> max allowed confidence band, per CLAUDE.md's
# "low retrieval similarity caps the max reported confidence" rule.
SIMILARITY_CONFIDENCE_CAPS = [
    (0.50, 100),
    (0.35, 75),
    (0.20, 50),
    (0.0, 30),
]

MIN_SIMILARITY_FOR_CONTEXT = 0.15


def get_anthropic_credentials() -> dict:
    api_key = os.environ.get("ANTHROPIC_AUTH_TOKEN") or os.environ.get("ANTHROPIC_API_KEY")
    base_url = os.environ.get("ANTHROPIC_BASE_URL")
    if not api_key:
        raise RuntimeError(
            "No Anthropic credentials found. Set ANTHROPIC_AUTH_TOKEN or ANTHROPIC_API_KEY."
        )
    kwargs = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    return kwargs


def get_llm(max_tokens: int = 1024, temperature: float = 0.0) -> ChatAnthropic:
    return ChatAnthropic(
        model=CLAUDE_MODEL,
        max_tokens=max_tokens,
        temperature=temperature,
        **get_anthropic_credentials(),
    )


def similarity_to_confidence_cap(similarity: float) -> int:
    for threshold, cap in SIMILARITY_CONFIDENCE_CAPS:
        if similarity >= threshold:
            return cap
    return SIMILARITY_CONFIDENCE_CAPS[-1][1]


@lru_cache(maxsize=1)
def get_chroma_collection():
    import chromadb
    from chromadb.utils import embedding_functions

    client = chromadb.PersistentClient(path=CHROMA_DIR)
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL_NAME)
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"},
    )
