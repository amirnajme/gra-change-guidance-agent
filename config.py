"""Shared configuration: paths, model settings, and confidence-scoring
constants. Client construction (Claude, Chroma, SQLite) lives in
llm.py/vectorstore.py/db.py.
"""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

UPLOADS_DIR = BASE_DIR / "uploads"
CHROMA_DIR = str(BASE_DIR / "chroma_db")
DATA_DIR = BASE_DIR / "data"
DB_PATH = DATA_DIR / "gra_agent.db"

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


def similarity_to_confidence_cap(similarity: float) -> int:
    for threshold, cap in SIMILARITY_CONFIDENCE_CAPS:
        if similarity >= threshold:
            return cap
    return SIMILARITY_CONFIDENCE_CAPS[-1][1]
