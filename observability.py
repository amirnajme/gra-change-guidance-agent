"""Structured observability: every log record goes into the SQLite `logs`
table (db.py) instead of a JSONL file, plus a LangChain/LangGraph callback
handler that hooks into node/LLM execution automatically (the "required
hooks" from CLAUDE.md: pre-retrieval, post-retrieval, post-generation,
on-error). A single store keeps the whole app's audit trail (uploaded
documents, extracted/validated variables, and pipeline logs) queryable
from one file.
"""
import time
import uuid
from typing import Any

from langchain_core.callbacks.base import BaseCallbackHandler

import db


def log_event(event: str, run_id: str = "", **fields: Any) -> None:
    db.log_event(event, run_id=run_id, doc_id=fields.pop("doc_id", ""), **fields)


def new_run_id() -> str:
    return uuid.uuid4().hex[:12]


class ObservabilityCallbackHandler(BaseCallbackHandler):
    """Logs every graph node and every LLM call that runs under it."""

    def __init__(self, run_id: str, doc_id: str = ""):
        self.run_id = run_id
        self.doc_id = doc_id
        self._starts: dict[str, float] = {}

    def _elapsed_ms(self, key: str) -> float:
        start = self._starts.pop(key, None)
        return round((time.monotonic() - start) * 1000, 1) if start else -1.0

    def on_chain_start(self, serialized, inputs, *, run_id, tags=None, metadata=None, **kwargs):
        name = (metadata or {}).get("langgraph_node") or (serialized or {}).get("name", "chain")
        self._starts[str(run_id)] = time.monotonic()
        log_event("node_start", self.run_id, doc_id=self.doc_id, node=name)

    def on_chain_end(self, outputs, *, run_id, **kwargs):
        elapsed = self._elapsed_ms(str(run_id))
        log_event("node_end", self.run_id, doc_id=self.doc_id, elapsed_ms=elapsed)

    def on_chain_error(self, error, *, run_id, **kwargs):
        elapsed = self._elapsed_ms(str(run_id))
        log_event("node_error", self.run_id, doc_id=self.doc_id, error=str(error), elapsed_ms=elapsed)

    def on_llm_start(self, serialized, prompts, *, run_id, **kwargs):
        self._starts[f"llm-{run_id}"] = time.monotonic()
        log_event("llm_start", self.run_id, doc_id=self.doc_id, prompt_preview=str(prompts)[:500])

    def on_llm_end(self, response, *, run_id, **kwargs):
        elapsed = self._elapsed_ms(f"llm-{run_id}")
        usage = getattr(response, "llm_output", None) or {}
        log_event("llm_end", self.run_id, doc_id=self.doc_id, elapsed_ms=elapsed, usage=usage.get("usage") if usage else None)

    def on_llm_error(self, error, *, run_id, **kwargs):
        elapsed = self._elapsed_ms(f"llm-{run_id}")
        log_event("llm_error", self.run_id, doc_id=self.doc_id, error=str(error), elapsed_ms=elapsed)
