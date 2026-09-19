"""LangGraph pipeline: normalize -> retrieve -> check_context -> generate ->
score_confidence -> guardrail -> END. See CLAUDE.md for the spec this
implements, and the approved plan for the guardrail simplification (single
deterministic fallback instead of a re-retrieve loop).
"""
import json
from typing import Any, Optional, TypedDict

from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

from config import (
    MIN_SIMILARITY_FOR_CONTEXT,
    TOP_K,
    get_chroma_collection,
    get_llm,
    similarity_to_confidence_cap,
)
from lexical import get_bm25_index
from observability import ObservabilityCallbackHandler, log_event, new_run_id
from units import NumericMention, extract_numeric_mentions, numeric_mentions_match

RRF_K = 60
LEXICAL_ONLY_SIMILARITY = 0.45  # confidence-cap placeholder when a chunk has no vector score

NOT_FOUND_MESSAGE = (
    "I couldn't find a clearly supported answer to this in the guidance document."
)


class RetrievedChunk(TypedDict):
    id: str
    text: str
    page: int
    section: str
    similarity: float
    numeric_match: bool
    rrf_score: float


class GraphState(TypedDict, total=False):
    query: str
    numeric_mentions: list[dict]
    retrieved: list[RetrievedChunk]
    context_sufficient: bool
    answer: str
    citations: list[str]
    self_confidence: int
    final_confidence: int
    grounded: bool
    guardrail_reason: str
    error: Optional[str]


class AnswerOutput(BaseModel):
    answer: str = Field(description="The answer, using only the provided document excerpts.")
    cited_chunk_ids: list[str] = Field(description="Ids of the excerpts actually used, e.g. ['p78_185'].")
    self_confidence: int = Field(description="0-100: how directly the excerpts support this answer.")
    found_in_document: bool = Field(description="False if the document does not contain the answer.")


class GuardrailOutput(BaseModel):
    grounded: bool = Field(description="True only if every claim in the answer is directly supported by the excerpts.")
    reason: str = Field(description="Short justification.")


def _cosine_similarity(distance: float) -> float:
    return max(0.0, 1.0 - distance / 2.0)


def _invoke_structured(structured_llm, messages, run_id: str = "", node: str = "", max_retries: int = 3):
    """The Claude-compatible proxy occasionally returns a malformed/incomplete
    tool call, which fails pydantic validation. Retry a couple of times
    before giving up, so a transient parse glitch doesn't turn into a false
    "not found in document" refusal."""
    last_exc = None
    for attempt in range(max_retries + 1):
        try:
            return structured_llm.invoke(messages)
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            log_event("structured_output_retry", run_id, node=node, attempt=attempt, error=str(exc))
    raise last_exc


def normalize_query_node(state: GraphState) -> dict:
    mentions = extract_numeric_mentions(state["query"])
    return {"numeric_mentions": [m.__dict__ for m in mentions]}


def retrieve_node(state: GraphState) -> dict:
    try:
        collection = get_chroma_collection()
        vector_result = collection.query(query_texts=[state["query"]], n_results=TOP_K)
        bm25_hits = get_bm25_index().top_n(state["query"], n=TOP_K * 2)

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

        query_mentions = [NumericMention(**m) for m in state.get("numeric_mentions", [])]
        retrieved: dict[str, RetrievedChunk] = {}

        for cid in candidate_ids:
            meta = vector_metas[cid]
            chunk_mentions = [NumericMention(**m) for m in json.loads(meta.get("numeric_json", "[]"))]
            rrf_score = 1 / (RRF_K + vector_rank[cid] + 1) if cid in vector_rank else 0.0
            rrf_score += 1 / (RRF_K + bm25_rank[cid] + 1) if cid in bm25_rank else 0.0
            retrieved[cid] = {
                "id": cid,
                "text": vector_docs[cid],
                "page": meta["page"],
                "section": meta["section"],
                "similarity": similarity_by_id.get(cid, LEXICAL_ONLY_SIMILARITY),
                "numeric_match": bool(query_mentions) and numeric_mentions_match(query_mentions, chunk_mentions),
                "rrf_score": rrf_score,
            }

        if query_mentions:
            all_items = collection.get(include=["documents", "metadatas"])
            for cid, doc, meta in zip(all_items["ids"], all_items["documents"], all_items["metadatas"]):
                if cid in retrieved:
                    continue
                chunk_mentions = [NumericMention(**m) for m in json.loads(meta.get("numeric_json", "[]"))]
                if numeric_mentions_match(query_mentions, chunk_mentions):
                    retrieved[cid] = {
                        "id": cid,
                        "text": doc,
                        "page": meta["page"],
                        "section": meta["section"],
                        "similarity": LEXICAL_ONLY_SIMILARITY,
                        "numeric_match": True,
                        "rrf_score": 0.0,
                    }

        ranked = sorted(retrieved.values(), key=lambda c: (c["numeric_match"], c["rrf_score"]), reverse=True)
        return {"retrieved": ranked[: TOP_K + 2]}
    except Exception as exc:  # on-error hook: never crash the graph
        return {"error": f"retrieve_node: {exc}", "retrieved": []}


def check_context_node(state: GraphState) -> dict:
    if state.get("error"):
        return {"context_sufficient": False}
    retrieved = state.get("retrieved", [])
    best = max((c["similarity"] for c in retrieved), default=0.0)
    has_numeric_hit = any(c["numeric_match"] for c in retrieved)
    return {"context_sufficient": bool(retrieved) and (best >= MIN_SIMILARITY_FOR_CONTEXT or has_numeric_hit)}


def route_after_context(state: GraphState) -> str:
    return "generate_answer" if state.get("context_sufficient") else "no_context"


def no_context_node(state: GraphState) -> dict:
    return {
        "answer": NOT_FOUND_MESSAGE,
        "citations": [],
        "self_confidence": 0,
        "final_confidence": 5,
        "grounded": True,
        "guardrail_reason": "No sufficiently relevant passage was retrieved.",
    }


def generate_answer_node(state: GraphState) -> dict:
    try:
        context = "\n\n".join(
            f"[{c['id']} | page {c['page']} | {c['section']}]\n{c['text']}" for c in state["retrieved"]
        )
        system = (
            "You are a regulatory assistant that answers questions ONLY using the provided "
            "excerpts from a pharma guidance document. Never use outside knowledge. If the "
            "excerpts do not contain the answer, set found_in_document=false and say so in the "
            "answer field. Always list the ids of the excerpts you actually relied on. Do NOT "
            "attribute the answer to a section/clause number unless that exact number appears "
            "together with the answer content in the SAME excerpt — never borrow a heading or "
            "section number from a different excerpt just because it was also retrieved. "
            "Numbers in different units that are mathematically equivalent to the question's "
            "number (e.g. '90 days' and '3 months', '15km' and '1500m') count as directly "
            "answering the question — treat the excerpt's stated value as found_in_document=true "
            "and state the equivalence, rather than saying it wasn't mentioned just because the "
            "unit differs."
        )
        llm = get_llm().with_structured_output(AnswerOutput)
        out: AnswerOutput = _invoke_structured(
            llm,
            [
                ("system", system),
                ("human", f"Document excerpts:\n\n{context}\n\nQuestion: {state['query']}"),
            ],
            node="generate_answer",
        )
        if not out.found_in_document:
            return {
                "answer": NOT_FOUND_MESSAGE,
                "citations": [],
                "self_confidence": 0,
            }
        return {
            "answer": out.answer,
            "citations": out.cited_chunk_ids,
            "self_confidence": max(0, min(100, out.self_confidence)),
        }
    except Exception as exc:
        return {"error": f"generate_answer_node: {exc}"}


def score_confidence_node(state: GraphState) -> dict:
    if state.get("error"):
        return {}
    cited = {c["id"]: c for c in state.get("retrieved", [])}
    cited_similarities = [cited[i]["similarity"] for i in state.get("citations", []) if i in cited]
    best_similarity = max(cited_similarities, default=0.0)
    cap = similarity_to_confidence_cap(best_similarity)
    final = min(state.get("self_confidence", 0), cap)
    return {"final_confidence": final}


def guardrail_node(state: GraphState) -> dict:
    if state.get("error") or not state.get("citations"):
        return {"grounded": True}
    try:
        cited = {c["id"]: c for c in state.get("retrieved", [])}
        excerpts = "\n\n".join(cited[i]["text"] for i in state["citations"] if i in cited)
        llm = get_llm().with_structured_output(GuardrailOutput)
        out: GuardrailOutput = _invoke_structured(
            llm,
            [
                (
                    "system",
                    "Check whether every factual claim in the answer is directly supported by "
                    "the excerpts. Be strict: if the answer adds anything not present in the "
                    "excerpts, grounded must be false. Exception: unit/number conversions are "
                    "allowed and count as grounded — e.g. if an excerpt says '3 months' and the "
                    "answer notes this is equivalent to '90 days', that is a supported "
                    "mathematical restatement, not an unsupported addition.",
                ),
                ("human", f"Excerpts:\n\n{excerpts}\n\nAnswer to check:\n{state['answer']}"),
            ],
            node="guardrail",
        )
        if not out.grounded:
            return {
                "grounded": False,
                "guardrail_reason": out.reason,
                "answer": NOT_FOUND_MESSAGE,
                "citations": [],
                "final_confidence": min(state.get("final_confidence", 0), 20),
            }
        return {"grounded": True, "guardrail_reason": out.reason}
    except Exception as exc:
        return {"error": f"guardrail_node: {exc}"}


def error_fallback_node(state: GraphState) -> dict:
    return {
        "answer": NOT_FOUND_MESSAGE,
        "citations": [],
        "self_confidence": 0,
        "final_confidence": 0,
        "grounded": True,
        "guardrail_reason": f"Pipeline error: {state.get('error')}",
    }


def route_after_generate(state: GraphState) -> str:
    return "error_fallback" if state.get("error") else "score_confidence"


def route_after_guardrail(state: GraphState) -> str:
    return "error_fallback" if state.get("error") else END


def build_graph():
    graph = StateGraph(GraphState)
    graph.add_node("normalize_query", normalize_query_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("check_context", check_context_node)
    graph.add_node("no_context", no_context_node)
    graph.add_node("generate_answer", generate_answer_node)
    graph.add_node("score_confidence", score_confidence_node)
    graph.add_node("guardrail", guardrail_node)
    graph.add_node("error_fallback", error_fallback_node)

    graph.set_entry_point("normalize_query")
    graph.add_edge("normalize_query", "retrieve")
    graph.add_edge("retrieve", "check_context")
    graph.add_conditional_edges(
        "check_context", route_after_context, {"generate_answer": "generate_answer", "no_context": "no_context"}
    )
    graph.add_edge("no_context", END)
    graph.add_conditional_edges(
        "generate_answer", route_after_generate, {"score_confidence": "score_confidence", "error_fallback": "error_fallback"}
    )
    graph.add_edge("score_confidence", "guardrail")
    graph.add_conditional_edges("guardrail", route_after_guardrail, {END: END, "error_fallback": "error_fallback"})
    graph.add_edge("error_fallback", END)

    return graph.compile()


_compiled_graph = None


def get_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph


def run_query(question: str) -> dict[str, Any]:
    run_id = new_run_id()
    handler = ObservabilityCallbackHandler(run_id)
    log_event("query_start", run_id, query=question)
    result = get_graph().invoke({"query": question}, config={"callbacks": [handler], "run_name": "guidance_query"})
    log_event(
        "query_end",
        run_id,
        answer=result.get("answer"),
        confidence=result.get("final_confidence"),
        citations=result.get("citations"),
    )
    result["run_id"] = run_id
    return result
