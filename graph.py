"""LangGraph pipeline: normalize -> retrieve -> check_context -> extract_variable
-> score_confidence -> guardrail -> END. One pass through this graph extracts
ONE critical regulatory variable (variable_schema.VariableDef) from ONE
uploaded document; `run_all_extractions` loops it over the full variable list
for a document, reporting progress via `on_variable` for the UI's running
commentary.
"""
import traceback
from typing import Any, Optional, TypedDict

from langgraph.graph import END, StateGraph

import db
from config import MIN_SIMILARITY_FOR_CONTEXT, similarity_to_confidence_cap
from extraction import extract_variable
from generation import check_groundedness
from observability import ObservabilityCallbackHandler, log_event, new_run_id
from retrieval import RetrievedChunk, retrieve_chunks
from units import extract_numeric_mentions
from variable_schema import CRITICAL_VARIABLES, VariableDef

NOT_FOUND_VALUE = "not found in document"


class GraphState(TypedDict, total=False):
    doc_id: str
    variable: VariableDef
    variable_key: str
    query: str
    run_id: str
    numeric_mentions: list[dict]
    retrieved: list[RetrievedChunk]
    context_sufficient: bool
    value: str
    normalized_value: Optional[str]
    rationale: str
    citations: list[str]
    self_confidence: int
    final_confidence: int
    found_in_document: bool
    grounded: bool
    guardrail_reason: str
    error: Optional[str]


def normalize_query_node(state: GraphState) -> dict:
    mentions = extract_numeric_mentions(state["query"])
    return {"numeric_mentions": [m.__dict__ for m in mentions]}


def retrieve_node(state: GraphState) -> dict:
    try:
        retrieved = retrieve_chunks(state["doc_id"], state["query"], state.get("numeric_mentions", []))
        return {"retrieved": retrieved}
    except Exception as exc:  # on-error hook: never crash the graph
        log_event(
            "node_error", state.get("run_id", ""), doc_id=state.get("doc_id", ""),
            node="retrieve_node", error=str(exc), traceback=traceback.format_exc(),
        )
        return {"error": f"retrieve_node: {exc}", "retrieved": []}


def check_context_node(state: GraphState) -> dict:
    if state.get("error"):
        return {"context_sufficient": False}
    retrieved = state.get("retrieved", [])
    best = max((c["similarity"] for c in retrieved), default=0.0)
    has_numeric_hit = any(c["numeric_match"] for c in retrieved)
    return {"context_sufficient": bool(retrieved) and (best >= MIN_SIMILARITY_FOR_CONTEXT or has_numeric_hit)}


def route_after_context(state: GraphState) -> str:
    return "extract_variable" if state.get("context_sufficient") else "no_context"


def no_context_node(state: GraphState) -> dict:
    return {
        "value": NOT_FOUND_VALUE,
        "normalized_value": None,
        "rationale": "",
        "citations": [],
        "self_confidence": 0,
        "final_confidence": 5,
        "found_in_document": False,
        "grounded": True,
        "guardrail_reason": "Variable not discussed in this document.",
    }


def extract_variable_node(state: GraphState) -> dict:
    try:
        context = "\n\n".join(
            f"[{c['id']} | page {c['page']} | {c['section']}]\n{c['text']}" for c in state["retrieved"]
        )
        out = extract_variable(state["variable"], context, run_id=state.get("run_id", ""))
        if not out.found_in_document:
            return {
                "value": NOT_FOUND_VALUE,
                "normalized_value": None,
                "rationale": out.rationale,
                "citations": [],
                "self_confidence": 0,
                "found_in_document": False,
            }
        return {
            "value": out.value,
            "normalized_value": out.normalized_value,
            "rationale": out.rationale,
            "citations": out.cited_chunk_ids,
            "self_confidence": max(0, min(100, out.self_confidence)),
            "found_in_document": True,
        }
    except Exception as exc:
        log_event(
            "node_error", state.get("run_id", ""), doc_id=state.get("doc_id", ""),
            node="extract_variable_node", error=str(exc), traceback=traceback.format_exc(),
        )
        return {"error": f"extract_variable_node: {exc}"}


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
    if state.get("error") or not state.get("found_in_document") or not state.get("citations"):
        return {"grounded": True}
    try:
        cited = {c["id"]: c for c in state.get("retrieved", [])}
        excerpts = "\n\n".join(cited[i]["text"] for i in state["citations"] if i in cited)
        answer_text = f"{state.get('value')} — {state.get('rationale')}"
        out = check_groundedness(answer_text, excerpts, run_id=state.get("run_id", ""))
        if not out.grounded:
            return {
                "grounded": False,
                "guardrail_reason": out.reason,
                "value": NOT_FOUND_VALUE,
                "found_in_document": False,
                "citations": [],
                "final_confidence": min(state.get("final_confidence", 0), 20),
            }
        return {"grounded": True, "guardrail_reason": out.reason}
    except Exception as exc:
        log_event(
            "node_error", state.get("run_id", ""), doc_id=state.get("doc_id", ""),
            node="guardrail_node", error=str(exc), traceback=traceback.format_exc(),
        )
        return {"error": f"guardrail_node: {exc}"}


def error_fallback_node(state: GraphState) -> dict:
    return {
        "value": NOT_FOUND_VALUE,
        "normalized_value": None,
        "rationale": "",
        "citations": [],
        "self_confidence": 0,
        "final_confidence": 0,
        "found_in_document": False,
        "grounded": True,
        "guardrail_reason": f"Pipeline error: {state.get('error')}",
    }


def route_after_extract(state: GraphState) -> str:
    return "error_fallback" if state.get("error") else "score_confidence"


def route_after_guardrail(state: GraphState) -> str:
    return "error_fallback" if state.get("error") else END


def build_graph():
    graph = StateGraph(GraphState)
    graph.add_node("normalize_query", normalize_query_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("check_context", check_context_node)
    graph.add_node("no_context", no_context_node)
    graph.add_node("extract_variable", extract_variable_node)
    graph.add_node("score_confidence", score_confidence_node)
    graph.add_node("guardrail", guardrail_node)
    graph.add_node("error_fallback", error_fallback_node)

    graph.set_entry_point("normalize_query")
    graph.add_edge("normalize_query", "retrieve")
    graph.add_edge("retrieve", "check_context")
    graph.add_conditional_edges(
        "check_context", route_after_context, {"extract_variable": "extract_variable", "no_context": "no_context"}
    )
    graph.add_edge("no_context", END)
    graph.add_conditional_edges(
        "extract_variable", route_after_extract, {"score_confidence": "score_confidence", "error_fallback": "error_fallback"}
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


def run_extraction(doc_id: str, variable: VariableDef) -> dict[str, Any]:
    run_id = new_run_id()
    handler = ObservabilityCallbackHandler(run_id, doc_id=doc_id)
    log_event("extraction_start", run_id, doc_id=doc_id, variable_key=variable.key)
    result = get_graph().invoke(
        {"doc_id": doc_id, "variable": variable, "variable_key": variable.key, "query": variable.search_query, "run_id": run_id},
        config={"callbacks": [handler], "run_name": "variable_extraction"},
    )
    log_event(
        "extraction_end", run_id, doc_id=doc_id, variable_key=variable.key,
        value=result.get("value"), confidence=result.get("final_confidence"), error=result.get("error"),
    )
    result["run_id"] = run_id

    citations = result.get("citations", [])
    cited_map = {c["id"]: c for c in result.get("retrieved", [])}
    first_cited = cited_map.get(citations[0]) if citations else None

    db.save_variable_result(
        doc_id,
        variable.key,
        variable.label,
        value=result.get("value"),
        normalized_value=result.get("normalized_value"),
        rationale=result.get("rationale"),
        cited_chunk_ids=citations,
        page=first_cited["page"] if first_cited else None,
        section=first_cited["section"] if first_cited else None,
        self_confidence=result.get("self_confidence"),
        final_confidence=result.get("final_confidence"),
        found_in_document=result.get("found_in_document"),
        grounded=result.get("grounded"),
        guardrail_reason=result.get("guardrail_reason"),
        run_id=run_id,
    )
    return result


def run_all_extractions(doc_id: str, on_variable=None) -> list[dict[str, Any]]:
    """Runs the full CRITICAL_VARIABLES list through run_extraction for
    `doc_id`. `on_variable(index, total, variable, result)`, if given, is
    called after each variable completes (used by the UI's running
    commentary during processing)."""
    results = []
    total = len(CRITICAL_VARIABLES)
    for index, variable in enumerate(CRITICAL_VARIABLES, start=1):
        result = run_extraction(doc_id, variable)
        results.append(result)
        if on_variable:
            on_variable(index, total, variable, result)
    return results
