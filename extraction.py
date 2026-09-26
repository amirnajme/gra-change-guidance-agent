"""LLM call that drafts a value for one regulatory variable from retrieved
excerpts. Kept separate from the groundedness guardrail (generation.py) so
the extraction and its verification are two independent LLM calls, mirroring
the anti-hallucination stance and cross-chunk attribution rules that were
hardened for Q&A generation in this app's earlier version.
"""
from typing import Optional

from pydantic import BaseModel, Field

from llm import get_llm, invoke_structured_with_retry
from variable_schema import VariableDef


class VariableExtractionOutput(BaseModel):
    value: str = Field(description="The extracted value/statement as it appears in the document, or 'not found in document'.")
    normalized_value: Optional[str] = Field(
        default=None, description="If the value is numeric, the value + unit normalized (e.g. '5 days'). Otherwise null."
    )
    rationale: str = Field(description="Quote or close paraphrase of the excerpt supporting this value.")
    cited_chunk_ids: list[str] = Field(description="Ids of the excerpts actually used, e.g. ['p78_185'].")
    self_confidence: int = Field(description="0-100: how directly the excerpts support this value.")
    found_in_document: bool = Field(description="False if the document does not state this variable.")


def extract_variable(variable: VariableDef, context: str, run_id: str = "") -> VariableExtractionOutput:
    system = (
        "You are a regulatory analyst extracting a single specific data point from excerpts of a "
        "pharma regulatory guidance document, for use in a regulatory change submission. Use ONLY "
        "the provided excerpts — never outside knowledge. If the excerpts do not state this "
        "variable, set found_in_document=false and value='not found in document'. Always list the "
        "ids of the excerpts you actually relied on. Do NOT attribute a value to a section/clause "
        "number unless that exact number appears together with the value in the SAME excerpt — "
        "never borrow a heading or section number from a different excerpt just because it was "
        "also retrieved. If the value is numeric, also fill normalized_value with the value and "
        "its unit exactly as stated.\n\n"
        f"Variable to extract: {variable.label}\n"
        f"Extraction instructions: {variable.extraction_instructions}"
    )
    llm = get_llm().with_structured_output(VariableExtractionOutput)
    return invoke_structured_with_retry(
        llm,
        [
            ("system", system),
            ("human", f"Document excerpts:\n\n{context}"),
        ],
        run_id=run_id,
        node="extract_variable",
    )
