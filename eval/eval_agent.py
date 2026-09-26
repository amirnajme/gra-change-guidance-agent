"""Independent evaluation subagent: an LLM-as-judge that scores an
extracted variable's value against the test-set's expected value/page
using a separate Claude call from the one that produced the extraction, so
the model never grades its own work.
"""
import sys
from pathlib import Path

from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from llm import get_llm, invoke_structured_with_retry  # noqa: E402
from variable_schema import VariableDef  # noqa: E402


class JudgeOutput(BaseModel):
    hit: bool = Field(description="True if the system retrieved/cited content from the expected page.")
    groundedness: bool = Field(description="True if the extracted value's claims are supported by the cited excerpts.")
    value_correctness: int = Field(description="0-100: how correct the extracted value is vs. the expected value.")
    reason: str = Field(description="Short justification.")


JUDGE_SYSTEM = (
    "You are an independent evaluator for a document-grounded regulatory variable extraction "
    "system. You did NOT perform the extraction being judged. Score it strictly against the "
    "variable definition, the expected reference, and what the system actually extracted."
)


def judge_extraction(variable: VariableDef, expected: dict, generated: dict, max_retries: int = 2) -> JudgeOutput:
    llm = get_llm().with_structured_output(JudgeOutput)
    cited_pages = sorted({c["page"] for c in generated.get("retrieved", []) if c["id"] in generated.get("citations", [])})
    prompt = (
        f"Variable: {variable.label}\n\n"
        f"Expected page: {expected.get('expected_page')}\n"
        f"Expected value should contain: {expected.get('expected_value_contains')}\n"
        f"Is this variable expected to be found in the document: {expected.get('expected_found')}\n\n"
        f"System's extracted value: {generated.get('value')}\n"
        f"System's found_in_document: {generated.get('found_in_document')}\n"
        f"System's final confidence: {generated.get('final_confidence')}\n"
        f"Pages the system cited: {cited_pages}\n"
    )
    messages = [("system", JUDGE_SYSTEM), ("human", prompt)]
    return invoke_structured_with_retry(llm, messages, node="eval_judge", max_retries=max_retries)
