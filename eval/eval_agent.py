"""Independent evaluation subagent: an LLM-as-judge that scores a generated
answer against the test-set's expected answer/section using a separate
Claude call from the one that produced the answer, so the model never
grades its own work.
"""
import sys
from pathlib import Path

from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import get_llm  # noqa: E402


class JudgeOutput(BaseModel):
    hit: bool = Field(description="True if the answer used content from the expected page/section.")
    groundedness: bool = Field(description="True if the answer's claims are supported by the cited excerpts.")
    relevance_score: int = Field(description="0-100: how well the answer addresses the question.")
    reason: str = Field(description="Short justification.")


JUDGE_SYSTEM = (
    "You are an independent evaluator for a document-grounded Q&A system. You did NOT "
    "generate the answer being judged. Score it strictly against the question, the expected "
    "reference, and the excerpts the system actually cited."
)


def judge_answer(question: str, expected: dict, generated: dict, max_retries: int = 2) -> JudgeOutput:
    llm = get_llm().with_structured_output(JudgeOutput)
    cited_pages = sorted({c["page"] for c in generated.get("retrieved", []) if c["id"] in generated.get("citations", [])})
    prompt = (
        f"Question: {question}\n\n"
        f"Expected page: {expected.get('expected_page')}\n"
        f"Expected answer should mention: {expected.get('expected_answer_contains')}\n"
        f"Is this question expected to be answerable from the document: {expected.get('answerable')}\n\n"
        f"System's generated answer: {generated.get('answer')}\n"
        f"System's final confidence: {generated.get('final_confidence')}\n"
        f"Pages the system cited: {cited_pages}\n"
    )
    messages = [("system", JUDGE_SYSTEM), ("human", prompt)]
    last_exc = None
    for _attempt in range(max_retries + 1):
        try:
            return llm.invoke(messages)
        except Exception as exc:  # noqa: BLE001 - proxy occasionally returns malformed structured output
            last_exc = exc
    raise last_exc
