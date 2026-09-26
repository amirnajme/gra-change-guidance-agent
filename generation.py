"""The independent groundedness guardrail call, kept separate from the
extraction call (extraction.py) so the model never grades its own work.
This prompt was hardened through real debugging (unit-equivalence
handling) — do not reword it.
"""
from pydantic import BaseModel, Field

from llm import get_llm, invoke_structured_with_retry


class GuardrailOutput(BaseModel):
    grounded: bool = Field(description="True only if every claim in the answer is directly supported by the excerpts.")
    reason: str = Field(description="Short justification.")


def check_groundedness(answer: str, excerpts: str, run_id: str = "") -> GuardrailOutput:
    llm = get_llm().with_structured_output(GuardrailOutput)
    return invoke_structured_with_retry(
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
            ("human", f"Excerpts:\n\n{excerpts}\n\nAnswer to check:\n{answer}"),
        ],
        run_id=run_id,
        node="guardrail",
    )
