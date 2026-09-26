"""Claude client construction and a shared retry wrapper for structured
output calls. The Claude-compatible proxy this app talks to occasionally
returns a malformed/incomplete tool call, which fails pydantic validation;
both the live pipeline (graph.py) and the independent eval judge
(eval/eval_agent.py) need to retry through that, so the wrapper lives here
once instead of being copy-pasted in both places.
"""
import os
from functools import lru_cache

from langchain_anthropic import ChatAnthropic

from config import CLAUDE_MODEL
from observability import log_event


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


@lru_cache(maxsize=1)
def get_llm(max_tokens: int = 1024, temperature: float = 0.0) -> ChatAnthropic:
    return ChatAnthropic(
        model=CLAUDE_MODEL,
        max_tokens=max_tokens,
        temperature=temperature,
        **get_anthropic_credentials(),
    )


def invoke_structured_with_retry(structured_llm, messages, run_id: str = "", node: str = "", max_retries: int = 3):
    """Retry a structured-output call a few times before giving up, so a
    transient malformed tool-call response doesn't turn into a false
    "not found in document" refusal (or a false negative from the eval
    judge)."""
    last_exc = None
    for attempt in range(max_retries + 1):
        try:
            return structured_llm.invoke(messages)
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            log_event("structured_output_retry", run_id, node=node, attempt=attempt, error=str(exc))
    raise last_exc
