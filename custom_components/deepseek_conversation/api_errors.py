"""User-facing messages for DeepSeek / OpenAI SDK errors."""

from __future__ import annotations

import openai

_CONTEXT_HINT = (
    "The request was too large for the model: input tokens (system prompt, chat history, "
    "and especially large tool results such as GetLiveContext) exceed the model limit. "
    "max_tokens only limits the reply length. Reduce entities exposed to Assist, narrow the "
    "voice assistant area, or ask more specific questions so tools return less data."
)


def openai_exception_user_message(err: BaseException) -> str:
    """Return a short explanation for Assist / service callers."""
    text = str(err).lower()
    if isinstance(err, openai.BadRequestError):
        if any(
            w in text
            for w in (
                "token",
                "context",
                "length",
                "maximum",
                "too large",
                "exceed",
                "payload",
            )
        ):
            return _CONTEXT_HINT
        return f"Request rejected by the API: {err}"
    if isinstance(err, openai.APIStatusError):
        if err.status_code == 400 and any(
            w in text for w in ("token", "context", "length", "maximum", "too large")
        ):
            return _CONTEXT_HINT
        return f"DeepSeek API error ({err.status_code}): {err}"
    if isinstance(err, openai.OpenAIError):
        return f"DeepSeek API error: {err}"
    return f"Error: {err}"
