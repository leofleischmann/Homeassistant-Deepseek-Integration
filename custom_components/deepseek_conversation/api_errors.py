"""User-facing messages for DeepSeek / OpenAI SDK errors."""

from __future__ import annotations

import openai

from .const import VISION_CHAT_MODEL

_CONTEXT_HINT = (
    "The request was too large for the model: input tokens (system prompt, chat history, "
    "and especially large tool results such as GetLiveContext) exceed the model limit. "
    "max_tokens only limits the reply length. Reduce entities exposed to Assist, narrow the "
    "voice assistant area, or ask more specific questions so tools return less data."
)


def openai_exception_user_message(err: BaseException) -> str:
    """Return a short explanation for Assist / service callers."""
    text = str(err).lower()
    if "image_url" in text and "unknown variant" in text:
        return (
            "The API endpoint rejected the image (image_url content parts). On the "
            f"official DeepSeek API only {VISION_CHAT_MODEL} accepts images - select "
            "it under Configure -> Model. On a custom base URL, choose a model the "
            "gateway serves with vision support, or send the request without images."
        )
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
