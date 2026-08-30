"""What the endpoint and the model id mean.

Everything that answers "which API is this, and what does that model id still
get us" lives here: the official DeepSeek API has a catalogue we can reason
about, a custom OpenAI-compatible gateway does not. Vision support, structured
output format and the migration off retired ids all key off that distinction.

Pure functions over strings, so this module can be unit-tested on its own.
"""

from __future__ import annotations

from .const import (
    DEEPSEEK_API_BASE_URL,
    LEGACY_CHAT_MODELS,
    RECOMMENDED_CHAT_MODEL,
)


def normalize_model_id(model: str | None) -> str:
    """Return a model id in the form the catalogue sets in const.py are keyed by."""
    return (model or "").strip().lower()


def is_official_deepseek_api_base_url(base_url: str | None) -> bool:
    """True for DeepSeek's hosted API.

    Only that endpoint has a model catalogue we can reason about: it serves the
    ids in CHAT_MODEL_OPTIONS and nothing else. A custom OpenAI-compatible
    gateway may map any id to any backend, so vision support, structured output
    format and legacy-id migration all key off this.
    """
    raw = (base_url or DEEPSEEK_API_BASE_URL).strip().lower()
    while raw.endswith("/"):
        raw = raw[:-1]
    if raw.endswith("/v1"):
        raw = raw[:-3]
    while raw.endswith("/"):
        raw = raw[:-1]
    return raw in ("https://api.deepseek.com", "http://api.deepseek.com")


def migrate_legacy_chat_model(model: str | None, *, base_url: str | None) -> str | None:
    """Return the replacement for a retired model id, or ``None`` to keep it.

    ``deepseek-chat`` and ``deepseek-reasoner`` stopped being served by the
    official API on LEGACY_CHAT_MODEL_RETIRED_ON, so an entry left on one of
    them fails every request. A custom gateway may still route those ids
    somewhere, so entries pointing at one are left untouched.
    """
    if not is_official_deepseek_api_base_url(base_url):
        return None
    if normalize_model_id(model) not in LEGACY_CHAT_MODELS:
        return None
    return RECOMMENDED_CHAT_MODEL


def is_retired_chat_model(model: str | None, *, base_url: str | None) -> bool:
    """Whether this endpoint has stopped serving ``model``."""
    return migrate_legacy_chat_model(model, base_url=base_url) is not None


def model_uses_deepseek_thinking_api(model: str) -> bool:
    """Whether to send DeepSeek ``extra_body.thinking`` for this model id.

    An empty id means the caller never overrode the default, which is a
    DeepSeek one - so the field applies.
    """
    m = normalize_model_id(model)
    if not m:
        return True
    return m.startswith("deepseek")
