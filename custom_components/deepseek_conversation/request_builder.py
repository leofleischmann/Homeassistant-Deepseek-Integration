"""Assembling the kwargs for a chat completion request.

One place decides what actually goes on the wire, so the Assist loop
(``chat_session.py``), the ``generate_content`` action and the
debug suite cannot drift apart on thinking flags, sampling parameters or the
token limit.

Pure functions over mappings, so this module can be unit-tested on its own.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .const import (
    CONF_CHAT_MODEL,
    CONF_MAX_TOKENS,
    CONF_REASONING_EFFORT,
    CONF_RESPONSE_FORMAT,
    CONF_TEMPERATURE,
    CONF_THINKING_ENABLED,
    CONF_TOP_P,
    DEFAULT_THINKING_ENABLED,
    RECOMMENDED_CHAT_MODEL,
    RECOMMENDED_MAX_TOKENS,
    RECOMMENDED_REASONING_EFFORT,
    RECOMMENDED_TEMPERATURE,
    RECOMMENDED_TOP_P,
    RESPONSE_FORMAT_JSON_OBJECT,
)
from .models import model_uses_deepseek_thinking_api
from .options import coerce_max_tokens, normalized_reasoning_effort


def deepseek_chat_extra_body(*, thinking_enabled: bool) -> dict[str, Any]:
    """OpenAI-SDK extra_body for the DeepSeek thinking toggle.

    V4 models default to thinking **enabled** when this field is omitted; send
    ``disabled`` explicitly when the integration option is off. See
    build_chat_completion_args().
    """
    return {"thinking": {"type": "enabled" if thinking_enabled else "disabled"}}


def deepseek_chat_thinking_params(
    *,
    thinking_enabled: bool,
    reasoning_effort: str = RECOMMENDED_REASONING_EFFORT,
    model: str = RECOMMENDED_CHAT_MODEL,
) -> dict[str, Any]:
    """kwargs for chat.completions.create matching DeepSeek thinking docs.

    DeepSeek model ids get an explicit thinking on/off via extra_body (V4 default is on).
    Other model ids on a custom base_url get no extra_body so OpenAI-compatible proxies
    are not sent DeepSeek-only fields.
    """
    if not model_uses_deepseek_thinking_api(model):
        return {}
    params: dict[str, Any] = {
        "extra_body": deepseek_chat_extra_body(thinking_enabled=thinking_enabled),
    }
    if thinking_enabled:
        params["reasoning_effort"] = normalized_reasoning_effort(reasoning_effort)
    return params


def build_chat_completion_args(
    *,
    model: str,
    messages: list[dict[str, Any]],
    options: Mapping[str, Any],
    stream: bool,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str | dict[str, Any] | None = None,
    response_format: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build kwargs for ``client.chat.completions.create``.

    Shared by the Assist loop and the ``generate_content`` action.
    Temperature and top_p are omitted when reasoning/thinking is enabled.
    """
    thinking_on = bool(options.get(CONF_THINKING_ENABLED, DEFAULT_THINKING_ENABLED))
    args: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": coerce_max_tokens(
            options.get(CONF_MAX_TOKENS, RECOMMENDED_MAX_TOKENS)
        ),
        "stream": stream,
        **deepseek_chat_thinking_params(
            thinking_enabled=thinking_on,
            reasoning_effort=options.get(
                CONF_REASONING_EFFORT, RECOMMENDED_REASONING_EFFORT
            ),
            model=model,
        ),
    }
    if not thinking_on:
        args["top_p"] = options.get(CONF_TOP_P, RECOMMENDED_TOP_P)
        args["temperature"] = options.get(CONF_TEMPERATURE, RECOMMENDED_TEMPERATURE)
    if tools:
        args["tools"] = tools
    if tool_choice:
        args["tool_choice"] = tool_choice
    if response_format is not None:
        args["response_format"] = response_format
    if stream:
        args["stream_options"] = {"include_usage": True}
    return args


def resolve_generate_content_model(
    agent_options: Mapping[str, Any], service_data: Mapping[str, Any]
) -> str:
    """Return the model a ``generate_content`` call will use.

    Split out so the caller can check image support and migrate a retired id
    before any request is built.
    """
    model = str(agent_options.get(CONF_CHAT_MODEL, RECOMMENDED_CHAT_MODEL))
    if override_model := service_data.get(CONF_CHAT_MODEL):
        model = str(override_model).strip() or model
    return model


def build_generate_content_completion_args(
    *,
    agent_options: Mapping[str, Any],
    messages: list[dict[str, Any]],
    service_data: Mapping[str, Any],
    model: str | None = None,
) -> tuple[str, dict[str, Any]]:
    """Build completion kwargs for ``generate_content`` with optional per-call overrides.

    Overrides: chat_model, temperature, thinking_enabled, max_tokens, response_format.
    Unset fields fall back to the calling agent's settings. ``model`` overrides
    the resolved id, so the caller can pass one it already migrated.
    """
    effective_options = dict(agent_options)
    model = model or resolve_generate_content_model(agent_options, service_data)

    if CONF_TEMPERATURE in service_data:
        effective_options[CONF_TEMPERATURE] = service_data[CONF_TEMPERATURE]
    if CONF_THINKING_ENABLED in service_data:
        effective_options[CONF_THINKING_ENABLED] = service_data[CONF_THINKING_ENABLED]
    if CONF_MAX_TOKENS in service_data:
        effective_options[CONF_MAX_TOKENS] = service_data[CONF_MAX_TOKENS]

    response_format: dict[str, str] | None = None
    if service_data.get(CONF_RESPONSE_FORMAT) == RESPONSE_FORMAT_JSON_OBJECT:
        response_format = {"type": RESPONSE_FORMAT_JSON_OBJECT}

    args = build_chat_completion_args(
        model=model,
        messages=messages,
        options=effective_options,
        stream=False,
        response_format=response_format,
    )
    return model, args


def effective_thinking_enabled_for_generate_content(
    agent_options: Mapping[str, Any],
    service_data: Mapping[str, Any],
) -> bool:
    """Resolve whether reasoning is active for a ``generate_content`` call.

    The options are the calling agent's; the config entry itself carries
    only the credentials.
    """
    if CONF_THINKING_ENABLED in service_data:
        return bool(service_data[CONF_THINKING_ENABLED])
    return bool(agent_options.get(CONF_THINKING_ENABLED, DEFAULT_THINKING_ENABLED))


def reasoning_text_from_chat_message(message: Any) -> str:
    """Return DeepSeek reasoning text from a chat completion message object."""
    for attr in ("reasoning_content", "reasoning"):
        value = getattr(message, attr, None)
        if isinstance(value, str):
            return value
    return ""
