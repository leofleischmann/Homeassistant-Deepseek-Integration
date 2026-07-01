"""Constants for the DeepSeek Conversation integration."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

DOMAIN = "deepseek_conversation"
LOGGER: logging.Logger = logging.getLogger(__package__)

# Configuration keys
CONF_CHAT_MODEL = "chat_model"
CONF_MAX_TOKENS = "max_tokens"
CONF_MAX_TOOL_ITERATIONS = "max_tool_iterations"
CONF_PROMPT = "prompt"
CONF_TEMPERATURE = "temperature"
CONF_TOP_P = "top_p"
CONF_THINKING_ENABLED = "thinking_enabled"
CONF_REASONING_EFFORT = "reasoning_effort"
CONF_STRIP_MARKDOWN = "strip_markdown"
CONF_VISION_ENABLED = "vision_enabled"
CONF_CONTEXT_MANAGEMENT_ENABLED = "context_management_enabled"
CONF_MAX_TOOL_RESULT_CHARS = "max_tool_result_chars"
CONF_BASE_URL = "base_url"
CONF_FILENAMES = "filenames"
CONF_RESPONSE_FORMAT = "response_format"

RESPONSE_FORMAT_JSON_OBJECT = "json_object"

# Default system prompt (Jinja: ha_name, user_name, llm_context)
DEFAULT_SYSTEM_PROMPT = """You are an assistant for Home Assistant, the open-source home automation platform.
Answer truthfully. Reply in plain text unless the user asks for another format (e.g. markdown or a list).
When tools are available to read or change the home, use them when the user's request needs current state or actions.
Keep answers concise for short questions; add detail only when asked or when it clearly helps."""

# DeepSeek V4 (legacy IDs deepseek-chat / deepseek-reasoner retire 2026-07-24)
RECOMMENDED_CHAT_MODEL = "deepseek-v4-flash"

CHAT_MODEL_OPTIONS: tuple[tuple[str, str], ...] = (
    ("deepseek-v4-flash", "DeepSeek V4 Flash (fast, default)"),
    ("deepseek-v4-pro", "DeepSeek V4 Pro (most capable)"),
    ("deepseek-chat", "Legacy: deepseek-chat (until 2026-07-24)"),
    ("deepseek-reasoner", "Legacy: deepseek-reasoner (until 2026-07-24)"),
)

RECOMMENDED_MAX_TOKENS = 1500
RECOMMENDED_MAX_TOOL_ITERATIONS = 10
MAX_TOOL_ITERATIONS_UPPER_BOUND = 20
RECOMMENDED_MAX_TOOL_RESULT_CHARS = 12_000
RECOMMENDED_TEMPERATURE = 1.0
RECOMMENDED_TOP_P = 1.0
DEFAULT_THINKING_ENABLED = False
DEFAULT_STRIP_MARKDOWN = False
DEFAULT_VISION_ENABLED = True
DEFAULT_CONTEXT_MANAGEMENT_ENABLED = True

REASONING_EFFORT_SELECT: tuple[tuple[str, str], ...] = (
    ("low", "Low"),
    ("medium", "Medium"),
    ("high", "High"),
    ("max", "Max"),
    ("xhigh", "xHigh"),
)
REASONING_EFFORT_VALUES: frozenset[str] = frozenset(v for v, _ in REASONING_EFFORT_SELECT)
RECOMMENDED_REASONING_EFFORT = "high"

MAX_TOKENS_UPPER_BOUND = 1_000_000
DEEPSEEK_API_BASE_URL = "https://api.deepseek.com/v1"


def deepseek_chat_extra_body(*, thinking_enabled: bool) -> dict[str, Any]:
    """OpenAI-SDK extra_body for DeepSeek thinking toggle.

    V4 models default to thinking **enabled** when this field is omitted; send
    ``disabled`` explicitly when the integration option is off. See conversation.py
    and build_chat_completion_args().
    """
    return {"thinking": {"type": "enabled" if thinking_enabled else "disabled"}}


def model_uses_deepseek_thinking_api(model: str) -> bool:
    """Whether to send DeepSeek ``extra_body.thinking`` for this model id."""
    m = (model or "").strip().lower()
    if not m:
        return True
    return m.startswith("deepseek")


def coerce_max_tokens(value: Any, *, fallback: int = RECOMMENDED_MAX_TOKENS) -> int:
    """Parse max_tokens from config options; clamp to [1, MAX_TOKENS_UPPER_BOUND]."""
    try:
        n = int(float(value))
    except (TypeError, ValueError):
        return fallback
    return max(1, min(n, MAX_TOKENS_UPPER_BOUND))


def coerce_max_tool_iterations(
    value: Any, *, fallback: int = RECOMMENDED_MAX_TOOL_ITERATIONS
) -> int:
    """Parse max_tool_iterations from config options; clamp to [1, MAX_TOOL_ITERATIONS_UPPER_BOUND]."""
    try:
        n = int(float(value))
    except (TypeError, ValueError):
        return fallback
    return max(1, min(n, MAX_TOOL_ITERATIONS_UPPER_BOUND))


def normalized_reasoning_effort(value: Any) -> str:
    """Return a valid reasoning_effort string for the DeepSeek API."""
    if isinstance(value, str) and value in REASONING_EFFORT_VALUES:
        return value
    return RECOMMENDED_REASONING_EFFORT


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
    response_format: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Build kwargs for ``client.chat.completions.create``.

    Shared by conversation.py (Assist) and __init__.py (generate_content).
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


def build_generate_content_completion_args(
    *,
    entry_options: Mapping[str, Any],
    messages: list[dict[str, Any]],
    service_data: Mapping[str, Any],
) -> tuple[str, dict[str, Any]]:
    """Build completion kwargs for ``generate_content`` with optional per-call overrides.

    Overrides: chat_model, temperature, thinking_enabled, max_tokens, response_format.
    Unset fields fall back to the config entry options. Used only from __init__.py.
    """
    effective_options = dict(entry_options)
    model = str(entry_options.get(CONF_CHAT_MODEL, RECOMMENDED_CHAT_MODEL))

    if override_model := service_data.get(CONF_CHAT_MODEL):
        model = str(override_model).strip() or model
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
