"""Constants for the DeepSeek Conversation integration."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

from homeassistant.const import CONF_LLM_HASS_API  # pyright: ignore[reportMissingImports]
from homeassistant.helpers import llm  # pyright: ignore[reportMissingImports]

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
#: Removed in 1.8.0; only fold_context_switch() still knows the name, to
#: turn an entry that had it switched off into the two limits it forced.
CONF_CONTEXT_MANAGEMENT_ENABLED = "context_management_enabled"
CONF_MAX_TOOL_RESULT_CHARS = "max_tool_result_chars"
CONF_MAX_HISTORY_ROUNDS = "max_history_rounds"
CONF_INCLUDE_USER_CONTEXT = "include_user_context"
CONF_REQUEST_TIMEOUT = "request_timeout"
CONF_BASE_URL = "base_url"
CONF_BRAVE_API_KEY = "brave_api_key"
CONF_FILENAMES = "filenames"
CONF_RESPONSE_FORMAT = "response_format"
#: Set when an agent is left on the recommended settings, so the flow knows to
#: skip the advanced step and the stored data stays small.
CONF_RECOMMENDED = "recommended"

# One config entry holds the credentials; every agent is a subentry of it.
SUBENTRY_TYPE_CONVERSATION = "conversation"
SUBENTRY_TYPE_AI_TASK = "ai_task_data"
SUBENTRY_TYPES: tuple[str, ...] = (SUBENTRY_TYPE_CONVERSATION, SUBENTRY_TYPE_AI_TASK)

DEFAULT_CONVERSATION_NAME = "DeepSeek Conversation"
DEFAULT_AI_TASK_NAME = "DeepSeek AI Task"

RESPONSE_FORMAT_JSON_OBJECT = "json_object"

# Default system prompt. Available Jinja variables: ha_name and llm_context from
# Home Assistant, plus everything in user_context.USER_CONTEXT_VARS (user_id,
# user_name, user_area, ...) which this integration defines. Unknown speaker
# values render as empty strings, so `{% if user_name %}` is the way to branch.
DEFAULT_SYSTEM_PROMPT = """You are an assistant for Home Assistant, the open-source home automation platform.
Answer truthfully. Reply in plain text unless the user asks for another format (e.g. markdown or a list).
When tools are available to read or change the home, use them when the user's request needs current state or actions.
Keep answers concise for short questions; add detail only when asked or when it clearly helps."""

RECOMMENDED_CHAT_MODEL = "deepseek-v4-flash"

#: The only official model that accepts image input; see vision.py.
VISION_CHAT_MODEL = "deepseek-v4-flash-vision-exp"

CHAT_MODEL_OPTIONS: tuple[tuple[str, str], ...] = (
    ("deepseek-v4-flash", "DeepSeek V4 Flash (fast, default)"),
    ("deepseek-v4-pro", "DeepSeek V4 Pro (most capable)"),
    (VISION_CHAT_MODEL, "DeepSeek V4 Flash Vision (experimental, image input)"),
)

#: Model ids that accept OpenAI-style ``image_url`` content parts.
VISION_CHAT_MODELS: frozenset[str] = frozenset({VISION_CHAT_MODEL})

#: Retired: the official API stopped serving these on LEGACY_CHAT_MODEL_RETIRED_ON.
#: Entries still configured with one are migrated by migrate_legacy_chat_model().
LEGACY_CHAT_MODELS: frozenset[str] = frozenset({"deepseek-chat", "deepseek-reasoner"})
LEGACY_CHAT_MODEL_RETIRED_ON = "2026-07-24"

RECOMMENDED_MAX_TOKENS = 1500
RECOMMENDED_MAX_TOOL_ITERATIONS = 10
MAX_TOOL_ITERATIONS_UPPER_BOUND = 20
RECOMMENDED_MAX_TOOL_RESULT_CHARS = 12_000
RECOMMENDED_MAX_HISTORY_ROUNDS = 0
RECOMMENDED_TEMPERATURE = 1.0
RECOMMENDED_TOP_P = 1.0
DEFAULT_THINKING_ENABLED = False
#: On by default: a reply is read out loud far more often than it is read,
#: and "asterisk asterisk" is never what anyone wanted to hear.
DEFAULT_STRIP_MARKDOWN = True
# Opt-in: sending a household member's name to the API is the user's call, so
# an update must not start doing it on its own.
DEFAULT_INCLUDE_USER_CONTEXT = False
DEFAULT_VISION_ENABLED = True

REASONING_EFFORT_SELECT: tuple[tuple[str, str], ...] = (
    ("low", "Low"),
    ("medium", "Medium"),
    ("high", "High"),
    ("max", "Max"),
    ("xhigh", "xHigh"),
)
REASONING_EFFORT_VALUES: frozenset[str] = frozenset(v for v, _ in REASONING_EFFORT_SELECT)
RECOMMENDED_REASONING_EFFORT = "high"

#: Ceiling for the reply length option. V4 models take a 1M token context but
#: generate at most 384K, so anything above this could only ever be rejected.
MAX_TOKENS_UPPER_BOUND = 384_000
DEEPSEEK_API_BASE_URL = "https://api.deepseek.com/v1"

# Request limits. The OpenAI SDK defaults to a 600 s timeout and two retries, so
# an unresponsive endpoint can block a voice pipeline for ten minutes; a voice
# assistant is better served by failing early.
RECOMMENDED_REQUEST_TIMEOUT = 60
REQUEST_TIMEOUT_LOWER_BOUND = 5
REQUEST_TIMEOUT_UPPER_BOUND = 600
#: Floor for non-streamed calls (generate_content). httpx applies the timeout
#: per read: for a streamed call it is the gap between two chunks, while a
#: blocking call must fit the whole generation into it - and a reasoning run
#: with a large max_tokens legitimately takes minutes.
MIN_BLOCKING_REQUEST_TIMEOUT = 300
#: One retry, not the SDK default of two: on voice, a late answer is a failure.
DEEPSEEK_MAX_RETRIES = 1

# Starting point for a newly added agent. Everything absent from a subentry's
# data falls back to the RECOMMENDED_* / DEFAULT_* values above at read time,
# so an agent left on the recommended settings stores only these few keys.
RECOMMENDED_CONVERSATION_OPTIONS: dict[str, Any] = {
    CONF_RECOMMENDED: True,
    CONF_LLM_HASS_API: [llm.LLM_API_ASSIST],
    CONF_PROMPT: DEFAULT_SYSTEM_PROMPT,
    CONF_CHAT_MODEL: RECOMMENDED_CHAT_MODEL,
}

#: An AI Task generates data for an automation, so it starts without control
#: over the home; add a Home Assistant API to it if the task needs one.
RECOMMENDED_AI_TASK_OPTIONS: dict[str, Any] = {
    CONF_RECOMMENDED: True,
    CONF_PROMPT: DEFAULT_SYSTEM_PROMPT,
    CONF_CHAT_MODEL: RECOMMENDED_CHAT_MODEL,
}


def normalize_model_id(model: str | None) -> str:
    """Return a model id in the form the catalogue sets above are keyed by."""
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


#: The settings the first step of the agent form asks for. Everything else is
#: an override of a recommended default.
BASIC_AGENT_OPTIONS: frozenset[str] = frozenset(
    {CONF_RECOMMENDED, CONF_PROMPT, CONF_LLM_HASS_API, CONF_CHAT_MODEL}
)


def recommended_agent_options(options: Mapping[str, Any]) -> dict[str, Any]:
    """Drop the overrides an agent no longer wants to keep.

    Switching an agent back to the recommended settings has to forget what was
    set behind them. Keeping the values would leave the agent running on a
    reply limit or a reasoning effort its own form no longer shows.
    """
    return {key: value for key, value in options.items() if key in BASIC_AGENT_OPTIONS}


#: Settings that only mean something when a person is on the other end.
#: Markdown stripping and naming the speaker are about being spoken to, and a
#: history cap needs a history - an AI Task chat log is a single turn.
ASSIST_ONLY_OPTIONS: frozenset[str] = frozenset(
    {CONF_STRIP_MARKDOWN, CONF_INCLUDE_USER_CONTEXT, CONF_MAX_HISTORY_ROUNDS}
)


def ai_task_options_from(options: Mapping[str, Any]) -> dict[str, Any]:
    """Return agent settings with the Assist-only ones removed."""
    return {
        key: value
        for key, value in options.items()
        if key not in ASSIST_ONLY_OPTIONS
    }


#: What an entry carried when its owner never touched the setting. 1.7.0 wrote
#: every default into the entry, so a stored ``False`` here says nothing about
#: what the user wanted - it is just the old default written down.
_PREVIOUS_STRIP_MARKDOWN_DEFAULT = False


def adopt_strip_markdown_default(options: dict[str, Any]) -> dict[str, Any]:
    """Let an untouched markdown setting follow the new default.

    Dropping the key rather than flipping it is the point: the agent then
    follows DEFAULT_STRIP_MARKDOWN, and an owner who had deliberately turned it
    on keeps that. Only the value that was merely the old default gives way.
    """
    if options.get(CONF_STRIP_MARKDOWN) == _PREVIOUS_STRIP_MARKDOWN_DEFAULT:
        options.pop(CONF_STRIP_MARKDOWN)
    return options


def fold_context_switch(options: dict[str, Any]) -> dict[str, Any]:
    """Replace the removed context switch with the limits it stood for.

    ``context_management_enabled`` did nothing except force both limits to
    zero, which is what zero already means in either field. Two ways to say one
    thing is what made the form hard to read, so the switch is gone and an
    entry that had it turned off keeps its behaviour as explicit zeros.
    """
    if CONF_CONTEXT_MANAGEMENT_ENABLED not in options:
        return options
    if not options.pop(CONF_CONTEXT_MANAGEMENT_ENABLED):
        options[CONF_MAX_TOOL_RESULT_CHARS] = 0
        options[CONF_MAX_HISTORY_ROUNDS] = 0
    return options


def is_retired_chat_model(model: str | None, *, base_url: str | None) -> bool:
    """Whether this endpoint has stopped serving ``model``."""
    return migrate_legacy_chat_model(model, base_url=base_url) is not None


def deepseek_chat_extra_body(*, thinking_enabled: bool) -> dict[str, Any]:
    """OpenAI-SDK extra_body for DeepSeek thinking toggle.

    V4 models default to thinking **enabled** when this field is omitted; send
    ``disabled`` explicitly when the integration option is off. See conversation.py
    and build_chat_completion_args().
    """
    return {"thinking": {"type": "enabled" if thinking_enabled else "disabled"}}


def model_uses_deepseek_thinking_api(model: str) -> bool:
    """Whether to send DeepSeek ``extra_body.thinking`` for this model id."""
    m = normalize_model_id(model)
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


def coerce_request_timeout(
    value: Any, *, fallback: int = RECOMMENDED_REQUEST_TIMEOUT
) -> float:
    """Parse request_timeout from config options; clamp to the allowed range."""
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return float(fallback)
    if seconds <= 0:
        return float(fallback)
    return max(
        float(REQUEST_TIMEOUT_LOWER_BOUND),
        min(seconds, float(REQUEST_TIMEOUT_UPPER_BOUND)),
    )


def request_timeout_from_options(options: Mapping[str, Any]) -> float:
    """Timeout for streamed calls: the longest accepted gap between two chunks."""
    return coerce_request_timeout(
        options.get(CONF_REQUEST_TIMEOUT, RECOMMENDED_REQUEST_TIMEOUT)
    )


def blocking_request_timeout_from_options(options: Mapping[str, Any]) -> float:
    """Timeout for non-streamed calls, which must cover the whole generation.

    See MIN_BLOCKING_REQUEST_TIMEOUT: the configured value is a stall detector
    for streaming and would cut off a legitimate long reasoning run here.
    """
    return max(
        request_timeout_from_options(options),
        float(MIN_BLOCKING_REQUEST_TIMEOUT),
    )


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
    response_format: dict[str, Any] | None = None,
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
    the resolved id, so the caller can pass one it already migrated. Used only
    from __init__.py.
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
