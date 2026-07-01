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
CONF_PROMPT = "prompt"
CONF_TEMPERATURE = "temperature"
CONF_TOP_P = "top_p"
CONF_THINKING_ENABLED = "thinking_enabled"
CONF_REASONING_EFFORT = "reasoning_effort"
CONF_STRIP_MARKDOWN = "strip_markdown"
CONF_API_KEY = "api_key"
CONF_BASE_URL = "base_url"
CONF_FILENAMES = "filenames"

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
RECOMMENDED_TEMPERATURE = 1.0
RECOMMENDED_TOP_P = 1.0
DEFAULT_THINKING_ENABLED = False
DEFAULT_STRIP_MARKDOWN = False

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


def deepseek_chat_extra_body() -> dict[str, Any]:
    """Build OpenAI-SDK extra_body for DeepSeek thinking mode (enabled only)."""
    return {"thinking": {"type": "enabled"}}


def coerce_max_tokens(value: Any, *, fallback: int = RECOMMENDED_MAX_TOKENS) -> int:
    """Parse max_tokens from config options; clamp to [1, MAX_TOKENS_UPPER_BOUND]."""
    try:
        n = int(float(value))
    except (TypeError, ValueError):
        return fallback
    return max(1, min(n, MAX_TOKENS_UPPER_BOUND))


def normalized_reasoning_effort(value: Any) -> str:
    """Return a valid reasoning_effort string for the DeepSeek API."""
    if isinstance(value, str) and value in REASONING_EFFORT_VALUES:
        return value
    return RECOMMENDED_REASONING_EFFORT


def deepseek_chat_thinking_params(
    *, thinking_enabled: bool, reasoning_effort: str = RECOMMENDED_REASONING_EFFORT
) -> dict[str, Any]:
    """kwargs for chat.completions.create matching DeepSeek thinking docs.

    When thinking is off, returns nothing so generic OpenAI-compatible endpoints
    are not sent DeepSeek-only extra_body fields (conversation + generate_content).
    """
    if not thinking_enabled:
        return {}
    return {
        "extra_body": deepseek_chat_extra_body(),
        "reasoning_effort": normalized_reasoning_effort(reasoning_effort),
    }


def build_chat_completion_args(
    *,
    model: str,
    messages: list[dict[str, Any]],
    options: Mapping[str, Any],
    stream: bool,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str | dict[str, Any] | None = None,
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
        ),
    }
    if not thinking_on:
        args["top_p"] = options.get(CONF_TOP_P, RECOMMENDED_TOP_P)
        args["temperature"] = options.get(CONF_TEMPERATURE, RECOMMENDED_TEMPERATURE)
    if tools:
        args["tools"] = tools
    if tool_choice:
        args["tool_choice"] = tool_choice
    return args
