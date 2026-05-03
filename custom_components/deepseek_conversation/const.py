"""Constants for the DeepSeek Conversation integration."""

from __future__ import annotations

import logging
from typing import Any

# Changed domain to reflect DeepSeek
DOMAIN = "deepseek_conversation"
LOGGER: logging.Logger = logging.getLogger(__package__)

# Configuration keys
CONF_CHAT_MODEL = "chat_model"
CONF_MAX_TOKENS = "max_tokens"
CONF_PROMPT = "prompt"  # System instructions (Jinja); rendered by Home Assistant

# Default system prompt for new installs and when the option is empty.
# (Jinja: ha_name, user_name, llm_context — same variables as core LLM prompts.)
DEFAULT_SYSTEM_PROMPT = """You are an assistant for Home Assistant, the open-source home automation platform.
Answer truthfully. Reply in plain text unless the user asks for another format (e.g. markdown or a list).
When tools are available to read or change the home, use them when the user's request needs current state or actions.
Keep answers concise for short questions; add detail only when asked or when it clearly helps."""
CONF_TEMPERATURE = "temperature"
CONF_TOP_P = "top_p"
CONF_THINKING_ENABLED = "thinking_enabled"
CONF_REASONING_EFFORT = "reasoning_effort"
CONF_API_KEY = "api_key" # Already defined in homeassistant.const, but useful here
CONF_BASE_URL = "base_url" # Added for clarity, though set internally

# Service related (Image generation removed)
CONF_FILENAMES = "filenames" # Kept for potential future file support if DeepSeek adds it

# Default values — DeepSeek V4 (legacy IDs deepseek-chat / deepseek-reasoner retire 2026-07-24 per DeepSeek)
RECOMMENDED_CHAT_MODEL = "deepseek-v4-flash"

# (model_id, English label for selector; override via translations selector.chat_model.options)
CHAT_MODEL_OPTIONS: tuple[tuple[str, str], ...] = (
    ("deepseek-v4-flash", "DeepSeek V4 Flash (fast, default)"),
    ("deepseek-v4-pro", "DeepSeek V4 Pro (most capable)"),
    ("deepseek-chat", "Legacy: deepseek-chat (until 2026-07-24)"),
    ("deepseek-reasoner", "Legacy: deepseek-reasoner (until 2026-07-24)"),
)
# Adjusted default tokens, temperature, top_p if needed, keeping OpenAI's for now
RECOMMENDED_MAX_TOKENS = 1500
RECOMMENDED_TEMPERATURE = 1.0
RECOMMENDED_TOP_P = 1.0
DEFAULT_THINKING_ENABLED = False
# DeepSeek thinking mode: API accepts low/medium/high/max/xhigh (low/medium→high, xhigh→max).
REASONING_EFFORT_SELECT: tuple[tuple[str, str], ...] = (
    ("low", "Low"),
    ("medium", "Medium"),
    ("high", "High"),
    ("max", "Max"),
    ("xhigh", "xHigh"),
)
REASONING_EFFORT_VALUES: frozenset[str] = frozenset(v for v, _ in REASONING_EFFORT_SELECT)
RECOMMENDED_REASONING_EFFORT = "high"

# DeepSeek API endpoint
DEEPSEEK_API_BASE_URL = "https://api.deepseek.com/v1"


def deepseek_chat_extra_body(*, thinking_enabled: bool) -> dict[str, Any]:
    """Build OpenAI-SDK extra_body for DeepSeek thinking/reasoning toggle."""
    thinking_type = "enabled" if thinking_enabled else "disabled"
    return {"thinking": {"type": thinking_type}}


def coerce_max_tokens(value: Any, *, fallback: int = RECOMMENDED_MAX_TOKENS) -> int:
    """Parse max_tokens from config options (int/float/str); clamp to a sane range."""
    try:
        n = int(float(value))
    except (TypeError, ValueError):
        return fallback
    return max(1, min(n, 1_000_000))


def normalized_reasoning_effort(value: Any) -> str:
    """Return a valid reasoning_effort string for the DeepSeek API."""
    if isinstance(value, str) and value in REASONING_EFFORT_VALUES:
        return value
    return RECOMMENDED_REASONING_EFFORT


def deepseek_chat_thinking_params(
    *, thinking_enabled: bool, reasoning_effort: str = RECOMMENDED_REASONING_EFFORT
) -> dict[str, Any]:
    """kwargs for chat.completions.create (OpenAI SDK) matching DeepSeek thinking docs."""
    params: dict[str, Any] = {
        "extra_body": deepseek_chat_extra_body(thinking_enabled=thinking_enabled),
    }
    if thinking_enabled:
        params["reasoning_effort"] = normalized_reasoning_effort(reasoning_effort)
    return params


# Removed OpenAI specific constants
# CONF_RECOMMENDED = "recommended" # No longer using OpenAI recommended toggle
# CONF_WEB_SEARCH = "web_search"
# CONF_WEB_SEARCH_USER_LOCATION = "user_location"
# CONF_WEB_SEARCH_CONTEXT_SIZE = "search_context_size"
# CONF_WEB_SEARCH_CITY = "city"
# CONF_WEB_SEARCH_REGION = "region"
# CONF_WEB_SEARCH_COUNTRY = "country"
# CONF_WEB_SEARCH_TIMEZONE = "timezone"
# RECOMMENDED_WEB_SEARCH = False
# RECOMMENDED_WEB_SEARCH_CONTEXT_SIZE = "medium"
# RECOMMENDED_WEB_SEARCH_USER_LOCATION = False
# UNSUPPORTED_MODELS = [...] # Removed OpenAI unsupported models list

