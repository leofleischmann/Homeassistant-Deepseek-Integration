"""Constants for the DeepSeek Conversation integration.

Values only. The logic that reads them lives next to it, split by what it
decides about:

* ``models.py`` — which endpoint and which model id is in play,
* ``options.py`` — turning stored agent settings into usable values,
* ``request_builder.py`` — assembling the kwargs for a chat completion.

Keeping this module free of behaviour is what lets those three be imported and
unit-tested on their own.
"""

from __future__ import annotations

import logging
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
#: Removed in 1.8.0; only options.fold_context_switch() still knows the name,
#: to turn an entry that had it switched off into the two limits it forced.
CONF_CONTEXT_MANAGEMENT_ENABLED = "context_management_enabled"
CONF_MAX_TOOL_RESULT_CHARS = "max_tool_result_chars"
CONF_MAX_HISTORY_ROUNDS = "max_history_rounds"
CONF_INCLUDE_USER_CONTEXT = "include_user_context"
CONF_REQUEST_TIMEOUT = "request_timeout"
CONF_BASE_URL = "base_url"
CONF_BRAVE_API_KEY = "brave_api_key"
#: Per agent: may this one search the web with the entry's Brave key. Until
#: 1.8.4 the tool was a globally registered LLM API instead, which put it in
#: every *other* conversation integration's API picker as well (#38).
CONF_WEB_SEARCH = "web_search"
#: How that API's id was built, so migration can recognise an agent that had
#: selected it and turn the selection into CONF_WEB_SEARCH.
LEGACY_WEB_SEARCH_API_ID_PREFIX = f"{DOMAIN}_web_search_"
CONF_AGENT = "agent"
CONF_CONFIG_ENTRY = "config_entry"
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
#: Entries still configured with one are migrated by models.migrate_legacy_chat_model().
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

# Bounds for the two context limits (see context_trim.py). Both are off at
# zero; a tool-result cap below MIN_TOOL_RESULT_CHARS would leave no room for
# the truncation notice, so a non-zero value is raised to it.
MAX_TOOL_RESULT_CHARS_UPPER_BOUND = 100_000
MIN_TOOL_RESULT_CHARS = 500
MAX_HISTORY_ROUNDS_UPPER_BOUND = 200

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

#: The settings the first step of the agent form asks for. Everything else is
#: an override of a recommended default. See options.recommended_agent_options().
BASIC_AGENT_OPTIONS: frozenset[str] = frozenset(
    {
        CONF_RECOMMENDED,
        CONF_PROMPT,
        CONF_LLM_HASS_API,
        CONF_WEB_SEARCH,
        CONF_CHAT_MODEL,
    }
)

#: Settings that only mean something when a person is on the other end.
#: Markdown stripping and naming the speaker are about being spoken to, and a
#: history cap needs a history - an AI Task chat log is a single turn.
#: See options.ai_task_options_from().
ASSIST_ONLY_OPTIONS: frozenset[str] = frozenset(
    {CONF_STRIP_MARKDOWN, CONF_INCLUDE_USER_CONTEXT, CONF_MAX_HISTORY_ROUNDS}
)

#: What an entry carried when its owner never touched the setting. 1.7.0 wrote
#: every default into the entry, so a stored ``False`` here says nothing about
#: what the user wanted - it is just the old default written down.
#: See options.adopt_strip_markdown_default().
PREVIOUS_STRIP_MARKDOWN_DEFAULT = False
