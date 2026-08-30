"""The forms the config and subentry flows show.

Kept apart from ``config_flow.py`` so that module is about flow control - which
step follows which, and what to do with the answers - while what each field
looks like is a table you can read top to bottom.

The advanced step is that table: ``_ADVANCED_FIELDS`` gives every setting its
default and its selector, and ``ADVANCED_SECTIONS`` decides which of them
appear together and in what order.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import voluptuous as vol  # pyright: ignore[reportMissingImports]

from homeassistant.config_entries import ConfigEntry  # pyright: ignore[reportMissingImports]
from homeassistant.const import CONF_API_KEY  # pyright: ignore[reportMissingImports]
from homeassistant.helpers.selector import (  # pyright: ignore[reportMissingImports]
    BooleanSelector,
    NumberSelector,
    NumberSelectorConfig,
    SelectOptionDict,
    SelectSelector,
    SelectSelectorConfig,
    TextSelector,
    TextSelectorConfig,
    TextSelectorType,
)

from .const import (
    CHAT_MODEL_OPTIONS,
    CONF_CHAT_MODEL,
    CONF_INCLUDE_USER_CONTEXT,
    CONF_MAX_HISTORY_ROUNDS,
    CONF_MAX_TOKENS,
    CONF_MAX_TOOL_ITERATIONS,
    CONF_MAX_TOOL_RESULT_CHARS,
    CONF_REASONING_EFFORT,
    CONF_REQUEST_TIMEOUT,
    CONF_STRIP_MARKDOWN,
    CONF_TEMPERATURE,
    CONF_THINKING_ENABLED,
    CONF_TOP_P,
    CONF_VISION_ENABLED,
    CONF_BASE_URL,
    CONF_BRAVE_API_KEY,
    DEEPSEEK_API_BASE_URL,
    DEFAULT_INCLUDE_USER_CONTEXT,
    DEFAULT_STRIP_MARKDOWN,
    DEFAULT_THINKING_ENABLED,
    DEFAULT_VISION_ENABLED,
    MAX_HISTORY_ROUNDS_UPPER_BOUND,
    MAX_TOKENS_UPPER_BOUND,
    MAX_TOOL_ITERATIONS_UPPER_BOUND,
    MAX_TOOL_RESULT_CHARS_UPPER_BOUND,
    REASONING_EFFORT_SELECT,
    RECOMMENDED_CHAT_MODEL,
    RECOMMENDED_MAX_HISTORY_ROUNDS,
    RECOMMENDED_MAX_TOKENS,
    RECOMMENDED_MAX_TOOL_ITERATIONS,
    RECOMMENDED_MAX_TOOL_RESULT_CHARS,
    RECOMMENDED_REASONING_EFFORT,
    RECOMMENDED_REQUEST_TIMEOUT,
    RECOMMENDED_TEMPERATURE,
    RECOMMENDED_TOP_P,
    REQUEST_TIMEOUT_LOWER_BOUND,
    REQUEST_TIMEOUT_UPPER_BOUND,
)
from .options import (
    coerce_max_history_rounds,
    coerce_max_tokens,
    coerce_max_tool_iterations,
    coerce_max_tool_result_chars,
    coerce_request_timeout,
)

#: The advanced step, grouped. Order and collapsed state follow how often a
#: setting is actually touched: the way an agent answers is open, the rest is
#: folded away until someone goes looking for it.
SECTION_RESPONSE = "response"
SECTION_TOOLS = "tools"
SECTION_CONVERSATION = "conversation"
SECTION_LIMITS = "limits"

ADVANCED_SECTIONS: tuple[tuple[str, tuple[str, ...], bool], ...] = (
    (
        SECTION_RESPONSE,
        (
            CONF_MAX_TOKENS,
            CONF_TEMPERATURE,
            CONF_TOP_P,
            CONF_THINKING_ENABLED,
            CONF_REASONING_EFFORT,
        ),
        False,
    ),
    (SECTION_TOOLS, (CONF_MAX_TOOL_ITERATIONS, CONF_MAX_TOOL_RESULT_CHARS), True),
    (
        SECTION_CONVERSATION,
        (CONF_STRIP_MARKDOWN, CONF_INCLUDE_USER_CONTEXT, CONF_MAX_HISTORY_ROUNDS),
        True,
    ),
    (SECTION_LIMITS, (CONF_REQUEST_TIMEOUT, CONF_VISION_ENABLED), True),
)

#: How one advanced field works out the ``default=`` its marker carries.
_Default = Callable[[str, Mapping[str, Any]], Any]


def _stored(fallback: Any, coerce: Callable[[Any], Any] | None = None) -> _Default:
    """Default to what the agent stored, else to the recommended value.

    ``coerce`` is the same one the reader uses, so a value saved by an older
    form - or outside today's bounds - is shown as the value that will actually
    be used, not as the one that was stored.
    """

    def resolve(key: str, options: Mapping[str, Any]) -> Any:
        value = options.get(key, fallback)
        return coerce(value) if coerce is not None else value

    return resolve


def _fixed(value: Any) -> _Default:
    """Default to the recommended value, whatever the agent stored.

    Only sampling uses this. The stored value still reaches the form through
    ``add_suggested_values_to_schema``; what this decides is what the field
    falls back to, and for temperature and top_p that is the recommendation.
    """

    def resolve(_key: str, _options: Mapping[str, Any]) -> Any:
        return value

    return resolve


#: Every setting the advanced step can show: how to default it, how to draw it.
_ADVANCED_FIELDS: dict[str, tuple[_Default, Callable[[], Any]]] = {
    CONF_MAX_TOKENS: (
        _stored(RECOMMENDED_MAX_TOKENS, coerce_max_tokens),
        lambda: NumberSelector(
            NumberSelectorConfig(min=1, max=MAX_TOKENS_UPPER_BOUND, mode="box", step=1)
        ),
    ),
    CONF_TEMPERATURE: (
        _fixed(RECOMMENDED_TEMPERATURE),
        lambda: NumberSelector(
            NumberSelectorConfig(min=0, max=2, step=0.05, mode="slider")
        ),
    ),
    CONF_TOP_P: (
        _fixed(RECOMMENDED_TOP_P),
        lambda: NumberSelector(
            NumberSelectorConfig(min=0, max=1, step=0.05, mode="slider")
        ),
    ),
    CONF_THINKING_ENABLED: (
        _stored(DEFAULT_THINKING_ENABLED),
        BooleanSelector,
    ),
    CONF_REASONING_EFFORT: (
        _stored(RECOMMENDED_REASONING_EFFORT),
        lambda: SelectSelector(
            SelectSelectorConfig(
                options=[
                    SelectOptionDict(label=value, value=value)
                    for value, _ in REASONING_EFFORT_SELECT
                ],
                translation_key=CONF_REASONING_EFFORT,
            )
        ),
    ),
    CONF_MAX_TOOL_ITERATIONS: (
        _stored(RECOMMENDED_MAX_TOOL_ITERATIONS, coerce_max_tool_iterations),
        lambda: NumberSelector(
            NumberSelectorConfig(
                min=1, max=MAX_TOOL_ITERATIONS_UPPER_BOUND, mode="box", step=1
            )
        ),
    ),
    CONF_MAX_TOOL_RESULT_CHARS: (
        _stored(RECOMMENDED_MAX_TOOL_RESULT_CHARS, coerce_max_tool_result_chars),
        lambda: NumberSelector(
            NumberSelectorConfig(
                min=0, max=MAX_TOOL_RESULT_CHARS_UPPER_BOUND, mode="box", step=500
            )
        ),
    ),
    CONF_MAX_HISTORY_ROUNDS: (
        _stored(RECOMMENDED_MAX_HISTORY_ROUNDS, coerce_max_history_rounds),
        lambda: NumberSelector(
            NumberSelectorConfig(
                min=0, max=MAX_HISTORY_ROUNDS_UPPER_BOUND, mode="box", step=1
            )
        ),
    ),
    CONF_REQUEST_TIMEOUT: (
        _stored(RECOMMENDED_REQUEST_TIMEOUT, coerce_request_timeout),
        lambda: NumberSelector(
            NumberSelectorConfig(
                min=REQUEST_TIMEOUT_LOWER_BOUND,
                max=REQUEST_TIMEOUT_UPPER_BOUND,
                mode="box",
                step=5,
                unit_of_measurement="s",
            )
        ),
    ),
    CONF_STRIP_MARKDOWN: (_stored(DEFAULT_STRIP_MARKDOWN), BooleanSelector),
    CONF_INCLUDE_USER_CONTEXT: (
        _stored(DEFAULT_INCLUDE_USER_CONTEXT),
        BooleanSelector,
    ),
    CONF_VISION_ENABLED: (_stored(DEFAULT_VISION_ENABLED), BooleanSelector),
}


def advanced_field(key: str, options: Mapping[str, Any]) -> tuple[Any, Any]:
    """Return the (marker, selector) pair for one advanced setting."""
    try:
        default, selector = _ADVANCED_FIELDS[key]
    except KeyError:
        raise ValueError(f"no selector defined for {key}") from None
    return vol.Optional(key, default=default(key, options)), selector()


def flatten_sections(user_input: dict[str, Any]) -> dict[str, Any]:
    """Undo the nesting a sectioned form returns, so a subentry stays flat."""
    flat: dict[str, Any] = {}
    for value in user_input.values():
        if isinstance(value, dict):
            flat.update(value)
    return flat


def normalize_llm_hass_api(value: Any) -> list[str] | None:
    """Normalize CONF_LLM_HASS_API to a list for multi-select, or None if unset."""
    if isinstance(value, list):
        return value if value else None
    if isinstance(value, str):
        return [value] if value != "none" else None
    return None


def _chat_model_select_options() -> list[SelectOptionDict]:
    return [SelectOptionDict(value=v, label=lbl) for v, lbl in CHAT_MODEL_OPTIONS]


def chat_model_selector() -> SelectSelector:
    return SelectSelector(
        SelectSelectorConfig(
            options=_chat_model_select_options(),
            custom_value=True,
            translation_key=CONF_CHAT_MODEL,
        )
    )


def _api_key_selector() -> TextSelector:
    return TextSelector(
        TextSelectorConfig(
            type=TextSelectorType.PASSWORD,
            autocomplete="current-password",
        )
    )


def _base_url_selector() -> TextSelector:
    return TextSelector(
        TextSelectorConfig(
            type=TextSelectorType.URL,
            autocomplete="url",
        )
    )


def get_user_step_schema() -> vol.Schema:
    """Schema for initial config (API key, URL, model, optional Brave key)."""
    return vol.Schema(
        {
            vol.Required(CONF_API_KEY): _api_key_selector(),
            vol.Optional(CONF_BASE_URL, default=DEEPSEEK_API_BASE_URL): _base_url_selector(),
            vol.Optional(CONF_BRAVE_API_KEY): _api_key_selector(),
            vol.Optional(
                CONF_CHAT_MODEL, default=RECOMMENDED_CHAT_MODEL
            ): chat_model_selector(),
        }
    )


STEP_REAUTH_DATA_SCHEMA = vol.Schema(
    {
        vol.Required(CONF_API_KEY): _api_key_selector(),
    }
)


def get_reconfigure_step_schema(entry: ConfigEntry) -> vol.Schema:
    """Schema for reconfigure (DeepSeek key, base URL, optional Brave key).

    Brave key: leave empty to keep the current key; enter ``-`` to remove it
    (clears web search LLM API registration after reload).
    """
    return vol.Schema(
        {
            vol.Required(CONF_API_KEY): _api_key_selector(),
            vol.Optional(
                CONF_BASE_URL,
                default=entry.data.get(CONF_BASE_URL, DEEPSEEK_API_BASE_URL),
            ): _base_url_selector(),
            vol.Optional(CONF_BRAVE_API_KEY): _api_key_selector(),
        }
    )
