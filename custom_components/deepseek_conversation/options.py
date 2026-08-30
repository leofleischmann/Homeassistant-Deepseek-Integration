"""Reading stored agent settings.

A subentry stores only what its owner actually chose; everything else resolves
to a RECOMMENDED_* / DEFAULT_* value from const.py as it is read. That is what
lets a later change to a default reach every agent that never overrode it - and
it means every read has to survive a missing key, a string where a number
belongs, or a value from an older version of the form.

Two kinds of function live here:

* ``coerce_*`` / ``*_from_options`` — turn one stored value into a usable one,
  clamped to the bounds in const.py;
* ``recommended_agent_options`` and the three migration helpers — reshape a
  whole settings mapping when the form or the defaults changed underneath it.

Pure functions over mappings, so this module can be unit-tested on its own.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .const import (
    ASSIST_ONLY_OPTIONS,
    BASIC_AGENT_OPTIONS,
    CONF_CONTEXT_MANAGEMENT_ENABLED,
    CONF_MAX_HISTORY_ROUNDS,
    CONF_MAX_TOOL_RESULT_CHARS,
    CONF_REQUEST_TIMEOUT,
    CONF_STRIP_MARKDOWN,
    MAX_HISTORY_ROUNDS_UPPER_BOUND,
    MAX_TOKENS_UPPER_BOUND,
    MAX_TOOL_ITERATIONS_UPPER_BOUND,
    MAX_TOOL_RESULT_CHARS_UPPER_BOUND,
    MIN_BLOCKING_REQUEST_TIMEOUT,
    MIN_TOOL_RESULT_CHARS,
    PREVIOUS_STRIP_MARKDOWN_DEFAULT,
    RECOMMENDED_MAX_HISTORY_ROUNDS,
    RECOMMENDED_MAX_TOKENS,
    RECOMMENDED_MAX_TOOL_ITERATIONS,
    RECOMMENDED_MAX_TOOL_RESULT_CHARS,
    RECOMMENDED_REASONING_EFFORT,
    RECOMMENDED_REQUEST_TIMEOUT,
    REASONING_EFFORT_VALUES,
    REQUEST_TIMEOUT_LOWER_BOUND,
    REQUEST_TIMEOUT_UPPER_BOUND,
)


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


def coerce_max_tool_result_chars(
    value: Any, *, fallback: int = RECOMMENDED_MAX_TOOL_RESULT_CHARS
) -> int:
    """Parse max_tool_result_chars; 0 disables truncation."""
    try:
        n = int(float(value))
    except (TypeError, ValueError):
        return fallback
    if n <= 0:
        return 0
    return max(MIN_TOOL_RESULT_CHARS, min(n, MAX_TOOL_RESULT_CHARS_UPPER_BOUND))


def max_tool_result_chars_from_options(options: Mapping[str, Any]) -> int:
    """Effective tool-result character limit; 0 means no truncation."""
    return coerce_max_tool_result_chars(
        options.get(CONF_MAX_TOOL_RESULT_CHARS, RECOMMENDED_MAX_TOOL_RESULT_CHARS)
    )


def coerce_max_history_rounds(
    value: Any, *, fallback: int = RECOMMENDED_MAX_HISTORY_ROUNDS
) -> int:
    """Parse max_history_rounds; 0 keeps the full conversation history."""
    try:
        n = int(float(value))
    except (TypeError, ValueError):
        return fallback
    if n <= 0:
        return 0
    return min(n, MAX_HISTORY_ROUNDS_UPPER_BOUND)


def max_history_rounds_from_options(options: Mapping[str, Any]) -> int:
    """Effective user-turn history cap; 0 means unlimited."""
    return coerce_max_history_rounds(
        options.get(CONF_MAX_HISTORY_ROUNDS, RECOMMENDED_MAX_HISTORY_ROUNDS)
    )


def normalized_reasoning_effort(value: Any) -> str:
    """Return a valid reasoning_effort string for the DeepSeek API."""
    if isinstance(value, str) and value in REASONING_EFFORT_VALUES:
        return value
    return RECOMMENDED_REASONING_EFFORT


def recommended_agent_options(options: Mapping[str, Any]) -> dict[str, Any]:
    """Drop the overrides an agent no longer wants to keep.

    Switching an agent back to the recommended settings has to forget what was
    set behind them. Keeping the values would leave the agent running on a
    reply limit or a reasoning effort its own form no longer shows.
    """
    return {key: value for key, value in options.items() if key in BASIC_AGENT_OPTIONS}


def ai_task_options_from(options: Mapping[str, Any]) -> dict[str, Any]:
    """Return agent settings with the Assist-only ones removed."""
    return {
        key: value
        for key, value in options.items()
        if key not in ASSIST_ONLY_OPTIONS
    }


def adopt_strip_markdown_default(options: dict[str, Any]) -> dict[str, Any]:
    """Let an untouched markdown setting follow the new default.

    Dropping the key rather than flipping it is the point: the agent then
    follows DEFAULT_STRIP_MARKDOWN, and an owner who had deliberately turned it
    on keeps that. Only the value that was merely the old default gives way.
    """
    if options.get(CONF_STRIP_MARKDOWN) == PREVIOUS_STRIP_MARKDOWN_DEFAULT:
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
