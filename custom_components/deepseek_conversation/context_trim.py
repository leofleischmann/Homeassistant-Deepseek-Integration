"""Context trimming for Assist API requests.

Stage 1: cap serialized Home Assistant tool result JSON before it is sent to
DeepSeek (large GetLiveContext payloads are the usual context-limit trigger).
Used from conversation.py when building the messages array. Options:
CONF_CONTEXT_MANAGEMENT_ENABLED, CONF_MAX_TOOL_RESULT_CHARS in config_flow.py.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from .const import (
    CONF_CONTEXT_MANAGEMENT_ENABLED,
    CONF_MAX_TOOL_RESULT_CHARS,
    DEFAULT_CONTEXT_MANAGEMENT_ENABLED,
    LOGGER,
    RECOMMENDED_MAX_TOOL_RESULT_CHARS,
)

MAX_TOOL_RESULT_CHARS_UPPER_BOUND = 100_000
MIN_TOOL_RESULT_CHARS = 500

_TRUNCATION_SUFFIX_TEMPLATE = (
    "\n… [truncated by DeepSeek integration, {omitted} chars omitted]"
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


def context_management_enabled(options: Mapping[str, Any]) -> bool:
    """Whether context trimming is active for this config entry."""
    return bool(
        options.get(CONF_CONTEXT_MANAGEMENT_ENABLED, DEFAULT_CONTEXT_MANAGEMENT_ENABLED)
    )


def max_tool_result_chars_from_options(options: Mapping[str, Any]) -> int:
    """Effective tool-result character limit; 0 means no truncation."""
    if not context_management_enabled(options):
        return 0
    return coerce_max_tool_result_chars(
        options.get(CONF_MAX_TOOL_RESULT_CHARS, RECOMMENDED_MAX_TOOL_RESULT_CHARS)
    )


def truncate_tool_result_json(
    serialized: str,
    *,
    max_chars: int,
    tool_name: str | None = None,
) -> str:
    """Shorten a tool result string if it exceeds ``max_chars``."""
    if max_chars <= 0 or len(serialized) <= max_chars:
        return serialized

    suffix_reserve = len(_TRUNCATION_SUFFIX_TEMPLATE.format(omitted=9_999_999))
    if suffix_reserve >= max_chars:
        LOGGER.warning(
            "[Debug context_trim]: max_tool_result_chars=%d too small for "
            "truncation suffix; skipping trim for tool %s",
            max_chars,
            tool_name or "unknown",
        )
        return serialized

    cut_at = max_chars - suffix_reserve
    truncated_body = serialized[:cut_at]
    omitted = len(serialized) - len(truncated_body)
    suffix = _TRUNCATION_SUFFIX_TEMPLATE.format(omitted=omitted)
    truncated = truncated_body + suffix
    LOGGER.debug(
        "[Debug context_trim]: truncated tool result %s %d -> %d chars",
        tool_name or "unknown",
        len(serialized),
        len(truncated),
    )
    return truncated


def format_tool_result_content(
    tool_result: Any,
    *,
    json_encoder: type[json.JSONEncoder],
    options: Mapping[str, Any],
    tool_name: str | None = None,
) -> str:
    """Serialize a tool result and apply the configured size cap."""
    serialized = json.dumps(tool_result, cls=json_encoder)
    return truncate_tool_result_json(
        serialized,
        max_chars=max_tool_result_chars_from_options(options),
        tool_name=tool_name,
    )
