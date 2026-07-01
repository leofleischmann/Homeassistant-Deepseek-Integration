"""Context trimming for Assist API requests.

Stage 1: cap serialized Home Assistant tool result JSON before it is sent to
DeepSeek (large GetLiveContext payloads are the usual context-limit trigger).
Stage 2: drop oldest complete user rounds from the messages array while keeping
the system prompt and intact assistant/tool chains per round.

Used from conversation.py when building or extending the messages array. Options:
CONF_CONTEXT_MANAGEMENT_ENABLED, CONF_MAX_TOOL_RESULT_CHARS,
CONF_MAX_HISTORY_ROUNDS in config_flow.py.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from .const import (
    CONF_CONTEXT_MANAGEMENT_ENABLED,
    CONF_MAX_HISTORY_ROUNDS,
    CONF_MAX_TOOL_RESULT_CHARS,
    DEFAULT_CONTEXT_MANAGEMENT_ENABLED,
    LOGGER,
    RECOMMENDED_MAX_HISTORY_ROUNDS,
    RECOMMENDED_MAX_TOOL_RESULT_CHARS,
)

MAX_TOOL_RESULT_CHARS_UPPER_BOUND = 100_000
MIN_TOOL_RESULT_CHARS = 500
MAX_HISTORY_ROUNDS_UPPER_BOUND = 200

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
    if not context_management_enabled(options):
        return 0
    return coerce_max_history_rounds(
        options.get(CONF_MAX_HISTORY_ROUNDS, RECOMMENDED_MAX_HISTORY_ROUNDS)
    )


def _split_system_and_user_rounds(
    messages: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[list[dict[str, Any]]]]:
    """Split API messages into leading system block and user-started rounds."""
    system: list[dict[str, Any]] = []
    rest_start = 0
    for index, message in enumerate(messages):
        if message.get("role") == "system":
            system.append(message)
            rest_start = index + 1
        else:
            break

    rest = messages[rest_start:]
    rounds: list[list[dict[str, Any]]] = []
    orphan_prefix: list[dict[str, Any]] = []
    current_round: list[dict[str, Any]] = []

    for message in rest:
        if message.get("role") == "user":
            if current_round:
                rounds.append(current_round)
            current_round = [message]
            continue
        if current_round:
            current_round.append(message)
        else:
            orphan_prefix.append(message)

    if current_round:
        rounds.append(current_round)

    if orphan_prefix:
        if rounds:
            rounds[0] = orphan_prefix + rounds[0]
        else:
            rounds = [orphan_prefix]

    return system, rounds


def trim_message_history_by_rounds(
    messages: list[dict[str, Any]],
    *,
    max_rounds: int,
) -> list[dict[str, Any]]:
    """Keep the system prompt and the newest ``max_rounds`` user turns intact.

    Each round starts at a user message and includes every following assistant
    and tool message until the next user message. Tool calls and tool results
    therefore stay paired when older rounds are removed.
    """
    if max_rounds <= 0 or not messages:
        return messages

    system, rounds = _split_system_and_user_rounds(messages)
    if len(rounds) <= max_rounds:
        return messages

    dropped_rounds = len(rounds) - max_rounds
    kept_rounds = rounds[dropped_rounds:]
    trimmed: list[dict[str, Any]] = [*system]
    for round_messages in kept_rounds:
        trimmed.extend(round_messages)

    LOGGER.debug(
        "[Debug context_trim]: trimmed history %d -> %d user round(s), "
        "dropped %d round(s), %d -> %d message(s)",
        len(rounds),
        len(kept_rounds),
        dropped_rounds,
        len(messages),
        len(trimmed),
    )
    return trimmed


def trim_messages_for_api(
    messages: list[dict[str, Any]],
    *,
    options: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Apply stage-2 history trimming before a chat completion request."""
    return trim_message_history_by_rounds(
        messages,
        max_rounds=max_history_rounds_from_options(options),
    )
