"""Context trimming for Assist API requests.

Caps serialized tool result JSON and optionally limits Assist history by user
turn before messages are sent to DeepSeek. Used from chat_session.py (history)
and chat_messages.py (tool results). Both
limits are off at zero, which is the only switch either of them needs.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from .const import LOGGER
from .options import (
    max_history_rounds_from_options,
    max_tool_result_chars_from_options,
)

_TRUNCATION_SUFFIX_TEMPLATE = (
    "\n… [truncated by DeepSeek integration, {omitted} chars omitted]"
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
    """Trim Assist history by user round before a chat completion request."""
    return trim_message_history_by_rounds(
        messages,
        max_rounds=max_history_rounds_from_options(options),
    )
