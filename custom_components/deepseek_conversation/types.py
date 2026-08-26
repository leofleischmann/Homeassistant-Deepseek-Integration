"""Shared types and config-entry shape helpers.

One config entry holds the credentials for an API key. Every agent - a
conversation agent or an AI Task entity - is a subentry of it with its own
prompt, model and tools, so a fast agent for voice and a capable one for
automations can share a single key.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, TypeAlias

import openai
from homeassistant.config_entries import ConfigEntry, ConfigSubentry  # pyright: ignore[reportMissingImports]
from homeassistant.helpers import device_registry as dr  # pyright: ignore[reportMissingImports]

from .const import (
    CONF_CHAT_MODEL,
    DOMAIN,
    RECOMMENDED_CHAT_MODEL,
    SUBENTRY_TYPE_CONVERSATION,
)
from .usage_metrics import UsageTracker


@dataclass
class DeepSeekRuntimeData:
    """Per-config-entry runtime state (OpenAI client + usage tracking)."""

    client: openai.AsyncClient
    usage: UsageTracker
    http_version: str | None = None
    warned_unexpected_reasoning: bool = False


DeepSeekConfigEntry: TypeAlias = ConfigEntry[DeepSeekRuntimeData]


def agent_subentries(
    entry: ConfigEntry, subentry_type: str
) -> list[ConfigSubentry]:
    """Return this entry's subentries of one type, in creation order."""
    return [
        subentry
        for subentry in entry.subentries.values()
        if subentry.subentry_type == subentry_type
    ]


def default_agent_options(
    entry: ConfigEntry, subentry_type: str = SUBENTRY_TYPE_CONVERSATION
) -> Mapping[str, Any]:
    """Return the options the entry-wide actions run with.

    ``generate_content`` and ``run_debug`` address a config entry, not a
    specific agent, so they follow the first agent of the given type. An entry
    with no agent at all falls back to an empty mapping, which every reader
    resolves to the recommended defaults.
    """
    subentries = agent_subentries(entry, subentry_type)
    return subentries[0].data if subentries else {}


def usage_device_info(entry: ConfigEntry) -> dr.DeviceInfo:
    """Device that carries the token counters and the reset button.

    Deliberately the config entry rather than an agent: usage is billed per API
    key, and the counters have to keep adding up across every agent that shares
    it. Each agent gets its own device from its subentry.
    """
    return dr.DeviceInfo(
        identifiers={(DOMAIN, entry.entry_id)},
        name=entry.title,
        manufacturer="DeepSeek",
        model="DeepSeek API",
        entry_type=dr.DeviceEntryType.SERVICE,
    )


def agent_device_info(entry: ConfigEntry, subentry: ConfigSubentry) -> dr.DeviceInfo:
    """Device for one agent, named after the subentry."""
    return dr.DeviceInfo(
        identifiers={(DOMAIN, subentry.subentry_id)},
        name=subentry.title,
        manufacturer="DeepSeek",
        model=str(subentry.data.get(CONF_CHAT_MODEL, RECOMMENDED_CHAT_MODEL)),
        entry_type=dr.DeviceEntryType.SERVICE,
    )
