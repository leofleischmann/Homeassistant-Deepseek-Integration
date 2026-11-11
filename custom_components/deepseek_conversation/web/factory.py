"""Building a search provider from a config entry.

The one place that knows both sides: which class answers to a provider id, and
where its credential and its HTTP client come from in Home Assistant. The
providers themselves take both as arguments and import neither.

Affects / influenced by:
- ``selection.py`` names the ids; every id it can return needs a branch here.
- ``agent_tools.py`` calls this once per conversation turn.
- ``SEARCH_TIMEOUT`` bounds every provider, because the search runs inside the
  Assist tool loop.
- DeepSeek Search borrows ``client.async_http_client``, the HTTP/2 client the
  chat API runs on: the search then travels on the connection the chat round
  that asked for it has just used, instead of opening one of its own.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from homeassistant.const import CONF_API_KEY  # pyright: ignore[reportMissingImports]
from homeassistant.core import HomeAssistant  # pyright: ignore[reportMissingImports]
from homeassistant.helpers.httpx_client import get_async_client  # pyright: ignore[reportMissingImports]

from ..client import async_http_client
from ..const import (
    CONF_BASE_URL,
    DEEPSEEK_API_BASE_URL,
    WEB_SEARCH_PROVIDER_BRAVE,
    WEB_SEARCH_PROVIDER_DEEPSEEK,
)
from .brave import BraveSearchProvider
from .deepseek import DeepSeekSearchProvider, DeepSeekWebClient
from .selection import brave_api_key
from .types import SearchProvider

#: This runs inside the Assist tool loop, so a hanging search hangs the whole
#: voice turn. HA's shared client carries no explicit timeout of its own.
SEARCH_TIMEOUT = 10.0


def build_search_provider(
    hass: HomeAssistant, entry_data: Mapping[str, Any], provider_id: str
) -> SearchProvider:
    """Return the provider behind ``provider_id``, bound to this entry.

    The HTTP client is handed over as a callable, so nothing is fetched until
    a search actually runs.
    """
    if provider_id == WEB_SEARCH_PROVIDER_DEEPSEEK:
        return DeepSeekSearchProvider(
            DeepSeekWebClient(
                lambda: async_http_client(hass),
                api_key=entry_data[CONF_API_KEY],
                base_url=entry_data.get(CONF_BASE_URL, DEEPSEEK_API_BASE_URL),
                timeout=SEARCH_TIMEOUT,
            )
        )
    if provider_id == WEB_SEARCH_PROVIDER_BRAVE:
        return BraveSearchProvider(
            lambda: get_async_client(hass),
            api_key=brave_api_key(entry_data),
            timeout=SEARCH_TIMEOUT,
        )
    raise ValueError(f"no web search provider is registered as {provider_id!r}")
