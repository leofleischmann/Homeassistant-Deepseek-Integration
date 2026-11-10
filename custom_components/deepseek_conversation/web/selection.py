"""Which web tools one agent gets, worked out from stored settings alone.

Two questions, both answered from plain mappings so the stubbed suite can ask
them: what can this config entry offer at all, and what did this agent ask for.
An agent only gets what both agree on - a setting that outlived the key it
needs is simply not honoured, and the agent form clears it when next opened.

Affects / influenced by:
- ``agent_tools.py`` builds exactly the tools ``web_tools_plan`` names.
- ``config_flow.py`` shows the web search field only while
  ``available_search_providers`` is not empty.
- ``debug.py`` reports both.
- ``factory.py`` must be able to build every id returned here.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ..const import (
    CONF_BRAVE_API_KEY,
    CONF_WEB_SEARCH,
    WEB_SEARCH_PROVIDER_BRAVE,
)


def brave_api_key(entry_data: Mapping[str, Any]) -> str:
    """The Brave subscription token on this entry, or an empty string."""
    return str(entry_data.get(CONF_BRAVE_API_KEY) or "").strip()


def available_search_providers(entry_data: Mapping[str, Any]) -> tuple[str, ...]:
    """The search providers this entry can offer its agents, in form order."""
    providers: list[str] = []
    if brave_api_key(entry_data):
        providers.append(WEB_SEARCH_PROVIDER_BRAVE)
    return tuple(providers)


@dataclass(frozen=True, slots=True)
class WebToolsPlan:
    """The web tools one agent ends up with."""

    #: Id of the search provider to use, or ``None`` for no web search.
    search_provider: str | None = None

    def __bool__(self) -> bool:
        """Whether there is anything of ours to add to the agent's tools."""
        return self.search_provider is not None


def web_tools_plan(
    entry_data: Mapping[str, Any], options: Mapping[str, Any]
) -> WebToolsPlan:
    """What this agent asked for, limited to what its entry can provide."""
    available = available_search_providers(entry_data)
    if options.get(CONF_WEB_SEARCH) and WEB_SEARCH_PROVIDER_BRAVE in available:
        return WebToolsPlan(search_provider=WEB_SEARCH_PROVIDER_BRAVE)
    return WebToolsPlan()
