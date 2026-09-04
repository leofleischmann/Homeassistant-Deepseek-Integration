"""The Brave Search web tool.

Affects / influenced by:
- The credential lives in ``entry.data`` (setup / reconfigure in
  ``config_flow.py``); ``CONF_WEB_SEARCH`` in an agent's subentry says whether
  that agent may use it.
- ``agent_tools.py`` is what puts the tool in front of an agent. It is
  deliberately never registered with ``llm.async_register_api``: that registry
  is global, so registering it put a "Web Search (Brave)" entry into every
  *other* conversation integration's settings as well (#38).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import voluptuous as vol # pyright: ignore[reportMissingImports]

from homeassistant.config_entries import ConfigEntry  # pyright: ignore[reportMissingImports]
from homeassistant.core import HomeAssistant  # pyright: ignore[reportMissingImports]
from homeassistant.exceptions import HomeAssistantError  # pyright: ignore[reportMissingImports]
from homeassistant.helpers import llm  # pyright: ignore[reportMissingImports]
from homeassistant.helpers.httpx_client import get_async_client  # pyright: ignore[reportMissingImports]
from homeassistant.helpers.llm import (  # pyright: ignore[reportMissingImports]
    LLMContext,
    ToolInput,
)
from homeassistant.util.json import JsonObjectType  # pyright: ignore[reportMissingImports]

from .const import CONF_BRAVE_API_KEY, CONF_WEB_SEARCH, LOGGER

BRAVE_WEB_SEARCH_URL = "https://api.search.brave.com/res/v1/web/search"
DEFAULT_RESULT_COUNT = 5
MAX_RESULT_COUNT = 10
#: This runs inside the Assist tool loop, so a hanging search hangs the whole
#: voice turn. HA's shared client carries no explicit timeout of its own.
SEARCH_TIMEOUT = 10.0
WEB_SEARCH_API_PROMPT = (
    "Use web_search only for current facts from the public web "
    "(news, products, documentation, general knowledge). "
    "Do not use it for Home Assistant device state or home control."
)


def brave_api_key(entry: ConfigEntry) -> str:
    """The Brave subscription token on this entry, or an empty string."""
    return (entry.data.get(CONF_BRAVE_API_KEY) or "").strip()


def web_search_enabled(entry: ConfigEntry, options: Mapping[str, Any]) -> bool:
    """Whether this agent asked for web search and the entry can provide it."""
    return bool(options.get(CONF_WEB_SEARCH)) and bool(brave_api_key(entry))


class WebSearchTool(llm.Tool):
    """Call Brave Search and return compact web results."""

    name = "web_search"
    description = (
        "Search the public web via Brave Search. "
        "Use for up-to-date information that is not available from Home Assistant tools."
    )
    parameters = vol.Schema(
        {
            vol.Required("query"): str,
            vol.Optional("count", default=DEFAULT_RESULT_COUNT): vol.All(
                vol.Coerce(int), vol.Range(min=1, max=MAX_RESULT_COUNT)
            ),
        }
    )

    def __init__(self, api_key: str) -> None:
        """Store the Brave subscription token for this tool instance."""
        self._api_key = api_key

    async def async_call(
        self,
        hass: HomeAssistant,
        tool_input: ToolInput,
        llm_context: LLMContext,
    ) -> JsonObjectType:
        """Execute Brave web search and return title/url/description snippets."""
        query = str(tool_input.tool_args["query"]).strip()
        if not query:
            raise HomeAssistantError("web_search requires a non-empty query")

        count = int(tool_input.tool_args.get("count", DEFAULT_RESULT_COUNT))
        count = max(1, min(count, MAX_RESULT_COUNT))

        LOGGER.debug(
            "[Debug web_search]: query=%r count=%s platform=%s",
            query,
            count,
            llm_context.platform,
        )

        client = get_async_client(hass)
        try:
            response = await client.get(
                BRAVE_WEB_SEARCH_URL,
                params={"q": query, "count": count},
                headers={
                    "Accept": "application/json",
                    "Accept-Encoding": "gzip",
                    "X-Subscription-Token": self._api_key,
                },
                timeout=SEARCH_TIMEOUT,
            )
        except Exception as err:
            LOGGER.error("[Debug web_search]: request failed: %s", err)
            raise HomeAssistantError(f"Brave Search request failed: {err}") from err

        if response.status_code in (401, 403):
            LOGGER.error(
                "[Debug web_search]: auth rejected status=%s", response.status_code
            )
            raise HomeAssistantError(
                "Brave Search rejected the API key (unauthorized)"
            )
        if response.status_code >= 400:
            body_preview = (response.text or "")[:200]
            LOGGER.error(
                "[Debug web_search]: HTTP %s body=%r",
                response.status_code,
                body_preview,
            )
            raise HomeAssistantError(
                f"Brave Search returned HTTP {response.status_code}: {body_preview}"
            )

        try:
            payload: dict[str, Any] = response.json()
        except Exception as err:
            LOGGER.error("[Debug web_search]: invalid JSON: %s", err)
            raise HomeAssistantError(
                f"Brave Search returned invalid JSON: {err}"
            ) from err

        raw_results = (payload.get("web") or {}).get("results") or []
        results: list[dict[str, str]] = []
        for item in raw_results[:count]:
            if not isinstance(item, dict):
                continue
            title = item.get("title")
            url = item.get("url")
            description = item.get("description")
            if not title and not url:
                continue
            results.append(
                {
                    "title": str(title or ""),
                    "url": str(url or ""),
                    "description": str(description or ""),
                }
            )

        LOGGER.debug(
            "[Debug web_search]: returning %s result(s) for query=%r",
            len(results),
            query,
        )
        return {"query": query, "results": results}
