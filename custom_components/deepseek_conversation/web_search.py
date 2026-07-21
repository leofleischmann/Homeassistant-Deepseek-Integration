"""Brave Search web tool as a Home Assistant LLM API.

Affects / influenced by:
- Registered from ``__init__.async_setup_entry`` when ``CONF_BRAVE_API_KEY`` is set
  on the config entry; unregistered on entry unload.
- Credential lives in ``entry.data`` (setup / reconfigure in ``config_flow.py``).
- Appears in the Assist options multi-select via ``llm.async_get_apis``
  (``CONF_LLM_HASS_API``); the conversation tool loop in ``conversation.py``
  uses it only when the user selects this API.
"""

from __future__ import annotations

from typing import Any

import voluptuous as vol # pyright: ignore[reportMissingImports]

from homeassistant.config_entries import ConfigEntry  # pyright: ignore[reportMissingImports]
from homeassistant.core import HomeAssistant  # pyright: ignore[reportMissingImports]
from homeassistant.exceptions import HomeAssistantError  # pyright: ignore[reportMissingImports]
from homeassistant.helpers import llm  # pyright: ignore[reportMissingImports]
from homeassistant.helpers.httpx_client import get_async_client  # pyright: ignore[reportMissingImports]
from homeassistant.helpers.llm import (  # pyright: ignore[reportMissingImports]
    APIInstance,
    LLMContext,
    ToolInput,
)
from homeassistant.util.json import JsonObjectType  # pyright: ignore[reportMissingImports]

from .const import CONF_BRAVE_API_KEY, DOMAIN, LOGGER

BRAVE_WEB_SEARCH_URL = "https://api.search.brave.com/res/v1/web/search"
DEFAULT_RESULT_COUNT = 5
MAX_RESULT_COUNT = 10
WEB_SEARCH_API_PROMPT = (
    "Use web_search only for current facts from the public web "
    "(news, products, documentation, general knowledge). "
    "Do not use it for Home Assistant device state or home control."
)


def web_search_api_id(entry_id: str) -> str:
    """Stable LLM API id for a DeepSeek config entry's Brave web search."""
    return f"{DOMAIN}_web_search_{entry_id}"


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


class WebSearchAPI(llm.API):
    """LLM API exposing Brave web_search for a DeepSeek config entry."""

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry) -> None:
        """Bind API id/name to the config entry that holds the Brave key."""
        super().__init__(
            hass=hass,
            id=web_search_api_id(entry.entry_id),
            name=f"{entry.title} Web Search (Brave)",
        )
        self._entry = entry

    async def async_get_api_instance(self, llm_context: LLMContext) -> APIInstance:
        """Build an API instance with the current Brave key from entry data."""
        api_key = (self._entry.data.get(CONF_BRAVE_API_KEY) or "").strip()
        if not api_key:
            raise HomeAssistantError(
                "Brave Search API key is missing; reconfigure the DeepSeek integration"
            )
        return APIInstance(
            api=self,
            api_prompt=WEB_SEARCH_API_PROMPT,
            llm_context=llm_context,
            tools=[WebSearchTool(api_key)],
        )


def async_register_web_search_api(
    hass: HomeAssistant, entry: ConfigEntry
) -> None:
    """Register the Brave web search LLM API when a key is configured.

    Call from ``async_setup_entry``. Unregisters automatically on entry unload.
    """
    api_key = (entry.data.get(CONF_BRAVE_API_KEY) or "").strip()
    if not api_key:
        LOGGER.debug(
            "[Debug web_search]: no Brave key on entry %s; skip API registration",
            entry.entry_id,
        )
        return

    LOGGER.debug(
        "[Debug web_search]: registering LLM API id=%s for entry %s",
        web_search_api_id(entry.entry_id),
        entry.entry_id,
    )
    unregister = llm.async_register_api(hass, WebSearchAPI(hass=hass, entry=entry))
    entry.async_on_unload(unregister)
