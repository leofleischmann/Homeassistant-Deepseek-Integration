"""The ``web_search`` tool the model calls.

One tool whichever provider answers: it validates the call, asks the provider,
and puts the results into the shape the model reads. Everything about a
particular search API stays behind ``SearchProvider``.

Affects / influenced by:
- ``agent_tools.py`` is what puts the tool in front of an agent. It is
  deliberately never registered with ``llm.async_register_api``: that registry
  is global, so registering it put a web search entry into every *other*
  conversation integration's settings as well (#38).
- The provider comes from ``factory.py``; what a result looks like to the model
  is ``SearchResult.as_tool_dict`` in ``types.py``.
- ``llm_compat.py`` decides the shape the answer goes back in.
- A ``WebToolError`` becomes a ``HomeAssistantError`` here; core hands its text
  to the model as the result of the call.
"""

from __future__ import annotations

from typing import Any

import voluptuous as vol  # pyright: ignore[reportMissingImports]

from homeassistant.core import HomeAssistant  # pyright: ignore[reportMissingImports]
from homeassistant.exceptions import HomeAssistantError  # pyright: ignore[reportMissingImports]
from homeassistant.helpers import llm  # pyright: ignore[reportMissingImports]
from homeassistant.helpers.llm import (  # pyright: ignore[reportMissingImports]
    LLMContext,
    ToolInput,
)

from ..const import DOMAIN, LOGGER
from ..llm_compat import tool_result
from .types import SearchProvider, WebToolError

DEFAULT_RESULT_COUNT = 5
MAX_RESULT_COUNT = 10
WEB_SEARCH_API_PROMPT = (
    "Use web_search only for current facts from the public web "
    "(news, products, documentation, general knowledge). "
    "Do not use it for Home Assistant device state or home control."
)


class WebSearchTool(llm.Tool):
    """Search the public web through a provider and return compact results."""

    name = "web_search"
    parameters = vol.Schema(
        {
            vol.Required("query"): str,
            vol.Optional("count", default=DEFAULT_RESULT_COUNT): vol.All(
                vol.Coerce(int), vol.Range(min=1, max=MAX_RESULT_COUNT)
            ),
        }
    )
    #: Which integration provides the tool. Home Assistant 2026.10 reports every
    #: tool that leaves this unset and refuses it from the release named in
    #: ``llm.TOOL_INTEGRATION_BREAKS_IN_HA_VERSION`` (#44). Core derives the same
    #: domain from the module path as a stopgap; saying it here is what survives
    #: that stopgap being removed. Harmless on cores that have no such attribute.
    integration = DOMAIN

    def __init__(self, provider: SearchProvider) -> None:
        """Bind the provider this agent searches with."""
        self._provider = provider
        self.description = (
            f"Search the public web via {provider.label}. "
            "Use for up-to-date information that is not available from Home Assistant tools."
        )

    async def async_call(
        self,
        hass: HomeAssistant,
        tool_input: ToolInput,
        llm_context: LLMContext,
    ) -> Any:
        """Run one search and return what the provider found."""
        query = str(tool_input.tool_args["query"]).strip()
        if not query:
            raise HomeAssistantError("web_search requires a non-empty query")

        count = int(tool_input.tool_args.get("count", DEFAULT_RESULT_COUNT))
        count = max(1, min(count, MAX_RESULT_COUNT))

        LOGGER.debug(
            "[Debug web_search]: provider=%s query=%r count=%s platform=%s",
            self._provider.id,
            query,
            count,
            llm_context.platform,
        )

        try:
            results = await self._provider.search(query, count)
        except WebToolError as err:
            LOGGER.error(
                "[Debug web_search]: %s failed (%s): %s",
                self._provider.id,
                err.kind,
                err,
            )
            raise HomeAssistantError(str(err)) from err

        LOGGER.debug(
            "[Debug web_search]: returning %s result(s) for query=%r",
            len(results),
            query,
        )
        return tool_result(
            {"query": query, "results": [result.as_tool_dict() for result in results]}
        )
