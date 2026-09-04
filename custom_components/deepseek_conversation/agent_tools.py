"""Which tools one agent exposes, and how they reach Home Assistant.

An agent's tools come from two places: the Home Assistant LLM APIs its owner
selected, and whatever this integration adds itself - today only the Brave web
search tool in ``web_search.py``. Bringing them together is this module's whole
job, and it is kept apart from the tool itself so that adding a second one is a
change in one place.

The result is handed to ``ChatLog.async_provide_llm_data``, which accepts an
``llm.API`` object as readily as a registered id. That is what keeps a tool
private: ``llm.async_register_api`` writes into a registry every conversation
integration reads, so anything registered there turns up in Anthropic's,
Gemini's and OpenAI's agent settings too (#38).

Nothing is composed when the agent has no tools of ours - the selection is
passed through untouched, so the common case stays on Home Assistant's own path.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from homeassistant.config_entries import ConfigEntry  # pyright: ignore[reportMissingImports]
from homeassistant.core import HomeAssistant  # pyright: ignore[reportMissingImports]
from homeassistant.helpers import llm  # pyright: ignore[reportMissingImports]
from homeassistant.helpers.llm import (  # pyright: ignore[reportMissingImports]
    APIInstance,
    LLMContext,
)

from .const import DOMAIN, LOGGER
from .web_search import (
    brave_api_key,
    web_search_enabled,
    WEB_SEARCH_API_PROMPT,
    WebSearchTool,
)


class AgentToolsAPI(llm.API):
    """One agent's tools: what Home Assistant offers it, plus web search.

    Composing rather than merging keeps the tool names Home Assistant chose -
    ``llm.async_get_api`` already namespaces them when several APIs are
    selected, and doing it a second time would rename every tool the moment
    web search was switched on.
    """

    def __init__(
        self,
        hass: HomeAssistant,
        entry: ConfigEntry,
        selected: str | list[str] | llm.API | None,
    ) -> None:
        """Bind the entry that holds the Brave key and the agent's own choice."""
        super().__init__(
            hass=hass,
            id=f"{DOMAIN}_{entry.entry_id}",
            name=f"{entry.title} tools",
        )
        self._entry = entry
        self._selected = selected

    async def _async_selected_instance(
        self, llm_context: LLMContext
    ) -> APIInstance | None:
        """The Home Assistant APIs this agent selected, or None for none.

        Deliberately the same three cases ``async_provide_llm_data`` itself
        handles, in the same order and with the same failures: an id that no
        longer resolves raises here exactly as it would without us. Switching
        web search on adds a tool and changes nothing else - an agent that
        would have failed still fails, and one that would have worked works,
        so a broken selection cannot look like two different bugs.
        """
        selected = self._selected
        if not selected:
            return None
        if isinstance(selected, llm.API):
            return await selected.async_get_api_instance(llm_context)
        return await llm.async_get_api(self.hass, selected, llm_context)

    async def async_get_api_instance(self, llm_context: LLMContext) -> APIInstance:
        """Build the agent's tool set for one conversation turn."""
        base = await self._async_selected_instance(llm_context)
        tools: list[llm.Tool] = list(base.tools) if base is not None else []
        prompts: list[str] = []
        if base is not None and base.api_prompt:
            prompts.append(base.api_prompt)

        # Re-checked rather than assumed. ``agent_llm_api`` is the only caller
        # that screens on the key today, and a tool built with an empty token
        # would fail at Brave with an opaque 401 instead of saying so here.
        if api_key := brave_api_key(self._entry):
            tools.append(WebSearchTool(api_key))
            prompts.append(WEB_SEARCH_API_PROMPT)
        else:
            LOGGER.warning(
                "[Debug agent_tools]: web search is switched on for an agent of "
                "%s, but the entry has no Brave key; answering without it",
                self._entry.title,
            )

        LOGGER.debug(
            "[Debug agent_tools]: agent tools=%s", [tool.name for tool in tools]
        )
        return APIInstance(
            api=self,
            api_prompt="\n\n".join(prompts),
            llm_context=llm_context,
            tools=tools,
            custom_serializer=base.custom_serializer if base is not None else None,
        )


def agent_llm_api(
    hass: HomeAssistant,
    entry: ConfigEntry,
    options: Mapping[str, Any],
    selected: str | list[str] | llm.API | None,
) -> str | list[str] | llm.API | None:
    """What to hand ``ChatLog.async_provide_llm_data`` for this agent.

    Without web search that is the agent's own selection, untouched, so the
    common case goes through Home Assistant's own path. With it, the selection
    is wrapped in an API this integration keeps to itself.
    """
    if not web_search_enabled(entry, options):
        return selected
    return AgentToolsAPI(hass=hass, entry=entry, selected=selected)
