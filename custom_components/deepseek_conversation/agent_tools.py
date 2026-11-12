"""Which tools one agent exposes, and how they reach Home Assistant.

An agent's tools come from two places: the Home Assistant LLM APIs its owner
selected, and whatever this integration adds itself - today only web search
from the ``web`` package. Bringing them together is this module's whole job,
and it is kept apart from the tools themselves so that adding a second one is a
change in one place.

The result is handed to ``ChatLog.async_provide_llm_data``, which accepts an
``llm.API`` object as readily as a registered id. That is what keeps a tool
private: ``llm.async_register_api`` writes into a registry every conversation
integration reads, so anything registered there turns up in Anthropic's,
Gemini's and OpenAI's agent settings too (#38).

Nothing is composed when the agent has no tools of ours - the selection is
passed through untouched, so the common case stays on Home Assistant's own path.

Affects / influenced by:
- ``conversation.py`` and ``ai_task.py`` call ``agent_llm_api`` once per turn.
- ``web/selection.py`` says which web tools the agent gets (``web_tools_plan``);
  ``web/factory.py`` builds the provider and ``web/search_tool.py`` is the tool.
- Searches are counted on the entry's ``UsageTracker`` (usage_metrics.py), the
  same one the token sensors read.
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
from .options import max_tool_result_chars_from_options
from .web.factory import build_search_provider
from .web.search_tool import WEB_SEARCH_API_PROMPT, WebSearchTool
from .web.selection import web_tools_plan


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
        options: Mapping[str, Any],
        selected: str | list[str] | llm.API | None,
    ) -> None:
        """Bind the entry that holds the credentials and the agent's own choice."""
        super().__init__(
            hass=hass,
            id=f"{DOMAIN}_{entry.entry_id}",
            name=f"{entry.title} tools",
        )
        self._entry = entry
        self._options = options
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

        # Worked out again rather than assumed. ``agent_llm_api`` is the only
        # caller that screens on the plan today, and a provider built without
        # its credential would fail at the search API with an opaque 401
        # instead of saying so here.
        plan = web_tools_plan(self._entry.data, self._options)
        if plan.search_provider is not None:
            provider = build_search_provider(
                self.hass, self._entry.data, plan.search_provider
            )
            tools.append(
                WebSearchTool(
                    provider,
                    on_results=self._entry.runtime_data.usage.record_web_search,
                    max_result_chars=max_tool_result_chars_from_options(
                        self._options
                    ),
                )
            )
            prompts.append(WEB_SEARCH_API_PROMPT)
        else:
            LOGGER.warning(
                "[Debug agent_tools]: web search is switched on for an agent of "
                "%s, but the entry can no longer provide it; answering without it",
                self._entry.title,
            )

        LOGGER.debug(
            "[Debug agent_tools]: web_search provider=%s agent tools=%s",
            plan.search_provider,
            [tool.name for tool in tools],
        )
        return APIInstance(
            api=self,
            api_prompt="\n\n".join(prompts),
            llm_context=llm_context,
            tools=tools,
            # Core's serializer when the agent selected no Home Assistant API,
            # because without one there is nobody to carry it and a converter
            # left to itself renders a validator it does not recognise untyped
            # - our own tools use plain types today, so nothing changes for
            # them, but a tool added later would quietly lose its parameter
            # types. ``structured_output.py`` falls back the same way.
            custom_serializer=(
                base.custom_serializer
                if base is not None
                else llm.selector_serializer
            ),
        )


def agent_llm_api(
    hass: HomeAssistant,
    entry: ConfigEntry,
    options: Mapping[str, Any],
    selected: str | list[str] | llm.API | None,
) -> str | list[str] | llm.API | None:
    """What to hand ``ChatLog.async_provide_llm_data`` for this agent.

    Without web tools that is the agent's own selection, untouched, so the
    common case goes through Home Assistant's own path. With them, the
    selection is wrapped in an API this integration keeps to itself.
    """
    if not web_tools_plan(entry.data, options):
        return selected
    return AgentToolsAPI(hass=hass, entry=entry, options=options, selected=selected)
