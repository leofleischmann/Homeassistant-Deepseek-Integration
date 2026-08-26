"""AI Task support for DeepSeek Conversation.

Registers one ``AITaskEntity`` per AI Task subentry, each on its own device.
Uses the shared streaming loop in ``conversation.py`` (``async_handle_chat_log``).
Platform setup is wired from ``__init__.py``.

HA's ``AITaskEntity`` prepares the chat log with a generic system prompt;
``_async_apply_entry_llm_options`` replaces ``content[0]`` with the agent's own
prompt and LLM APIs before the API call.
"""

from __future__ import annotations

from collections.abc import Mapping
from json import JSONDecodeError
import logging
import re
from typing import Any

from homeassistant.components import ai_task, conversation  # pyright: ignore[reportMissingImports]
from homeassistant.config_entries import ConfigSubentry  # pyright: ignore[reportMissingImports]
from homeassistant.const import CONF_LLM_HASS_API  # pyright: ignore[reportMissingImports]
from homeassistant.core import HomeAssistant  # pyright: ignore[reportMissingImports]
from homeassistant.exceptions import HomeAssistantError  # pyright: ignore[reportMissingImports]
from homeassistant.helpers import llm  # pyright: ignore[reportMissingImports]
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback  # pyright: ignore[reportMissingImports]
from homeassistant.util.json import json_loads  # pyright: ignore[reportMissingImports]

from .const import (
    CONF_BASE_URL,
    CONF_PROMPT,
    DEEPSEEK_API_BASE_URL,
    DEFAULT_SYSTEM_PROMPT,
    DOMAIN,
    SUBENTRY_TYPE_AI_TASK,
)
from .conversation import async_handle_chat_log
from .structured_output import structure_schema_for_task
from .types import (
    agent_device_info,
    agent_subentries,
    DeepSeekConfigEntry,
)
from .user_context import EMPTY_SPEAKER_CONTEXT
from .vision import ai_task_entity_features_for_options

_LOGGER = logging.getLogger(__name__)

_JSON_FENCE_RE = re.compile(
    r"^```(?:json)?\s*\n?(.*?)\n?```\s*$",
    re.DOTALL | re.IGNORECASE,
)


def _parse_structured_task_response(text: str) -> Any:
    """Parse JSON from a structured AI Task reply, tolerating markdown fences."""
    stripped = text.strip()
    if match := _JSON_FENCE_RE.match(stripped):
        stripped = match.group(1).strip()
    return json_loads(stripped)


async def _async_apply_entry_llm_options(
    hass: HomeAssistant,
    options: Mapping[str, Any],
    chat_log: conversation.ChatLog,
    task: ai_task.GenDataTask,
) -> None:
    """Apply this config entry's Assist prompt and LLM APIs to the AI Task chat log.

    ``AITaskEntity`` (final) already called ``async_provide_llm_data`` with HA's
    generic default. Calling it again replaces ``content[0]`` only; task
    instructions and attachments in later content entries are preserved.

    An AI Task has no speaker: ``GenDataTask`` carries no ``Context``, and HA's
    own ``AITaskEntity`` passes ``context=None`` too, so the calling user is not
    available here at all. The speaker variables are still defined - as empty
    strings - so one prompt can be shared with Assist without tripping over
    undefined names.
    """
    user_llm_hass_api = (
        task.llm_api
        if task.llm_api is not None
        else options.get(CONF_LLM_HASS_API)
    )
    user_llm_prompt = (options.get(CONF_PROMPT) or "").strip() or DEFAULT_SYSTEM_PROMPT
    user_llm_prompt = EMPTY_SPEAKER_CONTEXT.apply_to_prompt(user_llm_prompt)

    _LOGGER.debug(
        "[Debug ai_task]: applying entry prompt (%d chars) llm_api=%r",
        len(user_llm_prompt),
        user_llm_hass_api,
    )

    try:
        await chat_log.async_provide_llm_data(
            llm_context=llm.LLMContext(
                platform=DOMAIN,
                context=None,
                language=None,
                assistant=DOMAIN,
                device_id=None,
            ),
            user_llm_hass_api=user_llm_hass_api,
            user_llm_prompt=user_llm_prompt,
        )
    except conversation.ConverseError as err:
        raise HomeAssistantError(f"Error preparing context: {err}") from err


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: DeepSeekConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up one AI Task entity per AI Task subentry."""
    for subentry in agent_subentries(config_entry, SUBENTRY_TYPE_AI_TASK):
        async_add_entities(
            [DeepSeekAITaskEntity(hass, config_entry, subentry)],
            config_subentry_id=subentry.subentry_id,
        )


class DeepSeekAITaskEntity(ai_task.AITaskEntity):
    """DeepSeek AI Task entity — exposes generate_data to automations."""

    _attr_has_entity_name = True
    _attr_name = None

    def __init__(
        self,
        hass: HomeAssistant,
        entry: DeepSeekConfigEntry,
        subentry: ConfigSubentry,
    ) -> None:
        """Initialise one AI Task agent from its subentry."""
        self.hass = hass
        self.entry = entry
        self.subentry = subentry
        self._attr_unique_id = subentry.subentry_id
        self._attr_device_info = agent_device_info(entry, subentry)
        self._attr_supported_features = ai_task_entity_features_for_options(
            subentry.data,
            base_url=entry.data.get(CONF_BASE_URL, DEEPSEEK_API_BASE_URL),
        )

    async def _async_generate_data(
        self,
        task: ai_task.GenDataTask,
        chat_log: conversation.ChatLog,
    ) -> ai_task.GenDataTaskResult:
        """Run a generate-data task; return text or parsed JSON."""
        _LOGGER.debug(
            "[Debug ai_task]: task_name=%r structured=%s",
            task.name,
            task.structure is not None,
        )

        options = self.subentry.data
        await _async_apply_entry_llm_options(self.hass, options, chat_log, task)

        response_schema = None
        if task.structure is not None:
            response_schema = structure_schema_for_task(chat_log, task.structure)

        await async_handle_chat_log(
            self.hass,
            self.entry,
            chat_log,
            options=options,
            agent_id=self.entity_id,
            force_json=task.structure is not None,
            response_schema=response_schema,
            usage_source="ai_task",
        )

        if not chat_log.content:
            raise HomeAssistantError("DeepSeek returned no assistant response")

        if not isinstance(chat_log.content[-1], conversation.AssistantContent):
            raise HomeAssistantError(
                "Last content in chat log is not an AssistantContent"
            )

        text = chat_log.content[-1].content or ""
        if not text.strip() and task.structure is None:
            thinking = getattr(chat_log.content[-1], "thinking_content", None)
            if isinstance(thinking, str) and thinking.strip():
                _LOGGER.debug(
                    "[Debug ai_task]: using thinking_content as plain-text fallback"
                )
                text = thinking.strip()

        if task.structure is None:
            return ai_task.GenDataTaskResult(
                conversation_id=chat_log.conversation_id,
                data=text,
            )

        try:
            data = _parse_structured_task_response(text)
        except JSONDecodeError as err:
            _LOGGER.error(
                "[Debug ai_task]: failed to parse JSON response: %s. Response: %s",
                err,
                text,
            )
            raise HomeAssistantError(
                "DeepSeek returned a non-JSON response for a structured task"
            ) from err
        except (TypeError, ValueError) as err:
            _LOGGER.error(
                "[Debug ai_task]: structured response is not valid JSON: %s. Response: %s",
                err,
                text,
            )
            raise HomeAssistantError(
                "DeepSeek returned a non-JSON response for a structured task"
            ) from err

        return ai_task.GenDataTaskResult(
            conversation_id=chat_log.conversation_id,
            data=data,
        )
