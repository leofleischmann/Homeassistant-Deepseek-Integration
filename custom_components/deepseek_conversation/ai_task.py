"""AI Task support for DeepSeek Conversation.

Registers an ``AITaskEntity`` per config entry on the same device as the
conversation agent. Uses the shared streaming loop in ``conversation.py``
(``async_handle_chat_log``). Platform setup is wired from ``__init__.py``.
"""

from __future__ import annotations

from json import JSONDecodeError
import logging

from homeassistant.components import ai_task, conversation  # pyright: ignore[reportMissingImports]
from homeassistant.core import HomeAssistant  # pyright: ignore[reportMissingImports]
from homeassistant.exceptions import HomeAssistantError  # pyright: ignore[reportMissingImports]
from homeassistant.helpers import device_registry as dr  # pyright: ignore[reportMissingImports]
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback  # pyright: ignore[reportMissingImports]
from homeassistant.util.json import json_loads  # pyright: ignore[reportMissingImports]

from .const import CONF_CHAT_MODEL, DOMAIN, RECOMMENDED_CHAT_MODEL
from .conversation import async_handle_chat_log
from .types import DeepSeekConfigEntry
from .vision import ai_task_entity_features_for_options

_LOGGER = logging.getLogger(__name__)


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: DeepSeekConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up the DeepSeek AI Task entity."""
    async_add_entities([DeepSeekAITaskEntity(hass, config_entry)])


class DeepSeekAITaskEntity(ai_task.AITaskEntity):
    """DeepSeek AI Task entity — exposes generate_data to automations."""

    _attr_has_entity_name = True
    _attr_name = None

    def __init__(self, hass: HomeAssistant, entry: DeepSeekConfigEntry) -> None:
        """Initialise the entity, sharing the device with the conversation entity."""
        self.hass = hass
        self.entry = entry
        self._attr_unique_id = f"{entry.entry_id}_ai_task"
        self._attr_device_info = dr.DeviceInfo(
            identifiers={(DOMAIN, entry.entry_id)},
            name=entry.title,
            manufacturer="DeepSeek",
            model=entry.options.get(CONF_CHAT_MODEL, RECOMMENDED_CHAT_MODEL),
            entry_type=dr.DeviceEntryType.SERVICE,
        )
        self._sync_features_from_entry(entry)

    def _sync_features_from_entry(self, entry: DeepSeekConfigEntry) -> None:
        """Refresh supported features when entry options change."""
        self._attr_supported_features = ai_task_entity_features_for_options(
            entry.options
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
        await async_handle_chat_log(
            self.hass,
            self.entry,
            chat_log,
            agent_id=self.entity_id,
            force_json=task.structure is not None,
            usage_source="ai_task",
        )

        if not isinstance(chat_log.content[-1], conversation.AssistantContent):
            raise HomeAssistantError(
                "Last content in chat log is not an AssistantContent"
            )

        text = chat_log.content[-1].content or ""

        if task.structure is None:
            return ai_task.GenDataTaskResult(
                conversation_id=chat_log.conversation_id,
                data=text,
            )

        try:
            data = json_loads(text)
        except JSONDecodeError as err:
            _LOGGER.error(
                "[Debug ai_task]: failed to parse JSON response: %s. Response: %s",
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
