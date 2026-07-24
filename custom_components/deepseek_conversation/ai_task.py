"""AI Task support for DeepSeek Conversation.

Registers an ``AITaskEntity`` per config entry on the same device as the
conversation agent. Uses the shared streaming loop in ``conversation.py``
(``async_handle_chat_log``). Platform setup is wired from ``__init__.py``.
"""

from __future__ import annotations

from json import JSONDecodeError
import logging
import re
from typing import Any

from homeassistant.components import ai_task, conversation  # pyright: ignore[reportMissingImports]
from homeassistant.config_entries import ConfigFlow  # pyright: ignore[reportMissingImports]
from homeassistant.core import HomeAssistant  # pyright: ignore[reportMissingImports]
from homeassistant.exceptions import HomeAssistantError  # pyright: ignore[reportMissingImports]
from homeassistant.helpers import device_registry as dr  # pyright: ignore[reportMissingImports]
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback  # pyright: ignore[reportMissingImports]
from homeassistant.util.json import json_loads  # pyright: ignore[reportMissingImports]

from .const import DOMAIN
from .conversation import async_handle_chat_log
from .types import DeepSeekConfigEntry
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
        )
        self._sync_features_from_entry(entry)

    def _sync_features_from_entry(self, entry: DeepSeekConfigEntry) -> None:
        """Refresh supported features when entry options change."""
        self._attr_supported_features = ai_task_entity_features_for_options(
            entry.options
        )

    async def async_added_to_hass(self) -> None:
        """Register option updates (vision flag -> SUPPORT_ATTACHMENTS)."""
        await super().async_added_to_hass()
        self.entry.async_on_unload(
            self.entry.add_update_listener(self._async_entry_update_listener)
        )

    async def _async_entry_update_listener(
        self, hass: HomeAssistant, entry: DeepSeekConfigEntry
    ) -> None:
        """Apply option changes without a full config-entry reload."""
        data_changed = dict(entry.data) != dict(self.entry.data)
        self.entry = entry
        if data_changed and hasattr(ConfigFlow, "async_update_and_abort"):
            return
        self._sync_features_from_entry(entry)
        self.async_write_ha_state()

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
