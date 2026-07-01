"""Token usage sensors for DeepSeek Conversation."""

from __future__ import annotations

from homeassistant.components.sensor import (  # pyright: ignore[reportMissingImports]
    RestoreSensor,
    SensorEntity,
    SensorStateClass,
)
from homeassistant.core import HomeAssistant  # pyright: ignore[reportMissingImports]
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback  # pyright: ignore[reportMissingImports]
from homeassistant.helpers import device_registry as dr  # pyright: ignore[reportMissingImports]

from .const import DOMAIN
from .types import DeepSeekConfigEntry
from .usage_metrics import CompletionUsage, UsageTracker, connection_changed_since_last_setup


class DeepSeekUsageCounterSensor(RestoreSensor, SensorEntity):
    """Cumulative counter (persists across restarts)."""

    _attr_has_entity_name = True
    _attr_state_class = SensorStateClass.TOTAL_INCREASING

    def __init__(
        self,
        entry: DeepSeekConfigEntry,
        translation_key: str,
        unique_suffix: str,
        *,
        unit: str,
        icon: str,
        *,
        reset_on_add: bool = False,
    ) -> None:
        self._entry = entry
        self._reset_on_add = reset_on_add
        self._attr_translation_key = translation_key
        self._attr_unique_id = f"{entry.entry_id}_{unique_suffix}"
        self._attr_native_unit_of_measurement = unit
        self._attr_icon = icon
        self._attr_device_info = dr.DeviceInfo(
            identifiers={(DOMAIN, entry.entry_id)},
        )

    async def async_added_to_hass(self) -> None:
        await super().async_added_to_hass()
        if self._reset_on_add:
            self._attr_native_value = 0
            return
        if (last_state := await self.async_get_last_state()) is not None:
            try:
                self._attr_native_value = int(float(last_state.state))
            except (TypeError, ValueError):
                self._attr_native_value = 0
        elif self.native_value is None:
            self._attr_native_value = 0

    def increment(self, amount: int) -> None:
        if amount <= 0:
            return
        self._attr_native_value = int(self.native_value or 0) + amount
        self.async_write_ha_state()

    def reset_to_zero(self) -> None:
        self._attr_native_value = 0
        self.async_write_ha_state()


class DeepSeekSnapshotSensor(SensorEntity):
    """Value from the most recent API call (not restored across restarts)."""

    _attr_has_entity_name = True
    _attr_native_unit_of_measurement = "tokens"
    _attr_icon = "mdi:history"

    def __init__(
        self,
        entry: DeepSeekConfigEntry,
        translation_key: str,
        unique_suffix: str,
        *,
        reset_on_add: bool = False,
    ) -> None:
        self._entry = entry
        self._reset_on_add = reset_on_add
        self._attr_translation_key = translation_key
        self._attr_unique_id = f"{entry.entry_id}_{unique_suffix}"
        self._attr_device_info = dr.DeviceInfo(
            identifiers={(DOMAIN, entry.entry_id)},
        )
        self._attr_native_value = 0

    async def async_added_to_hass(self) -> None:
        await super().async_added_to_hass()
        if self._reset_on_add:
            self.reset_to_zero()

    def set_value(self, value: int) -> None:
        self._attr_native_value = value
        self.async_write_ha_state()

    def reset_to_zero(self) -> None:
        self._attr_native_value = 0
        self.async_write_ha_state()


class DeepSeekLastRequestSensor(SensorEntity):
    """Tokens used by the most recent API call."""

    _attr_has_entity_name = True
    _attr_native_unit_of_measurement = "tokens"
    _attr_icon = "mdi:history"
    _attr_translation_key = "last_request_tokens"

    def __init__(self, entry: DeepSeekConfigEntry, *, reset_on_add: bool = False) -> None:
        self._entry = entry
        self._reset_on_add = reset_on_add
        self._attr_unique_id = f"{entry.entry_id}_last_request_tokens"
        self._attr_device_info = dr.DeviceInfo(
            identifiers={(DOMAIN, entry.entry_id)},
        )
        self._attr_native_value = 0

    async def async_added_to_hass(self) -> None:
        await super().async_added_to_hass()
        if self._reset_on_add:
            self.reset_to_zero()

    def set_usage(
        self, usage: CompletionUsage, *, source: str, request_count: int
    ) -> None:
        total = usage.total_tokens or usage.prompt_tokens + usage.completion_tokens
        self._attr_native_value = total
        self._attr_extra_state_attributes = {
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "reasoning_tokens": usage.reasoning_tokens,
            "source": source,
            "request_count": request_count,
        }
        self.async_write_ha_state()

    def reset_to_zero(self) -> None:
        self._attr_native_value = 0
        self._attr_extra_state_attributes = {}
        self.async_write_ha_state()


async def async_setup_entry(
    hass: HomeAssistant,
    entry: DeepSeekConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up token usage sensors."""
    runtime = entry.runtime_data
    tracker: UsageTracker = runtime.usage

    connection_changed = connection_changed_since_last_setup(
        hass, entry.entry_id, entry.data
    )

    prompt = DeepSeekUsageCounterSensor(
        entry,
        "prompt_tokens",
        "prompt_tokens",
        unit="tokens",
        icon="mdi:counter",
        reset_on_add=connection_changed,
    )
    completion = DeepSeekUsageCounterSensor(
        entry,
        "completion_tokens",
        "completion_tokens",
        unit="tokens",
        icon="mdi:counter",
        reset_on_add=connection_changed,
    )
    total = DeepSeekUsageCounterSensor(
        entry,
        "total_tokens",
        "total_tokens",
        unit="tokens",
        icon="mdi:counter",
        reset_on_add=connection_changed,
    )
    reasoning = DeepSeekUsageCounterSensor(
        entry,
        "reasoning_tokens",
        "reasoning_tokens",
        unit="tokens",
        icon="mdi:brain",
        reset_on_add=connection_changed,
    )
    api_requests = DeepSeekUsageCounterSensor(
        entry,
        "api_requests",
        "api_requests",
        unit="requests",
        icon="mdi:api",
        reset_on_add=connection_changed,
    )
    last_request = DeepSeekLastRequestSensor(entry, reset_on_add=connection_changed)
    last_request_prompt = DeepSeekSnapshotSensor(
        entry,
        "last_request_prompt_tokens",
        "last_request_prompt_tokens",
        reset_on_add=connection_changed,
    )
    last_request_completion = DeepSeekSnapshotSensor(
        entry,
        "last_request_completion_tokens",
        "last_request_completion_tokens",
        reset_on_add=connection_changed,
    )

    tracker.bind_sensors(
        prompt=prompt,
        completion=completion,
        total=total,
        reasoning=reasoning,
        api_requests=api_requests,
        last_request=last_request,
        last_request_prompt=last_request_prompt,
        last_request_completion=last_request_completion,
    )

    async_add_entities(
        [
            prompt,
            completion,
            total,
            reasoning,
            api_requests,
            last_request,
            last_request_prompt,
            last_request_completion,
        ]
    )

    if connection_changed:
        tracker.request_count = 0
        tracker.last_usage = None
        tracker.last_source = None
