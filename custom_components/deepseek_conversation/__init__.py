"""The DeepSeek Conversation integration.

One config entry holds the credentials for an API key; every agent is a
subentry of it. This module is the entry lifecycle and nothing else - the work
it used to do alongside that now lives next door:

* ``client.py`` - building the API client and proving the credentials work
* ``migration.py`` - bringing a stored entry up to the current layout
* ``services.py`` - the ``generate_content`` and ``run_debug`` actions
"""

from __future__ import annotations

from homeassistant.config_entries import ConfigEntry  # pyright: ignore[reportMissingImports]
from homeassistant.const import Platform  # pyright: ignore[reportMissingImports]
from homeassistant.core import HomeAssistant  # pyright: ignore[reportMissingImports]
from homeassistant.helpers import (  # pyright: ignore[reportMissingImports]
    config_validation as cv,
    issue_registry as ir,
)
from homeassistant.helpers.typing import ConfigType  # pyright: ignore[reportMissingImports]

from .client import async_create_client
from .const import DOMAIN
from .migration import (
    async_migrate_entry,  # noqa: F401  # Home Assistant looks this up on the component
    async_migrate_legacy_model_option,
    legacy_model_issue_id,
)
from .services import async_setup_services
from .types import DeepSeekConfigEntry, DeepSeekRuntimeData
from .usage_metrics import UsageTracker
from .web_search import async_register_web_search_api

PLATFORMS = (Platform.AI_TASK, Platform.CONVERSATION, Platform.SENSOR, Platform.BUTTON)
CONFIG_SCHEMA = cv.config_entry_only_config_schema(DOMAIN)


async def async_setup(hass: HomeAssistant, config: ConfigType) -> bool:
    """Set up DeepSeek Conversation."""
    async_setup_services(hass)
    return True


async def _async_reload_entry(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Reload when the entry or one of its agents was edited.

    Agent settings live in subentries, and an entity holds the subentry object
    it was built from. Reloading is what hands the entities the new one; there
    is nothing cheap to refresh in place any more.
    """
    await hass.config_entries.async_reload(entry.entry_id)


async def async_setup_entry(hass: HomeAssistant, entry: DeepSeekConfigEntry) -> bool:
    """Set up DeepSeek Conversation from a config entry."""
    # Before the platforms build entities from the agents' settings.
    async_migrate_legacy_model_option(hass, entry)

    client = await async_create_client(hass, entry)
    entry.runtime_data = DeepSeekRuntimeData(client=client, usage=UsageTracker())

    # Optional Brave web_search LLM API (see web_search.py); only when key in entry.data.
    async_register_web_search_api(hass, entry)

    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)

    # Registered last, so the subentry rewrite above cannot trigger a reload.
    entry.async_on_unload(entry.add_update_listener(_async_reload_entry))

    return True


async def async_remove_entry(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Withdraw this entry's repair issues when it is deleted."""
    ir.async_delete_issue(hass, DOMAIN, legacy_model_issue_id(entry))


async def async_unload_entry(hass: HomeAssistant, entry: DeepSeekConfigEntry) -> bool:
    """Unload DeepSeek platforms.

    The OpenAI client is built on Home Assistant's shared httpx client (see
    ``client.async_http_client``). That connection pool is owned by HA and must
    not be closed here — doing so only triggers a framework warning without
    releasing anything — so unload just tears down the platforms.
    """
    return await hass.config_entries.async_unload_platforms(entry, PLATFORMS)
