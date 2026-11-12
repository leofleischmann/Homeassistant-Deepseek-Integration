"""Bringing a stored config entry up to the layout this version expects.

Two kinds of change land here. ``async_migrate_entry`` runs once per version
bump and rewrites what the entry looks like - most recently turning a single
agent's options into the subentries that let one API key carry several agents.
``async_migrate_legacy_model_option`` instead runs on **every** setup, because
an agent can arrive on a retired model id long after a versioned migration
would have seen it: from a restored backup, or simply because the model field
accepts free text.
"""

from __future__ import annotations

from types import MappingProxyType

from homeassistant.config_entries import (  # pyright: ignore[reportMissingImports]
    ConfigEntry,
    ConfigSubentry,
)
from homeassistant.const import CONF_LLM_HASS_API  # pyright: ignore[reportMissingImports]
from homeassistant.core import callback, HomeAssistant  # pyright: ignore[reportMissingImports]
from homeassistant.helpers import (  # pyright: ignore[reportMissingImports]
    entity_registry as er,
    issue_registry as ir,
)

from .const import (
    CONF_BASE_URL,
    CONF_CHAT_MODEL,
    CONF_RECOMMENDED,
    CONF_WEB_SEARCH,
    DEFAULT_AI_TASK_NAME,
    DEFAULT_CONVERSATION_NAME,
    DOMAIN,
    LOGGER,
    SUBENTRY_TYPE_AI_TASK,
    SUBENTRY_TYPE_CONVERSATION,
    SUBENTRY_TYPES,
)
from .models import (
    is_rerouted_chat_model,
    legacy_chat_model_retired_on,
    migrate_legacy_chat_model,
)
from .options import (
    adopt_strip_markdown_default,
    ai_task_options_from,
    fold_context_switch,
    move_web_search_selection,
    web_search_switch_to_provider,
)


async def async_migrate_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Bring a config entry up to the current layout."""
    if entry.version < 2:
        # Wrap a legacy string CONF_LLM_HASS_API in a list.
        options = {**entry.options}
        legacy = options.get(CONF_LLM_HASS_API)
        if isinstance(legacy, str):
            options[CONF_LLM_HASS_API] = [legacy] if legacy != "none" else []
        hass.config_entries.async_update_entry(entry, options=options, version=2)

    if entry.version < 3:
        _async_migrate_options_to_subentries(hass, entry)

    if entry.version < 4:
        _async_migrate_web_search_api_to_option(hass, entry)
        hass.config_entries.async_update_entry(entry, version=4)

    if entry.version < 5:
        _async_migrate_web_search_switch_to_provider(hass, entry)
        hass.config_entries.async_update_entry(entry, version=5)

    return True


@callback
def _async_migrate_web_search_api_to_option(
    hass: HomeAssistant, entry: ConfigEntry
) -> None:
    """Turn a selected Brave web search API into the agent's own setting.

    Web search used to be a registered LLM API, chosen in the same picker as
    Assist. That registry is global, so it also appeared in every other
    conversation integration's settings (#38); the tool is private to this
    integration's agents now and has its own switch. An agent that had picked
    the API keeps web search - the choice moves, the behaviour does not.
    """
    for subentry in list(entry.subentries.values()):
        if subentry.subentry_type not in SUBENTRY_TYPES:
            continue
        moved = move_web_search_selection(subentry.data)
        if moved is None:
            continue
        hass.config_entries.async_update_subentry(entry, subentry, data=moved)
        LOGGER.info(
            "Web search is now a setting of the agent %s rather than a Home "
            "Assistant API, so it is no longer offered to other integrations",
            subentry.title,
        )


@callback
def _async_migrate_web_search_switch_to_provider(
    hass: HomeAssistant, entry: ConfigEntry
) -> None:
    """Make every agent name its web search provider instead of a switch.

    The switch only ever meant Brave, so an agent that had it on keeps
    searching with Brave; nothing else about it changes. See
    ``options.web_search_switch_to_provider``.
    """
    for subentry in list(entry.subentries.values()):
        if subentry.subentry_type not in SUBENTRY_TYPES:
            continue
        moved = web_search_switch_to_provider(subentry.data)
        if moved is None:
            continue
        hass.config_entries.async_update_subentry(entry, subentry, data=moved)
        LOGGER.debug(
            "[Debug migration]: web search setting of agent %s is now %r",
            subentry.title,
            moved.get(CONF_WEB_SEARCH),
        )


@callback
def _async_adopt_entity(
    registry: er.EntityRegistry,
    domain: str,
    old_unique_id: str,
    subentry: ConfigSubentry,
) -> None:
    """Hand an existing entity over to the subentry that now configures it.

    Updating the registry entry rather than letting the platform create a new
    one is what keeps the entity id: automations and voice pipelines point at
    ``conversation.deepseek``, and that must survive the move. The device link
    is cleared because the old device stays behind with the usage sensors; the
    platform attaches the entity to its agent's own device on the next setup.
    """
    entity_id = registry.async_get_entity_id(domain, DOMAIN, old_unique_id)
    if entity_id is None:
        return
    registry.async_update_entity(
        entity_id,
        config_subentry_id=subentry.subentry_id,
        device_id=None,
        new_unique_id=subentry.subentry_id,
    )
    LOGGER.debug(
        "[Debug migration]: moved %s to subentry %s", entity_id, subentry.subentry_id
    )


@callback
def _async_migrate_options_to_subentries(
    hass: HomeAssistant, entry: ConfigEntry
) -> None:
    """Turn one entry's options into a conversation agent and an AI Task agent.

    Before this, an entry was a single agent and its settings lived in
    ``entry.options``. They now belong to subentries, so one API key can carry
    several agents. The existing settings are copied verbatim into both, which
    is what keeps this upgrade invisible: the conversation agent answers exactly
    as before, and so does the AI Task entity.
    """
    shared = adopt_strip_markdown_default(fold_context_switch(dict(entry.options)))
    ai_task_options = ai_task_options_from(shared)
    # The settings were explicit, so neither agent starts on the recommended
    # defaults - the reconfigure flow has to show the values that were in use.
    shared[CONF_RECOMMENDED] = False
    ai_task_options[CONF_RECOMMENDED] = False

    conversation_subentry = ConfigSubentry(
        data=MappingProxyType(shared),
        subentry_type=SUBENTRY_TYPE_CONVERSATION,
        title=DEFAULT_CONVERSATION_NAME,
        unique_id=None,
    )
    ai_task_subentry = ConfigSubentry(
        data=MappingProxyType(ai_task_options),
        subentry_type=SUBENTRY_TYPE_AI_TASK,
        title=DEFAULT_AI_TASK_NAME,
        unique_id=None,
    )
    hass.config_entries.async_add_subentry(entry, conversation_subentry)
    hass.config_entries.async_add_subentry(entry, ai_task_subentry)

    registry = er.async_get(hass)
    _async_adopt_entity(registry, "conversation", entry.entry_id, conversation_subentry)
    _async_adopt_entity(
        registry, "ai_task", f"{entry.entry_id}_ai_task", ai_task_subentry
    )

    hass.config_entries.async_update_entry(entry, options={}, version=3)
    LOGGER.info(
        "Migrated %s to per-agent configuration: the settings are now on the "
        "%s agent, and %s carries the same ones",
        entry.title,
        DEFAULT_CONVERSATION_NAME,
        DEFAULT_AI_TASK_NAME,
    )


def legacy_model_issue_id(entry: ConfigEntry) -> str:
    """Repair issue id for one entry left on a retired model."""
    return f"legacy_chat_model_{entry.entry_id}"


@callback
def async_migrate_legacy_model_option(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Move agents off a retired model id, and say so in Repairs where it matters.

    ``deepseek-chat`` / ``deepseek-reasoner`` are no longer served by the
    official API, so an agent still pointing at one fails every single request
    with no hint about why. Rewriting the setting keeps the agent working; the
    repair issue is what tells the user their model choice changed.

    The V4 Flash ids are moved too, but quietly: DeepSeek already answers them
    with V4.1 Flash, the model they move to, so nothing the agent does changes
    and a notice would only be one more thing to dismiss. 1.8.x also stored
    ``deepseek-v4-flash`` in every new agent, so it would reach nearly everyone.

    Runs on every setup rather than as a versioned migration: an agent can
    arrive on a retired id long after migration ran - from a restored backup, or
    because the model field accepts free text. Agents on a custom gateway are
    never touched (see ``migrate_legacy_chat_model``).

    The issue is deliberately never withdrawn here. Rewriting the setting is
    what makes ``migrate_legacy_chat_model`` return ``None`` on the next start,
    so withdrawing it then would erase the notice one restart after it appeared
    - before anyone had to have seen it. It stays until the user dismisses it,
    or until the entry is removed (``async_remove_entry``).
    """
    base_url = entry.data.get(CONF_BASE_URL)
    #: Agents whose model no longer answered, and the ids they were on.
    stranded: list[str] = []
    stranded_models: dict[str, str] = {}
    new_model = ""

    for subentry in list(entry.subentries.values()):
        if subentry.subentry_type not in SUBENTRY_TYPES:
            continue
        current = str(subentry.data.get(CONF_CHAT_MODEL) or "")
        replacement = migrate_legacy_chat_model(current, base_url=base_url)
        if replacement is None:
            continue
        hass.config_entries.async_update_subentry(
            entry,
            subentry,
            data={**subentry.data, CONF_CHAT_MODEL: replacement},
        )
        new_model = replacement
        retired_on = legacy_chat_model_retired_on(current)
        if is_rerouted_chat_model(current):
            LOGGER.info(
                "Agent %s was set to %s, which DeepSeek retired on %s and "
                "already answers with %s; it is set to %s now",
                subentry.title,
                current,
                retired_on,
                replacement,
                replacement,
            )
            continue
        stranded.append(subentry.title)
        stranded_models[current] = retired_on

    if not stranded:
        return

    old_models = ", ".join(sorted(stranded_models))
    retired_on = ", ".join(sorted(set(stranded_models.values())))
    LOGGER.warning(
        "Agent(s) %s are set to %s, which DeepSeek stopped serving on %s; "
        "switching them to %s",
        ", ".join(stranded),
        old_models,
        retired_on,
        new_model,
    )
    ir.async_create_issue(
        hass,
        DOMAIN,
        legacy_model_issue_id(entry),
        is_fixable=False,
        severity=ir.IssueSeverity.WARNING,
        translation_key="legacy_chat_model",
        translation_placeholders={
            "entry_title": ", ".join(stranded),
            "old_model": old_models,
            "new_model": new_model,
            "retired_on": retired_on,
        },
        learn_more_url="https://api-docs.deepseek.com/quick_start/pricing",
    )
