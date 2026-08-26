"""The DeepSeek Conversation integration."""

from __future__ import annotations

from contextlib import suppress
from types import MappingProxyType
from typing import Any

import openai
import voluptuous as vol

from homeassistant.config_entries import (  # pyright: ignore[reportMissingImports]
    ConfigEntry,
    ConfigSubentry,
)
from homeassistant.const import CONF_API_KEY, CONF_LLM_HASS_API, Platform  # pyright: ignore[reportMissingImports]
from homeassistant.core import (  # pyright: ignore[reportMissingImports]
    callback,
    HomeAssistant,
    ServiceCall,
    ServiceResponse,
    SupportsResponse,
)
from homeassistant.components import persistent_notification  # pyright: ignore[reportMissingImports]
from homeassistant.exceptions import (  # pyright: ignore[reportMissingImports]
    ConfigEntryAuthFailed,
    ConfigEntryNotReady,
    HomeAssistantError,
    ServiceValidationError,
)
from homeassistant.helpers import (  # pyright: ignore[reportMissingImports]
    config_validation as cv,
    entity_registry as er,
    issue_registry as ir,
    selector,
    translation,
)
from homeassistant.helpers.httpx_client import get_async_client  # pyright: ignore[reportMissingImports]
from homeassistant.helpers.typing import ConfigType  # pyright: ignore[reportMissingImports]

from .api_errors import openai_exception_user_message
from .config_flow import async_probe_deepseek_client
from .const import (
    ai_task_options_from,
    blocking_request_timeout_from_options,
    build_generate_content_completion_args,
    CONF_BASE_URL,
    CONF_CHAT_MODEL,
    CONF_FILENAMES,
    CONF_MAX_TOKENS,
    CONF_PROMPT,
    CONF_RESPONSE_FORMAT,
    CONF_TEMPERATURE,
    CONF_THINKING_ENABLED,
    CONF_RECOMMENDED,
    DEEPSEEK_MAX_RETRIES,
    DEFAULT_AI_TASK_NAME,
    DEFAULT_CONVERSATION_NAME,
    DEFAULT_SYSTEM_PROMPT,
    DEEPSEEK_API_BASE_URL,
    DOMAIN,
    effective_thinking_enabled_for_generate_content,
    fold_context_switch,
    LEGACY_CHAT_MODEL_RETIRED_ON,
    LOGGER,
    MAX_TOKENS_UPPER_BOUND,
    migrate_legacy_chat_model,
    reasoning_text_from_chat_message,
    request_timeout_from_options,
    resolve_generate_content_model,
    RESPONSE_FORMAT_JSON_OBJECT,
    SUBENTRY_TYPE_AI_TASK,
    SUBENTRY_TYPE_CONVERSATION,
    SUBENTRY_TYPES,
)
from .debug import async_run_debug_suite
from .structured_output import ensure_json_mode_prompt_keyword
from .types import (
    default_agent_options,
    DeepSeekConfigEntry,
    DeepSeekRuntimeData,
)
from .usage_metrics import UsageTracker, completion_usage_from_api
from .user_context import async_render_standalone_prompt
from .vision import (
    async_image_parts_from_filenames,
    raise_if_vision_unsupported,
    vision_enabled_in_options,
)
from .web_search import async_register_web_search_api


SERVICE_GENERATE_CONTENT = "generate_content"
SERVICE_RUN_DEBUG = "run_debug"

PLATFORMS = (Platform.AI_TASK, Platform.CONVERSATION, Platform.SENSOR, Platform.BUTTON)
CONFIG_SCHEMA = cv.config_entry_only_config_schema(DOMAIN)


async def _async_localize(
    hass: HomeAssistant, key: str, **placeholders: str
) -> str:
    """Return a localized string from this integration's ``common`` translations."""
    localize_key = f"component.{DOMAIN}.common.{key}"
    strings = await translation.async_get_translations(
        hass, hass.config.language, "common", integrations=[DOMAIN]
    )
    message = strings.get(localize_key, key)
    if placeholders:
        with suppress(KeyError):
            message = message.format(**placeholders)
    return message


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

    return True


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
    shared = fold_context_switch(dict(entry.options))
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


async def async_setup(hass: HomeAssistant, config: ConfigType) -> bool:
    """Set up DeepSeek Conversation."""

    async def send_prompt(call: ServiceCall) -> ServiceResponse:
        """Send a prompt to DeepSeek and return the response."""
        entry_id = call.data["config_entry"]
        entry = hass.config_entries.async_get_entry(entry_id)

        if entry is None or entry.domain != DOMAIN:
            raise ServiceValidationError(
                translation_domain=DOMAIN,
                translation_key="invalid_config_entry",
                translation_placeholders={"config_entry": entry_id},
            )

        runtime: DeepSeekRuntimeData = entry.runtime_data
        client: openai.AsyncClient = runtime.client

        base_url = entry.data.get(CONF_BASE_URL, DEEPSEEK_API_BASE_URL)
        # Resolved before anything is built: the per-call chat_model override
        # decides both whether images may be attached and whether the id is one
        # the API still serves.
        agent_options = default_agent_options(entry)
        model = resolve_generate_content_model(agent_options, call.data)
        replacement = migrate_legacy_chat_model(model, base_url=base_url)
        if replacement is not None:
            LOGGER.warning(
                "generate_content was called with %s, which the DeepSeek API "
                "stopped serving on %s; using %s for this call",
                model,
                LEGACY_CHAT_MODEL_RETIRED_ON,
                replacement,
            )
            model = replacement

        messages: list[dict[str, object]] = []
        system_prompt = (agent_options.get(CONF_PROMPT) or "").strip() or DEFAULT_SYSTEM_PROMPT
        # No chat log here, so nothing else renders the prompt for this service.
        # The speaker is whoever triggered the call - usually an automation, in
        # which case the speaker variables are simply empty.
        system_prompt = await async_render_standalone_prompt(
            hass, system_prompt, call.context
        )
        messages.append({"role": "system", "content": system_prompt})

        user_content: list[dict[str, object]] = [
            {"type": "text", "text": call.data[CONF_PROMPT]}
        ]

        filenames = call.data.get(CONF_FILENAMES, [])
        if filenames:
            if not vision_enabled_in_options(agent_options):
                raise HomeAssistantError(
                    "Vision is disabled in DeepSeek options. Enable "
                    "'Allow vision' or remove filenames from the service call."
                )
            raise_if_vision_unsupported(model, base_url=base_url)
            user_content.extend(
                await async_image_parts_from_filenames(hass, filenames)
            )

        messages.append({"role": "user", "content": user_content})

        if call.data.get(CONF_RESPONSE_FORMAT) == RESPONSE_FORMAT_JSON_OBJECT:
            if ensure_json_mode_prompt_keyword(messages):
                LOGGER.debug(
                    "[Debug generate_content]: prompt never mentioned json, which "
                    "json_object mode requires; appended the missing hint"
                )

        usage_payload: dict[str, int] | None = None
        response_text = ""
        thinking_on = effective_thinking_enabled_for_generate_content(
            agent_options, call.data
        )
        try:
            model, model_args = build_generate_content_completion_args(
                entry_options=agent_options,
                messages=messages,
                service_data=call.data,
                model=model,
            )
            LOGGER.debug(
                "[Debug generate_content]: model=%s thinking=%s overrides=%s",
                model,
                thinking_on,
                {
                    k: call.data[k]
                    for k in (
                        CONF_CHAT_MODEL,
                        CONF_TEMPERATURE,
                        CONF_THINKING_ENABLED,
                        CONF_MAX_TOKENS,
                        CONF_RESPONSE_FORMAT,
                    )
                    if k in call.data
                },
            )
            # Not streamed, so the read timeout has to cover the whole
            # generation rather than a gap between chunks.
            response = await client.with_options(
                timeout=blocking_request_timeout_from_options(agent_options)
            ).chat.completions.create(**model_args)
            message = response.choices[0].message
            response_text = message.content or ""
            if (parsed := completion_usage_from_api(response.usage)) is not None:
                runtime.usage.record(parsed, source="generate_content")
                usage_payload = runtime.usage.usage_as_dict(parsed)

        except openai.AuthenticationError as err:
            LOGGER.error("DeepSeek API key rejected: %s", err)
            entry.async_start_reauth(hass)
            raise HomeAssistantError(
                openai_exception_user_message(err)
            ) from err
        except openai.OpenAIError as err:
            LOGGER.error("Error generating content with DeepSeek: %s", err)
            raise HomeAssistantError(
                openai_exception_user_message(err)
            ) from err
        except (OSError, ValueError) as err:
            LOGGER.error("Error preparing input for DeepSeek: %s", err)
            raise HomeAssistantError(f"Error preparing input: {err}") from err

        result: dict[str, object] = {"text": response_text}
        if thinking_on:
            reasoning_text = reasoning_text_from_chat_message(message)
            result["reasoning"] = reasoning_text
            LOGGER.debug(
                "[Debug generate_content]: reasoning chars=%d",
                len(reasoning_text),
            )
        if usage_payload is not None:
            result["usage"] = usage_payload
        return result

    async def run_debug(call: ServiceCall) -> ServiceResponse:
        """Run DeepSeek diagnostics and write ``/config/deepseek_conversation_debug_report.txt``."""
        entry_id = call.data["config_entry"]
        entry = hass.config_entries.async_get_entry(entry_id)
        if entry is None or entry.domain != DOMAIN:
            raise ServiceValidationError(
                translation_domain=DOMAIN,
                translation_key="invalid_config_entry",
                translation_placeholders={"config_entry": entry_id},
            )
        log_tail = int(call.data.get("log_tail_lines", 600))
        report = await async_run_debug_suite(hass, entry, log_tail_lines=log_tail)
        path = report.get("report_path", "")
        summary = report.get("summary") or report.get("tests", {}).get("summary", {})
        parts = [
            await _async_localize(hass, "debug_notification_report", path=path),
            "",
            await _async_localize(
                hass, "debug_notification_summary", summary=str(summary)
            ),
            "",
            await _async_localize(hass, "debug_notification_errors_heading"),
        ]
        fails: list[str] = []
        for name, body in report.get("tests", {}).items():
            if name in ("summary", "client"):
                continue
            if isinstance(body, dict) and body.get("ok") is False:
                error_text = str(body.get("error", body))[:600]
                fails.append(
                    await _async_localize(
                        hass,
                        "debug_notification_error_line",
                        name=name,
                        error=error_text,
                    )
                )
        parts.extend(
            fails
            if fails
            else [await _async_localize(hass, "debug_notification_no_errors")]
        )
        parts.append("")
        parts.append(await _async_localize(hass, "debug_notification_details"))
        msg = "\n".join(parts)[:15000]
        persistent_notification.async_create(
            hass,
            title=await _async_localize(hass, "debug_notification_title"),
            message=msg,
            notification_id="deepseek_conversation_debug_done",
        )
        return {
            "report_path": path,
            "environment": report.get("environment", {}),
            "summary": report.get("summary", {}),
            "tests": report.get("tests", {}),
            "redacted_entry": report.get("redacted_entry"),
            "log_excerpt_chars": report.get("log_excerpt_chars", 0),
        }

    hass.services.async_register(
        DOMAIN,
        SERVICE_GENERATE_CONTENT,
        send_prompt,
        schema=vol.Schema(
            {
                vol.Required("config_entry"): selector.ConfigEntrySelector(
                    {"integration": DOMAIN},
                ),
                vol.Required(CONF_PROMPT): cv.string,
                vol.Optional(CONF_FILENAMES, default=[]): vol.All(
                    cv.ensure_list, [cv.string]
                ),
                vol.Optional(CONF_CHAT_MODEL): cv.string,
                vol.Optional(CONF_TEMPERATURE): vol.All(
                    vol.Coerce(float), vol.Range(min=0, max=2)
                ),
                vol.Optional(CONF_THINKING_ENABLED): cv.boolean,
                vol.Optional(CONF_MAX_TOKENS): vol.All(
                    vol.Coerce(int), vol.Range(min=1, max=MAX_TOKENS_UPPER_BOUND)
                ),
                vol.Optional(CONF_RESPONSE_FORMAT): vol.In(
                    [RESPONSE_FORMAT_JSON_OBJECT]
                ),
            }
        ),
        supports_response=SupportsResponse.ONLY,
    )

    hass.services.async_register(
        DOMAIN,
        SERVICE_RUN_DEBUG,
        run_debug,
        schema=vol.Schema(
            {
                vol.Required("config_entry"): selector.ConfigEntrySelector(
                    {"integration": DOMAIN},
                ),
                vol.Optional("log_tail_lines", default=600): vol.All(
                    int, vol.Range(min=50, max=8000)
                ),
            }
        ),
        supports_response=SupportsResponse.ONLY,
    )

    return True


def _async_http_client(hass: HomeAssistant) -> Any:
    """Return HA's shared httpx client, preferring the HTTP/2-capable one.

    The OpenAI SDK ends a streamed completion by breaking out of the SSE
    iterator at ``[DONE]`` and closing the response without draining it. Over
    HTTP/1.1 httpx cannot return such a connection to the pool, so **every** API
    round opens a new TCP+TLS connection — a full handshake per tool-calling
    round trip, which is expensive on long-haul or proxied routes. Over HTTP/2
    only the stream is closed and the connection stays pooled.

    Requires ``h2`` (declared in the manifest) and a core new enough for
    ``alpn_protocols``. ALPN negotiates per endpoint, so an API that offers no
    HTTP/2 transparently falls back to HTTP/1.1; if anything here is missing the
    plain shared HTTP/1.1 client is used, which is the previous behaviour.
    """
    try:
        import h2  # noqa: F401  # pyright: ignore[reportMissingImports]

        from homeassistant.util.ssl import SSL_ALPN_HTTP11_HTTP2  # pyright: ignore[reportMissingImports]
    except ImportError as err:
        LOGGER.debug(
            "[Debug setup]: HTTP/2 support unavailable (%s); using HTTP/1.1. "
            "Each API round will open a new TLS connection.",
            err,
        )
        return get_async_client(hass)

    try:
        client = get_async_client(hass, alpn_protocols=SSL_ALPN_HTTP11_HTTP2)
    except TypeError as err:
        # Core predates the alpn_protocols keyword.
        LOGGER.debug(
            "[Debug setup]: shared client does not accept alpn_protocols (%s); "
            "using HTTP/1.1",
            err,
        )
        return get_async_client(hass)

    LOGGER.debug(
        "[Debug setup]: using HTTP/2-capable shared client (falls back to "
        "HTTP/1.1 per endpoint via ALPN)"
    )
    return client


def _legacy_model_issue_id(entry: ConfigEntry) -> str:
    """Repair issue id for one entry left on a retired model."""
    return f"legacy_chat_model_{entry.entry_id}"


@callback
def _async_migrate_legacy_model_option(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Move agents off a retired model id, and say so in Repairs.

    ``deepseek-chat`` / ``deepseek-reasoner`` are no longer served by the
    official API, so an agent still pointing at one fails every single request
    with no hint about why. Rewriting the setting keeps the agent working; the
    repair issue is what tells the user their model choice changed.

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
    moved: list[str] = []
    old_model = new_model = ""

    for subentry in list(entry.subentries.values()):
        if subentry.subentry_type not in SUBENTRY_TYPES:
            continue
        current = subentry.data.get(CONF_CHAT_MODEL)
        replacement = migrate_legacy_chat_model(current, base_url=base_url)
        if replacement is None:
            continue
        hass.config_entries.async_update_subentry(
            entry,
            subentry,
            data={**subentry.data, CONF_CHAT_MODEL: replacement},
        )
        moved.append(subentry.title)
        old_model, new_model = str(current), replacement

    if not moved:
        return

    LOGGER.warning(
        "Agent(s) %s are set to %s, which the DeepSeek API stopped serving on "
        "%s; switching them to %s",
        ", ".join(moved),
        old_model,
        LEGACY_CHAT_MODEL_RETIRED_ON,
        new_model,
    )
    ir.async_create_issue(
        hass,
        DOMAIN,
        _legacy_model_issue_id(entry),
        is_fixable=False,
        severity=ir.IssueSeverity.WARNING,
        translation_key="legacy_chat_model",
        translation_placeholders={
            "entry_title": ", ".join(moved),
            "old_model": old_model,
            "new_model": new_model,
            "retired_on": LEGACY_CHAT_MODEL_RETIRED_ON,
        },
        learn_more_url="https://api-docs.deepseek.com/quick_start/pricing",
    )


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
    _async_migrate_legacy_model_option(hass, entry)

    base_url = entry.data.get(CONF_BASE_URL, DEEPSEEK_API_BASE_URL)
    client = openai.AsyncOpenAI(
        api_key=entry.data[CONF_API_KEY],
        base_url=base_url,
        http_client=_async_http_client(hass),
        # The SDK would default to 600 s and two retries, which lets one
        # unresponsive endpoint block a voice pipeline for ten minutes. Call
        # sites narrow this further per request (see request_timeout_from_options).
        timeout=request_timeout_from_options(default_agent_options(entry)),
        max_retries=DEEPSEEK_MAX_RETRIES,
    )

    try:
        await async_probe_deepseek_client(client)
    except openai.AuthenticationError as err:
        LOGGER.error("Invalid DeepSeek API key: %s", err)
        raise ConfigEntryAuthFailed("Invalid DeepSeek API key") from err
    except openai.APIStatusError as err:
        if err.status_code in (401, 403):
            LOGGER.error("DeepSeek rejected credentials (%s): %s", err.status_code, err)
            raise ConfigEntryAuthFailed("Invalid DeepSeek credentials") from err
        LOGGER.warning(
            "Unexpected DeepSeek status during setup probe (%s): %s",
            err.status_code,
            err,
        )
        raise ConfigEntryNotReady(
            f"DeepSeek API returned {err.status_code}: {err}"
        ) from err
    except openai.APIConnectionError as err:
        LOGGER.error("Failed to connect to DeepSeek API: %s", err)
        raise ConfigEntryNotReady(
            f"Failed to connect to DeepSeek API: {err}"
        ) from err
    except openai.OpenAIError as err:
        LOGGER.error("DeepSeek SDK error during setup: %s", err)
        raise ConfigEntryNotReady(f"DeepSeek API error: {err}") from err

    entry.runtime_data = DeepSeekRuntimeData(client=client, usage=UsageTracker())

    # Optional Brave web_search LLM API (see web_search.py); only when key in entry.data.
    async_register_web_search_api(hass, entry)

    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)

    # Registered last, so the subentry rewrite above cannot trigger a reload.
    entry.async_on_unload(entry.add_update_listener(_async_reload_entry))

    return True


async def async_remove_entry(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Withdraw this entry's repair issues when it is deleted."""
    ir.async_delete_issue(hass, DOMAIN, _legacy_model_issue_id(entry))


async def async_unload_entry(hass: HomeAssistant, entry: DeepSeekConfigEntry) -> bool:
    """Unload DeepSeek platforms.

    The OpenAI client is built on Home Assistant's shared httpx client (see
    ``get_async_client`` in ``async_setup_entry``). That connection pool is owned
    by HA and must not be closed here — doing so only triggers a framework warning
    without releasing anything — so unload just tears down the platforms.
    """
    return await hass.config_entries.async_unload_platforms(entry, PLATFORMS)
