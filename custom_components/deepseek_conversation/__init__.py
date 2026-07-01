"""The DeepSeek Conversation integration."""

from __future__ import annotations

from contextlib import suppress

import openai
import voluptuous as vol

from homeassistant.config_entries import ConfigEntry  # pyright: ignore[reportMissingImports]
from homeassistant.const import CONF_API_KEY, CONF_LLM_HASS_API, Platform  # pyright: ignore[reportMissingImports]
from homeassistant.core import (  # pyright: ignore[reportMissingImports]
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
    selector,
    translation,
)
from homeassistant.helpers.httpx_client import get_async_client  # pyright: ignore[reportMissingImports]
from homeassistant.helpers.typing import ConfigType  # pyright: ignore[reportMissingImports]

from .api_errors import openai_exception_user_message
from .config_flow import async_probe_deepseek_client
from .const import (
    build_generate_content_completion_args,
    CONF_BASE_URL,
    CONF_CHAT_MODEL,
    CONF_FILENAMES,
    CONF_MAX_TOKENS,
    CONF_PROMPT,
    CONF_RESPONSE_FORMAT,
    CONF_TEMPERATURE,
    CONF_THINKING_ENABLED,
    DEFAULT_SYSTEM_PROMPT,
    DEEPSEEK_API_BASE_URL,
    DOMAIN,
    effective_thinking_enabled_for_generate_content,
    LOGGER,
    MAX_TOKENS_UPPER_BOUND,
    reasoning_text_from_chat_message,
    RESPONSE_FORMAT_JSON_OBJECT,
)
from .debug import async_run_debug_suite
from .types import DeepSeekConfigEntry, DeepSeekRuntimeData
from .usage_metrics import UsageTracker, completion_usage_from_api
from .vision import (
    async_image_parts_from_filenames,
    raise_if_vision_unsupported_for_api,
    vision_enabled_in_options,
)


SERVICE_GENERATE_CONTENT = "generate_content"
SERVICE_RUN_DEBUG = "run_debug"

PLATFORMS = (Platform.CONVERSATION, Platform.SENSOR, Platform.BUTTON)
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
    """Migrate config entry to version 2 — wrap legacy string CONF_LLM_HASS_API in a list."""
    if entry.version < 2:
        options = {**entry.options}
        legacy = options.get(CONF_LLM_HASS_API)
        if isinstance(legacy, str):
            options[CONF_LLM_HASS_API] = [legacy] if legacy != "none" else []
        hass.config_entries.async_update_entry(entry, options=options, version=2)
    return True


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

        messages: list[dict[str, object]] = []
        system_prompt = (entry.options.get(CONF_PROMPT) or "").strip() or DEFAULT_SYSTEM_PROMPT
        messages.append({"role": "system", "content": system_prompt})

        user_content: list[dict[str, object]] = [
            {"type": "text", "text": call.data[CONF_PROMPT]}
        ]

        filenames = call.data.get(CONF_FILENAMES, [])
        if filenames:
            if not vision_enabled_in_options(entry.options):
                raise HomeAssistantError(
                    "Vision is disabled in DeepSeek options. Enable "
                    "'Allow vision' or remove filenames from the service call."
                )
            raise_if_vision_unsupported_for_api(
                entry.data.get(CONF_BASE_URL, DEEPSEEK_API_BASE_URL)
            )
            user_content.extend(
                await async_image_parts_from_filenames(hass, filenames)
            )

        messages.append({"role": "user", "content": user_content})

        usage_payload: dict[str, int] | None = None
        response_text = ""
        thinking_on = effective_thinking_enabled_for_generate_content(
            entry.options, call.data
        )
        try:
            model, model_args = build_generate_content_completion_args(
                entry_options=entry.options,
                messages=messages,
                service_data=call.data,
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
            response = await client.chat.completions.create(**model_args)
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


async def async_setup_entry(hass: HomeAssistant, entry: DeepSeekConfigEntry) -> bool:
    """Set up DeepSeek Conversation from a config entry."""
    base_url = entry.data.get(CONF_BASE_URL, DEEPSEEK_API_BASE_URL)
    client = openai.AsyncOpenAI(
        api_key=entry.data[CONF_API_KEY],
        base_url=base_url,
        http_client=get_async_client(hass),
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

    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)

    return True


async def async_unload_entry(hass: HomeAssistant, entry: DeepSeekConfigEntry) -> bool:
    """Unload DeepSeek platforms.

    The OpenAI client is built on Home Assistant's shared httpx client (see
    ``get_async_client`` in ``async_setup_entry``). That connection pool is owned
    by HA and must not be closed here — doing so only triggers a framework warning
    without releasing anything — so unload just tears down the platforms.
    """
    return await hass.config_entries.async_unload_platforms(entry, PLATFORMS)
