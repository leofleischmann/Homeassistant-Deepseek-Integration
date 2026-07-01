"""The DeepSeek Conversation integration."""

from __future__ import annotations

import base64
from mimetypes import guess_file_type
from pathlib import Path

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
from homeassistant.helpers import config_validation as cv, selector  # pyright: ignore[reportMissingImports]
from homeassistant.helpers.httpx_client import get_async_client  # pyright: ignore[reportMissingImports]
from homeassistant.helpers.typing import ConfigType  # pyright: ignore[reportMissingImports]

from .api_errors import openai_exception_user_message
from .config_flow import async_probe_deepseek_client
from .const import (
    build_chat_completion_args,
    CONF_BASE_URL,
    CONF_CHAT_MODEL,
    CONF_FILENAMES,
    CONF_PROMPT,
    DEFAULT_SYSTEM_PROMPT,
    DEEPSEEK_API_BASE_URL,
    DOMAIN,
    LOGGER,
    RECOMMENDED_CHAT_MODEL,
)
from .debug import async_run_debug_suite
from .types import DeepSeekConfigEntry

SERVICE_GENERATE_CONTENT = "generate_content"
SERVICE_RUN_DEBUG = "run_debug"

PLATFORMS = (Platform.CONVERSATION,)
CONFIG_SCHEMA = cv.config_entry_only_config_schema(DOMAIN)


async def async_migrate_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Migrate config entry to version 2 — wrap legacy string CONF_LLM_HASS_API in a list."""
    if entry.version < 2:
        options = {**entry.options}
        legacy = options.get(CONF_LLM_HASS_API)
        if isinstance(legacy, str):
            options[CONF_LLM_HASS_API] = [legacy] if legacy != "none" else []
        hass.config_entries.async_update_entry(entry, options=options, version=2)
    return True


def encode_file(file_path: str) -> tuple[str, str]:
    """Return base64 version of file contents."""
    mime_type, _ = guess_file_type(file_path)
    if mime_type is None:
        mime_type = "application/octet-stream"
    with open(file_path, "rb") as image_file:
        return (mime_type, base64.b64encode(image_file.read()).decode("utf-8"))


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

        model: str = entry.options.get(CONF_CHAT_MODEL, RECOMMENDED_CHAT_MODEL)
        client: openai.AsyncClient = entry.runtime_data

        messages: list[dict[str, object]] = []
        system_prompt = (entry.options.get(CONF_PROMPT) or "").strip() or DEFAULT_SYSTEM_PROMPT
        messages.append({"role": "system", "content": system_prompt})

        user_content: list[dict[str, object]] = [
            {"type": "text", "text": call.data[CONF_PROMPT]}
        ]

        async def append_files_to_content() -> None:
            for filename in call.data.get(CONF_FILENAMES, []):
                if not hass.config.is_allowed_path(filename):
                    LOGGER.warning(
                        "Cannot read %s, no access to path; "
                        "`allowlist_external_dirs` may need to be adjusted in "
                        "`configuration.yaml`",
                        filename,
                    )
                    continue
                if not Path(filename).exists():
                    LOGGER.warning("%s does not exist", filename)
                    continue

                try:
                    mime_type, base64_file = await hass.async_add_executor_job(
                        encode_file, filename
                    )
                    if "image/" in mime_type:
                        user_content.append(
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{mime_type};base64,{base64_file}",
                                },
                            }
                        )
                    else:
                        LOGGER.warning(
                            "Skipping file %s: unsupported type %s",
                            filename,
                            mime_type,
                        )
                except Exception as err:
                    LOGGER.error("Error processing file %s: %s", filename, err)

        if CONF_FILENAMES in call.data:
            await append_files_to_content()

        messages.append({"role": "user", "content": user_content})

        try:
            model_args = build_chat_completion_args(
                model=model,
                messages=messages,
                options=entry.options,
                stream=False,
            )
            response = await client.chat.completions.create(**model_args)
            response_text = response.choices[0].message.content

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

        return {"text": response_text or ""}

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
            f"Bericht: {path}",
            "",
            f"Zusammenfassung: {summary}",
            "",
            "Fehler (falls vorhanden):",
        ]
        fails: list[str] = []
        for name, body in report.get("tests", {}).items():
            if name in ("summary", "client"):
                continue
            if isinstance(body, dict) and body.get("ok") is False:
                fails.append(f"- {name}: {str(body.get('error', body))[:600]}")
        parts.extend(fails if fails else ["(keine)"])
        parts.append("")
        parts.append("Details: Berichtdatei öffnen (Text + JSON-Anhang).")
        msg = "\n".join(parts)[:15000]
        persistent_notification.async_create(
            hass,
            title="DeepSeek-Debug abgeschlossen",
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

    entry.runtime_data = client

    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)

    return True


async def async_unload_entry(hass: HomeAssistant, entry: DeepSeekConfigEntry) -> bool:
    """Unload DeepSeek and close the underlying OpenAI client.

    The OpenAI ``AsyncClient`` owns an httpx connection pool; HA recreates the
    entry on every options-update via ``async_reload``, so without an explicit
    ``close()`` the pool would leak per reload.
    """
    unload_ok = await hass.config_entries.async_unload_platforms(entry, PLATFORMS)
    client: openai.AsyncClient | None = getattr(entry, "runtime_data", None)
    if client is not None:
        try:
            await client.close()
        except (openai.OpenAIError, OSError, RuntimeError) as err:
            LOGGER.debug("Error closing DeepSeek client on unload: %s", err)
    return unload_ok
