"""The two actions this integration registers.

``generate_content`` is the one-shot prompt: no chat log, no tool loop, an
optional image, and a JSON mode for automations that want to parse the answer.
``run_debug`` writes the diagnostics report.

Both address an agent rather than a config entry, because one entity id says
both which credentials to use and which prompt and model to answer with -
see ``_async_resolve_target``.
"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import suppress
from functools import partial
from typing import Any

import openai
import voluptuous as vol

from homeassistant.components import persistent_notification  # pyright: ignore[reportMissingImports]
from homeassistant.config_entries import ConfigEntryState  # pyright: ignore[reportMissingImports]
from homeassistant.core import (  # pyright: ignore[reportMissingImports]
    callback,
    HomeAssistant,
    ServiceCall,
    ServiceResponse,
    SupportsResponse,
)
from homeassistant.exceptions import (  # pyright: ignore[reportMissingImports]
    HomeAssistantError,
    ServiceValidationError,
)
from homeassistant.helpers import (  # pyright: ignore[reportMissingImports]
    config_validation as cv,
    selector,
    translation,
)

from .api_errors import openai_exception_user_message
from .const import (
    CONF_AGENT,
    CONF_BASE_URL,
    CONF_CHAT_MODEL,
    CONF_CONFIG_ENTRY,
    CONF_FILENAMES,
    CONF_MAX_TOKENS,
    CONF_PROMPT,
    CONF_RESPONSE_FORMAT,
    CONF_TEMPERATURE,
    CONF_THINKING_ENABLED,
    DEEPSEEK_API_BASE_URL,
    DEFAULT_SYSTEM_PROMPT,
    DOMAIN,
    LEGACY_CHAT_MODEL_RETIRED_ON,
    LOGGER,
    MAX_TOKENS_UPPER_BOUND,
    RESPONSE_FORMAT_JSON_OBJECT,
)
from .debug import async_run_debug_suite
from .models import migrate_legacy_chat_model
from .options import blocking_request_timeout_from_options
from .request_builder import (
    build_generate_content_completion_args,
    effective_thinking_enabled_for_generate_content,
    reasoning_text_from_chat_message,
    resolve_generate_content_model,
)
from .structured_output import ensure_json_mode_prompt_keyword
from .types import (
    agent_for_entity,
    default_agent_options,
    DeepSeekConfigEntry,
    DeepSeekRuntimeData,
)
from .usage_metrics import completion_usage_from_api
from .user_context import async_render_standalone_prompt
from .vision import (
    async_image_parts_from_filenames,
    raise_if_vision_unsupported,
    vision_enabled_in_options,
)

SERVICE_GENERATE_CONTENT = "generate_content"
SERVICE_RUN_DEBUG = "run_debug"

#: Both actions accept either target: an agent entity, or the config entry
#: whose first agent then answers.
_TARGET_SCHEMA = {
    vol.Optional(CONF_CONFIG_ENTRY): selector.ConfigEntrySelector(
        {"integration": DOMAIN},
    ),
    vol.Optional(CONF_AGENT): selector.EntitySelector(
        {"integration": DOMAIN, "domain": ["conversation", "ai_task"]},
    ),
}

GENERATE_CONTENT_SCHEMA = vol.Schema(
    {
        **_TARGET_SCHEMA,
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
        vol.Optional(CONF_RESPONSE_FORMAT): vol.In([RESPONSE_FORMAT_JSON_OBJECT]),
    }
)

RUN_DEBUG_SCHEMA = vol.Schema(
    {
        **_TARGET_SCHEMA,
        vol.Optional("log_tail_lines", default=600): vol.All(
            int, vol.Range(min=50, max=8000)
        ),
    }
)


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


@callback
def _async_resolve_target(
    hass: HomeAssistant, call: ServiceCall
) -> tuple[DeepSeekConfigEntry, Mapping[str, Any]]:
    """Return the entry an action addresses, and whose settings to answer with.

    Naming an agent is the precise way: one entity id says both which
    credentials to use and which prompt and model to answer with. A bare config
    entry still works and follows that entry's first agent, which is what every
    call did before an entry could hold more than one.
    """
    agent_entity = call.data.get(CONF_AGENT)
    entry_id = call.data.get(CONF_CONFIG_ENTRY)

    if agent_entity:
        resolved = agent_for_entity(hass, agent_entity)
        if resolved is None:
            raise ServiceValidationError(
                translation_domain=DOMAIN,
                translation_key="invalid_agent",
                translation_placeholders={"agent": agent_entity},
            )
        entry, options = resolved
        if entry_id and entry_id != entry.entry_id:
            raise ServiceValidationError(
                translation_domain=DOMAIN,
                translation_key="agent_entry_mismatch",
                translation_placeholders={"agent": agent_entity},
            )
    elif entry_id:
        found = hass.config_entries.async_get_entry(entry_id)
        if found is None or found.domain != DOMAIN:
            raise ServiceValidationError(
                translation_domain=DOMAIN,
                translation_key="invalid_config_entry",
                translation_placeholders={"config_entry": entry_id},
            )
        entry, options = found, default_agent_options(found)
    else:
        raise ServiceValidationError(
            translation_domain=DOMAIN, translation_key="no_target"
        )

    if entry.state is not ConfigEntryState.LOADED:
        # runtime_data only exists on a loaded entry, and reaching for it
        # otherwise raises something nobody can act on.
        raise ServiceValidationError(
            translation_domain=DOMAIN,
            translation_key="entry_not_loaded",
            translation_placeholders={"config_entry": entry.title},
        )
    return entry, options


async def send_prompt(hass: HomeAssistant, call: ServiceCall) -> ServiceResponse:
    """Send a prompt to DeepSeek and return the response."""
    entry, agent_options = _async_resolve_target(hass, call)

    runtime: DeepSeekRuntimeData = entry.runtime_data
    client: openai.AsyncClient = runtime.client

    base_url = entry.data.get(CONF_BASE_URL, DEEPSEEK_API_BASE_URL)
    # Resolved before anything is built: the per-call chat_model override
    # decides both whether images may be attached and whether the id is one
    # the API still serves.
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
                "Images are switched off for this agent. Turn on "
                "'Allow images' in its settings, or call the action "
                "without filenames."
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
            agent_options=agent_options,
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


async def run_debug(hass: HomeAssistant, call: ServiceCall) -> ServiceResponse:
    """Run DeepSeek diagnostics and write ``/config/deepseek_conversation_debug_report.txt``."""
    entry, agent_options = _async_resolve_target(hass, call)
    log_tail = int(call.data.get("log_tail_lines", 600))
    report = await async_run_debug_suite(
        hass, entry, agent_options=agent_options, log_tail_lines=log_tail
    )
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


@callback
def async_setup_services(hass: HomeAssistant) -> None:
    """Register both actions. Called once from ``async_setup``."""
    hass.services.async_register(
        DOMAIN,
        SERVICE_GENERATE_CONTENT,
        partial(send_prompt, hass),
        schema=GENERATE_CONTENT_SCHEMA,
        supports_response=SupportsResponse.ONLY,
    )
    hass.services.async_register(
        DOMAIN,
        SERVICE_RUN_DEBUG,
        partial(run_debug, hass),
        schema=RUN_DEBUG_SCHEMA,
        supports_response=SupportsResponse.ONLY,
    )
