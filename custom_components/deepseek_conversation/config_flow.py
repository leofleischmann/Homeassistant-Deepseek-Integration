"""Config flow for DeepSeek Conversation integration."""

from __future__ import annotations

import logging
from types import MappingProxyType
from typing import Any

import openai
import voluptuous as vol

from homeassistant.config_entries import (  # pyright: ignore[reportMissingImports]
    ConfigEntry,
    ConfigFlow,
    ConfigFlowResult,
    OptionsFlow,
)
# --- Import CONF_LLM_HASS_API ---
from homeassistant.const import CONF_API_KEY, CONF_LLM_HASS_API  # pyright: ignore[reportMissingImports]
# --- End Import ---
from homeassistant.core import HomeAssistant  # pyright: ignore[reportMissingImports]
from homeassistant.helpers import llm  # pyright: ignore[reportMissingImports]
from homeassistant.helpers.httpx_client import get_async_client  # pyright: ignore[reportMissingImports]
from homeassistant.helpers.selector import (  # pyright: ignore[reportMissingImports]
    NumberSelector,
    NumberSelectorConfig,
    SelectOptionDict,
    SelectSelector,
    SelectSelectorConfig,
    TemplateSelector,
)
from homeassistant.helpers.typing import VolDictType  # pyright: ignore[reportMissingImports]

# Updated imports from const
from .const import (
    CONF_CHAT_MODEL,
    CONF_MAX_TOKENS,
    CONF_PROMPT,
    CONF_TEMPERATURE,
    CONF_TOP_P,
    CONF_BASE_URL,
    DOMAIN,
    RECOMMENDED_CHAT_MODEL,
    RECOMMENDED_MAX_TOKENS,
    RECOMMENDED_TEMPERATURE,
    RECOMMENDED_TOP_P,
    DEEPSEEK_API_BASE_URL,
)

_LOGGER = logging.getLogger(__name__)

STEP_USER_DATA_SCHEMA = vol.Schema(
    {
        vol.Required(CONF_API_KEY): str,
        vol.Optional(CONF_BASE_URL, default=DEEPSEEK_API_BASE_URL): str,
        vol.Optional(CONF_CHAT_MODEL, default=RECOMMENDED_CHAT_MODEL): str,
    }
)

# Add CONF_LLM_HASS_API back to default options if desired, e.g., default to Assist
DEFAULT_OPTIONS = {
    CONF_LLM_HASS_API: llm.LLM_API_ASSIST, # Default to Assist API
    CONF_PROMPT: llm.DEFAULT_INSTRUCTIONS_PROMPT,
    CONF_CHAT_MODEL: RECOMMENDED_CHAT_MODEL,
    CONF_MAX_TOKENS: RECOMMENDED_MAX_TOKENS, # Remember to increase this in UI!
    CONF_TEMPERATURE: RECOMMENDED_TEMPERATURE,
    CONF_TOP_P: RECOMMENDED_TOP_P,
}


async def validate_input(hass: HomeAssistant, data: dict[str, Any]) -> None:
    """Validate the user input allows us to connect."""
    base_url = data.get(CONF_BASE_URL, DEEPSEEK_API_BASE_URL)
    if base_url:
        base_url = base_url.strip()
    if not base_url:
        base_url = DEEPSEEK_API_BASE_URL
    
    model = data.get(CONF_CHAT_MODEL, RECOMMENDED_CHAT_MODEL)
    
    client = openai.AsyncOpenAI(
        api_key=data[CONF_API_KEY],
        base_url=base_url,
        http_client=get_async_client(hass)
    )
    await client.with_options(timeout=10.0).chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "Hi"}],
        max_tokens=1,
    )


class DeepSeekConfigFlow(ConfigFlow, domain=DOMAIN):
    """Handle a config flow for DeepSeek Conversation."""

    VERSION = 1 # Consider incrementing if significant changes are made

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Handle the initial step."""
        if user_input is None:
            return self.async_show_form(
                step_id="user", data_schema=STEP_USER_DATA_SCHEMA
            )

        errors: dict[str, str] = {}

        try:
            await validate_input(self.hass, user_input)
        except openai.APIConnectionError:
            errors["base"] = "cannot_connect"
        except openai.AuthenticationError:
            errors["base"] = "invalid_auth"
        except openai.OpenAIError as e:
            _LOGGER.error("DeepSeek API error during validation: %s", e)
            errors["base"] = "api_error"
        except Exception:
            _LOGGER.exception("Unexpected exception during validation")
            errors["base"] = "unknown"
        else:
            # Separate data (connection settings) from options (model settings)
            entry_data = {
                CONF_API_KEY: user_input[CONF_API_KEY],
                CONF_BASE_URL: user_input.get(CONF_BASE_URL, DEEPSEEK_API_BASE_URL),
            }
            # Move chat_model to options if provided
            entry_options = {**DEFAULT_OPTIONS}
            if CONF_CHAT_MODEL in user_input:
                entry_options[CONF_CHAT_MODEL] = user_input[CONF_CHAT_MODEL]
            
            return self.async_create_entry(
                title="DeepSeek",
                data=entry_data,
                options=entry_options,
            )

        return self.async_show_form(
            step_id="user", data_schema=STEP_USER_DATA_SCHEMA, errors=errors
        )

    @staticmethod
    def async_get_options_flow(
        config_entry: ConfigEntry,
    ) -> OptionsFlow:
        """Create the options flow."""
        return DeepSeekOptionsFlow(config_entry)


class DeepSeekOptionsFlow(OptionsFlow):
    """DeepSeek config flow options handler."""

    # --- Reinstated __init__ method ---
    def __init__(self, config_entry: ConfigEntry) -> None:
        """Initialize options flow."""
        self.config_entry = config_entry
    # --- End reinstatement ---

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Manage the options."""
        # self.config_entry should be automatically available here
        errors: dict[str, str] = {}

        if user_input is not None:
            # Handle CONF_LLM_HASS_API selection
            if user_input.get(CONF_LLM_HASS_API) == "none":
                 user_input.pop(CONF_LLM_HASS_API, None)

            # Handle base URL update - move it from options to data if changed
            base_url_changed = False
            if CONF_BASE_URL in user_input:
                base_url = user_input.pop(CONF_BASE_URL).strip()
                # Validate URL is not empty
                if not base_url:
                    errors[CONF_BASE_URL] = "url_required"
                else:
                    # Normalize URL (ensure it ends with /v1 or similar, or let user specify full path)
                    if base_url != self.config_entry.data.get(CONF_BASE_URL, DEEPSEEK_API_BASE_URL):
                        base_url_changed = True
                        # Update the config entry data
                        new_data = {**self.config_entry.data, CONF_BASE_URL: base_url}
                        self.hass.config_entries.async_update_entry(self.config_entry, data=new_data)

            if not errors:
                # Merge new user input with existing options before creating entry
                updated_options = {**self.config_entry.options, **user_input}
                result = self.async_create_entry(title="", data=updated_options)
                
                # Reload the entry if base URL was changed to apply the new URL
                if base_url_changed:
                    self.hass.async_create_task(
                        self.hass.config_entries.async_reload(self.config_entry.entry_id)
                    )
                
                return result

        # Pass options from self.config_entry to the schema function
        schema = deepseek_config_option_schema(self.hass, self.config_entry.options, self.config_entry)
        return self.async_show_form(
            step_id="init",
            data_schema=vol.Schema(schema),
            errors=errors,
        )


def deepseek_config_option_schema(
    hass: HomeAssistant,
    options: dict[str, Any] | MappingProxyType[str, Any],
    config_entry: ConfigEntry | None = None,
) -> VolDictType:
    """Return a schema for DeepSeek completion options."""
    # Re-add HASS API selection
    hass_apis: list[SelectOptionDict] = [
        SelectOptionDict(label="No control", value="none")
    ]
    hass_apis.extend(
        SelectOptionDict(label=api.name, value=api.id)
        for api in llm.async_get_apis(hass)
    )

    # Get base URL from config entry data if available, otherwise from options or default
    base_url = DEEPSEEK_API_BASE_URL
    if config_entry:
        base_url = config_entry.data.get(CONF_BASE_URL, DEEPSEEK_API_BASE_URL)
    elif CONF_BASE_URL in options:
        base_url = options[CONF_BASE_URL]

    schema: VolDictType = {
        vol.Optional(
            CONF_BASE_URL,
            description={"suggested_value": base_url},
            default=base_url,
        ): str,
        vol.Optional(
            CONF_PROMPT,
            description={
                "suggested_value": options.get(
                    CONF_PROMPT, llm.DEFAULT_INSTRUCTIONS_PROMPT
                )
            },
            default=llm.DEFAULT_INSTRUCTIONS_PROMPT,
        ): TemplateSelector(),
        # Add selector for CONF_LLM_HASS_API
        vol.Optional(
            CONF_LLM_HASS_API,
            description={"suggested_value": options.get(CONF_LLM_HASS_API)},
            default=options.get(CONF_LLM_HASS_API) or "none",
        ): SelectSelector(SelectSelectorConfig(options=hass_apis)),
        vol.Optional(
            CONF_CHAT_MODEL,
            description={"suggested_value": options.get(CONF_CHAT_MODEL)},
            default=RECOMMENDED_CHAT_MODEL,
        ): str,
        vol.Optional(
            CONF_MAX_TOKENS,
            description={"suggested_value": options.get(CONF_MAX_TOKENS)},
            default=RECOMMENDED_MAX_TOKENS,
        ): int,
        vol.Optional(
            CONF_TOP_P,
            description={"suggested_value": options.get(CONF_TOP_P)},
            default=RECOMMENDED_TOP_P,
        ): NumberSelector(NumberSelectorConfig(min=0, max=1, step=0.05, mode="slider")),
        vol.Optional(
            CONF_TEMPERATURE,
            description={"suggested_value": options.get(CONF_TEMPERATURE)},
            default=RECOMMENDED_TEMPERATURE,
        ): NumberSelector(NumberSelectorConfig(min=0, max=2, step=0.05, mode="slider")),
    }
    return schema
