"""Config flow for DeepSeek Conversation integration."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

import openai
import voluptuous as vol  # pyright: ignore[reportMissingImports]

from homeassistant.config_entries import (  # pyright: ignore[reportMissingImports]
    ConfigEntry,
    ConfigFlow,
    ConfigFlowResult,
    OptionsFlow,
    SOURCE_RECONFIGURE,
)
from homeassistant.const import CONF_API_KEY, CONF_LLM_HASS_API  # pyright: ignore[reportMissingImports]
from homeassistant.core import HomeAssistant  # pyright: ignore[reportMissingImports]
from homeassistant.helpers import llm  # pyright: ignore[reportMissingImports]
from homeassistant.helpers.httpx_client import get_async_client  # pyright: ignore[reportMissingImports]
from homeassistant.helpers.selector import (  # pyright: ignore[reportMissingImports]
    BooleanSelector,
    NumberSelector,
    NumberSelectorConfig,
    SelectOptionDict,
    SelectSelector,
    SelectSelectorConfig,
    TemplateSelector,
)
from homeassistant.helpers.typing import VolDictType  # pyright: ignore[reportMissingImports]

from .const import (
    CHAT_MODEL_OPTIONS,
    coerce_max_tokens,
    CONF_BASE_URL,
    CONF_CHAT_MODEL,
    CONF_MAX_TOKENS,
    CONF_PROMPT,
    CONF_REASONING_EFFORT,
    CONF_STRIP_MARKDOWN,
    CONF_TEMPERATURE,
    CONF_THINKING_ENABLED,
    CONF_TOP_P,
    DEFAULT_STRIP_MARKDOWN,
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_THINKING_ENABLED,
    DEEPSEEK_API_BASE_URL,
    DOMAIN,
    LOGGER,
    MAX_TOKENS_UPPER_BOUND,
    REASONING_EFFORT_SELECT,
    RECOMMENDED_CHAT_MODEL,
    RECOMMENDED_MAX_TOKENS,
    RECOMMENDED_REASONING_EFFORT,
    RECOMMENDED_TEMPERATURE,
    RECOMMENDED_TOP_P,
)

def _normalize_llm_hass_api(value: Any) -> list[str] | None:
    """Normalize CONF_LLM_HASS_API to a list for multi-select, or None if unset."""
    if isinstance(value, list):
        return value if value else None
    if isinstance(value, str):
        return [value] if value != "none" else None
    return None


def _chat_model_select_options() -> list[SelectOptionDict]:
    return [SelectOptionDict(value=v, label=lbl) for v, lbl in CHAT_MODEL_OPTIONS]


def _chat_model_selector() -> SelectSelector:
    return SelectSelector(
        SelectSelectorConfig(
            options=_chat_model_select_options(),
            custom_value=True,
            translation_key=CONF_CHAT_MODEL,
        )
    )


def get_user_step_schema() -> vol.Schema:
    """Schema for initial config (API key, URL, V4 / legacy model)."""
    return vol.Schema(
        {
            vol.Required(CONF_API_KEY): str,
            vol.Optional(CONF_BASE_URL, default=DEEPSEEK_API_BASE_URL): str,
            vol.Optional(
                CONF_CHAT_MODEL, default=RECOMMENDED_CHAT_MODEL
            ): _chat_model_selector(),
        }
    )


STEP_REAUTH_DATA_SCHEMA = vol.Schema(
    {
        vol.Required(CONF_API_KEY): str,
    }
)


def get_reconfigure_step_schema(entry: ConfigEntry) -> vol.Schema:
    """Schema for reconfigure (API key + base URL stored in config entry data)."""
    return vol.Schema(
        {
            vol.Required(CONF_API_KEY): str,
            vol.Optional(
                CONF_BASE_URL,
                default=entry.data.get(CONF_BASE_URL, DEEPSEEK_API_BASE_URL),
            ): str,
        }
    )

DEFAULT_OPTIONS = {
    CONF_LLM_HASS_API: [llm.LLM_API_ASSIST],
    CONF_PROMPT: DEFAULT_SYSTEM_PROMPT,
    CONF_CHAT_MODEL: RECOMMENDED_CHAT_MODEL,
    CONF_MAX_TOKENS: RECOMMENDED_MAX_TOKENS,
    CONF_TEMPERATURE: RECOMMENDED_TEMPERATURE,
    CONF_TOP_P: RECOMMENDED_TOP_P,
    CONF_THINKING_ENABLED: DEFAULT_THINKING_ENABLED,
    CONF_STRIP_MARKDOWN: DEFAULT_STRIP_MARKDOWN,
    CONF_REASONING_EFFORT: RECOMMENDED_REASONING_EFFORT,
}


_PROBE_TIMEOUT = 10.0


async def async_probe_deepseek_client(client: openai.AsyncOpenAI) -> None:
    """Validate credentials via ``models.list()`` without using completion quota.

    OpenAI-compatible gateways without ``/models`` (404/405/501) are skipped so the
    first real chat call can surface auth issues. Used by config_flow and __init__.
    """
    try:
        await client.with_options(timeout=_PROBE_TIMEOUT).models.list()
    except openai.APIStatusError as err:
        if err.status_code in (404, 405, 501):
            LOGGER.debug(
                "DeepSeek base URL does not implement /models (%s); skipping probe",
                err.status_code,
            )
            return
        raise


async def validate_input(hass: HomeAssistant, data: dict[str, Any]) -> None:
    """Validate the user input allows us to connect."""
    base_url = data.get(CONF_BASE_URL, DEEPSEEK_API_BASE_URL)
    if base_url:
        base_url = base_url.strip()
    if not base_url:
        base_url = DEEPSEEK_API_BASE_URL

    client = openai.AsyncOpenAI(
        api_key=data[CONF_API_KEY],
        base_url=base_url,
        http_client=get_async_client(hass),
    )
    try:
        await async_probe_deepseek_client(client)
    finally:
        await client.close()


class DeepSeekConfigFlow(ConfigFlow, domain=DOMAIN):
    """Handle a config flow for DeepSeek Conversation."""

    VERSION = 2

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Handle the initial step."""
        if user_input is None:
            return self.async_show_form(
                step_id="user", data_schema=get_user_step_schema()
            )

        errors: dict[str, str] = {}

        try:
            await validate_input(self.hass, user_input)
        except openai.APIConnectionError:
            errors["base"] = "cannot_connect"
        except openai.AuthenticationError:
            errors["base"] = "invalid_auth"
        except openai.APIStatusError as err:
            if err.status_code in (401, 403):
                errors["base"] = "invalid_auth"
            else:
                LOGGER.error("DeepSeek API status error during validation: %s", err)
                errors["base"] = "api_error"
        except openai.OpenAIError as e:
            LOGGER.error("DeepSeek API error during validation: %s", e)
            errors["base"] = "api_error"
        except Exception:
            LOGGER.exception("Unexpected exception during validation")
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
            step_id="user", data_schema=get_user_step_schema(), errors=errors
        )

    async def async_step_reauth(
        self, entry_data: Mapping[str, Any]
    ) -> ConfigFlowResult:
        """Perform reauth upon an API authentication error."""
        return await self.async_step_reauth_confirm()

    async def async_step_reauth_confirm(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Dialog that informs the user that reauth is required."""
        errors: dict[str, str] = {}
        reauth_entry = self._get_reauth_entry()

        if user_input is not None:
            validate_data = {
                CONF_API_KEY: user_input[CONF_API_KEY],
                CONF_BASE_URL: reauth_entry.data.get(
                    CONF_BASE_URL, DEEPSEEK_API_BASE_URL
                ),
            }
            try:
                await validate_input(self.hass, validate_data)
            except openai.APIConnectionError:
                errors["base"] = "cannot_connect"
            except openai.AuthenticationError:
                errors["base"] = "invalid_auth"
            except openai.APIStatusError as err:
                if err.status_code in (401, 403):
                    errors["base"] = "invalid_auth"
                else:
                    LOGGER.error("DeepSeek API status error during reauth: %s", err)
                    errors["base"] = "api_error"
            except openai.OpenAIError as e:
                LOGGER.error("DeepSeek API error during reauth: %s", e)
                errors["base"] = "api_error"
            except Exception:
                LOGGER.exception("Unexpected exception during reauth")
                errors["base"] = "unknown"
            else:
                return self.async_update_reload_and_abort(
                    reauth_entry,
                    data_updates={CONF_API_KEY: user_input[CONF_API_KEY]},
                )

        return self.async_show_form(
            step_id="reauth_confirm",
            data_schema=STEP_REAUTH_DATA_SCHEMA,
            errors=errors,
        )

    async def async_step_reconfigure(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Update API key and base URL from the integration menu (⋮ → Reconfigure)."""
        errors: dict[str, str] = {}
        reconfigure_entry = self._get_reconfigure_entry()
        current_base_url = reconfigure_entry.data.get(
            CONF_BASE_URL, DEEPSEEK_API_BASE_URL
        )

        if user_input is not None:
            base_url = user_input.get(CONF_BASE_URL, current_base_url)
            if isinstance(base_url, str):
                base_url = base_url.strip()
            if not base_url:
                errors[CONF_BASE_URL] = "url_required"
            else:
                validate_data = {
                    CONF_API_KEY: user_input[CONF_API_KEY],
                    CONF_BASE_URL: base_url,
                }
                try:
                    await validate_input(self.hass, validate_data)
                except openai.APIConnectionError:
                    errors["base"] = "cannot_connect"
                except openai.AuthenticationError:
                    errors["base"] = "invalid_auth"
                except openai.APIStatusError as err:
                    if err.status_code in (401, 403):
                        errors["base"] = "invalid_auth"
                    else:
                        LOGGER.error(
                            "DeepSeek API status error during reconfigure: %s", err
                        )
                        errors["base"] = "api_error"
                except openai.OpenAIError as e:
                    LOGGER.error("DeepSeek API error during reconfigure: %s", e)
                    errors["base"] = "api_error"
                except Exception:
                    LOGGER.exception("Unexpected exception during reconfigure")
                    errors["base"] = "unknown"
                else:
                    LOGGER.debug(
                        "[Debug config_flow]: reconfigure successful, reloading entry"
                    )
                    return self.async_update_reload_and_abort(
                        reconfigure_entry,
                        data_updates={
                            CONF_API_KEY: user_input[CONF_API_KEY],
                            CONF_BASE_URL: base_url,
                        },
                    )

        return self.async_show_form(
            step_id="reconfigure",
            data_schema=get_reconfigure_step_schema(reconfigure_entry),
            errors=errors,
        )

    @staticmethod
    def async_get_options_flow(
        config_entry: ConfigEntry,
    ) -> OptionsFlow:
        """Create the options flow."""
        return DeepSeekOptionsFlow()


class DeepSeekOptionsFlow(OptionsFlow):
    """DeepSeek config flow options handler."""

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Options entry menu: Assist settings or open reconfigure flow."""
        return self.async_show_menu(
            step_id="init",
            menu_options=["assist", "reconfigure"],
        )

    async def async_step_reconfigure(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Close options and open the config-flow reconfigure dialog."""
        try:
            config_entry = self.config_entry
        except AttributeError:
            LOGGER.error("config_entry not available in OptionsFlow")
            return self.async_abort(reason="config_entry_not_available")

        LOGGER.debug(
            "[Debug config_flow]: options menu requested reconfigure for entry %s",
            config_entry.entry_id,
        )
        self.hass.async_create_task(
            self.hass.config_entries.flow.async_init(
                DOMAIN,
                context={
                    "source": SOURCE_RECONFIGURE,
                    "entry_id": config_entry.entry_id,
                },
            ),
            eager_start=True,
        )
        return self.async_abort(reason="reconfigure_started")

    async def async_step_assist(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Manage Assist and model options."""
        # self.config_entry should be automatically available as a property
        # Access it safely to avoid AttributeError
        try:
            config_entry = self.config_entry
        except AttributeError:
            LOGGER.error("config_entry not available in OptionsFlow")
            return self.async_abort(reason="config_entry_not_available")

        errors: dict[str, str] = {}

        if user_input is not None:
            # Handle CONF_LLM_HASS_API selection
            normalized = _normalize_llm_hass_api(user_input.get(CONF_LLM_HASS_API))
            if normalized is None:
                user_input.pop(CONF_LLM_HASS_API, None)
            else:
                user_input[CONF_LLM_HASS_API] = normalized

            # Handle base URL update - move it from options to data if changed
            base_url_changed = False
            if CONF_BASE_URL in user_input:
                base_url = user_input.pop(CONF_BASE_URL).strip()
                # Validate URL is not empty
                if not base_url:
                    errors[CONF_BASE_URL] = "url_required"
                else:
                    # Normalize URL (ensure it ends with /v1 or similar, or let user specify full path)
                    if base_url != config_entry.data.get(CONF_BASE_URL, DEEPSEEK_API_BASE_URL):
                        base_url_changed = True
                        # Update the config entry data
                        new_data = {**config_entry.data, CONF_BASE_URL: base_url}
                        self.hass.config_entries.async_update_entry(config_entry, data=new_data)

            if not errors:
                updated_options = {**config_entry.options, **user_input}
                result = self.async_create_entry(title="", data=updated_options)

                # Connection change: recreate OpenAI client (models.list probe in setup)
                if base_url_changed:
                    LOGGER.debug(
                        "[Debug config_flow]: base_url changed, scheduling config entry reload"
                    )
                    self.hass.async_create_task(
                        self.hass.config_entries.async_reload(config_entry.entry_id)
                    )

                return result

        schema = deepseek_config_option_schema(
            self.hass,
            config_entry.options,
            config_entry,
        )
        suggested = dict(config_entry.options)
        if user_input is not None:
            suggested.update(user_input)
        return self.async_show_form(
            step_id="assist",
            data_schema=self.add_suggested_values_to_schema(
                vol.Schema(schema), suggested
            ),
            errors=errors,
        )


def deepseek_config_option_schema(
    hass: HomeAssistant,
    options: dict[str, Any] | MappingProxyType[str, Any],
    config_entry: ConfigEntry | None = None,
) -> VolDictType:
    """Return a schema for DeepSeek completion options.

    All fields stay visible regardless of the reasoning toggle. The API layer in
    ``build_chat_completion_args()`` decides what is actually sent.
    """
    # Re-add HASS API selection
    hass_apis: list[SelectOptionDict] = [
        SelectOptionDict(label=api.name, value=api.id)
        for api in llm.async_get_apis(hass)
    ]

    # Get base URL from config entry data if available, otherwise from options or default
    base_url = DEEPSEEK_API_BASE_URL
    if config_entry:
        base_url = config_entry.data.get(CONF_BASE_URL, DEEPSEEK_API_BASE_URL)
    elif CONF_BASE_URL in options:
        base_url = options[CONF_BASE_URL]

    reasoning_effort_options = [
        SelectOptionDict(label=value, value=value)
        for value, _ in REASONING_EFFORT_SELECT
    ]

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
                    CONF_PROMPT, DEFAULT_SYSTEM_PROMPT
                )
            },
            default=DEFAULT_SYSTEM_PROMPT,
        ): TemplateSelector(),
        # Add selector for CONF_LLM_HASS_API
        vol.Optional(
            CONF_LLM_HASS_API,
            description={"suggested_value": _normalize_llm_hass_api(options.get(CONF_LLM_HASS_API))},
            default=_normalize_llm_hass_api(options.get(CONF_LLM_HASS_API)),
        ): SelectSelector(SelectSelectorConfig(options=hass_apis, multiple=True)),
        vol.Optional(
            CONF_CHAT_MODEL,
            description={"suggested_value": options.get(CONF_CHAT_MODEL)},
            default=options.get(CONF_CHAT_MODEL, RECOMMENDED_CHAT_MODEL),
        ): _chat_model_selector(),
        vol.Optional(
            CONF_MAX_TOKENS,
            description={"suggested_value": options.get(CONF_MAX_TOKENS)},
            default=coerce_max_tokens(
                options.get(CONF_MAX_TOKENS, RECOMMENDED_MAX_TOKENS)
            ),
        ): NumberSelector(
            NumberSelectorConfig(min=1, max=MAX_TOKENS_UPPER_BOUND, mode="box", step=1)
        ),
        vol.Optional(
            CONF_THINKING_ENABLED,
            description={
                "suggested_value": options.get(
                    CONF_THINKING_ENABLED, DEFAULT_THINKING_ENABLED
                )
            },
            default=options.get(CONF_THINKING_ENABLED, DEFAULT_THINKING_ENABLED),
        ): BooleanSelector(),
        vol.Optional(
            CONF_REASONING_EFFORT,
            description={
                "suggested_value": options.get(
                    CONF_REASONING_EFFORT, RECOMMENDED_REASONING_EFFORT
                )
            },
            default=options.get(
                CONF_REASONING_EFFORT, RECOMMENDED_REASONING_EFFORT
            ),
        ): SelectSelector(
            SelectSelectorConfig(
                options=reasoning_effort_options,
                translation_key=CONF_REASONING_EFFORT,
            )
        ),
        vol.Optional(
            CONF_TOP_P,
            description={"suggested_value": options.get(CONF_TOP_P)},
            default=RECOMMENDED_TOP_P,
        ): NumberSelector(
            NumberSelectorConfig(min=0, max=1, step=0.05, mode="slider")
        ),
        vol.Optional(
            CONF_TEMPERATURE,
            description={"suggested_value": options.get(CONF_TEMPERATURE)},
            default=RECOMMENDED_TEMPERATURE,
        ): NumberSelector(
            NumberSelectorConfig(min=0, max=2, step=0.05, mode="slider")
        ),
        vol.Optional(
            CONF_STRIP_MARKDOWN,
            description={
                "suggested_value": options.get(
                    CONF_STRIP_MARKDOWN, DEFAULT_STRIP_MARKDOWN
                )
            },
            default=options.get(CONF_STRIP_MARKDOWN, DEFAULT_STRIP_MARKDOWN),
        ): BooleanSelector(),
    }

    return schema
