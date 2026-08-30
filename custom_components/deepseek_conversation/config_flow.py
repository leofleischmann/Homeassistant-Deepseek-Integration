"""Config flow for DeepSeek Conversation integration."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import openai
import voluptuous as vol  # pyright: ignore[reportMissingImports]

from homeassistant.config_entries import (  # pyright: ignore[reportMissingImports]
    ConfigEntry,
    ConfigEntryState,
    ConfigFlow,
    ConfigFlowResult,
    ConfigSubentryFlow,
    SubentryFlowResult,
)
from homeassistant.data_entry_flow import section  # pyright: ignore[reportMissingImports]
from homeassistant.const import (  # pyright: ignore[reportMissingImports]
    CONF_API_KEY,
    CONF_LLM_HASS_API,
    CONF_NAME,
)
from homeassistant.core import callback, HomeAssistant  # pyright: ignore[reportMissingImports]
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
    TextSelector,
    TextSelectorConfig,
    TextSelectorType,
)
from homeassistant.helpers.typing import VolDictType  # pyright: ignore[reportMissingImports]

from .client import async_probe_deepseek_client
from .const import (
    CHAT_MODEL_OPTIONS,
    CONF_BASE_URL,
    CONF_BRAVE_API_KEY,
    CONF_CHAT_MODEL,
    CONF_INCLUDE_USER_CONTEXT,
    CONF_MAX_TOOL_RESULT_CHARS,
    CONF_MAX_HISTORY_ROUNDS,
    CONF_MAX_TOKENS,
    CONF_MAX_TOOL_ITERATIONS,
    CONF_PROMPT,
    CONF_REASONING_EFFORT,
    CONF_RECOMMENDED,
    CONF_REQUEST_TIMEOUT,
    CONF_STRIP_MARKDOWN,
    CONF_TEMPERATURE,
    CONF_THINKING_ENABLED,
    CONF_TOP_P,
    CONF_VISION_ENABLED,
    DEFAULT_AI_TASK_NAME,
    DEFAULT_CONVERSATION_NAME,
    DEFAULT_INCLUDE_USER_CONTEXT,
    DEFAULT_VISION_ENABLED,
    DEFAULT_STRIP_MARKDOWN,
    DEFAULT_THINKING_ENABLED,
    DEEPSEEK_API_BASE_URL,
    DOMAIN,
    LOGGER,
    MAX_HISTORY_ROUNDS_UPPER_BOUND,
    MAX_TOKENS_UPPER_BOUND,
    MAX_TOOL_ITERATIONS_UPPER_BOUND,
    MAX_TOOL_RESULT_CHARS_UPPER_BOUND,
    REASONING_EFFORT_SELECT,
    RECOMMENDED_AI_TASK_OPTIONS,
    RECOMMENDED_CHAT_MODEL,
    RECOMMENDED_CONVERSATION_OPTIONS,
    RECOMMENDED_MAX_HISTORY_ROUNDS,
    RECOMMENDED_MAX_TOKENS,
    RECOMMENDED_MAX_TOOL_ITERATIONS,
    RECOMMENDED_MAX_TOOL_RESULT_CHARS,
    RECOMMENDED_REASONING_EFFORT,
    RECOMMENDED_REQUEST_TIMEOUT,
    RECOMMENDED_TEMPERATURE,
    RECOMMENDED_TOP_P,
    REQUEST_TIMEOUT_LOWER_BOUND,
    REQUEST_TIMEOUT_UPPER_BOUND,
    SUBENTRY_TYPE_AI_TASK,
    SUBENTRY_TYPE_CONVERSATION,
)
from .models import is_retired_chat_model
from .options import (
    coerce_max_history_rounds,
    coerce_max_tokens,
    coerce_max_tool_iterations,
    coerce_max_tool_result_chars,
    coerce_request_timeout,
    recommended_agent_options,
)

#: The advanced step, grouped. Order and collapsed state follow how often a
#: setting is actually touched: the way an agent answers is open, the rest is
#: folded away until someone goes looking for it.
SECTION_RESPONSE = "response"
SECTION_TOOLS = "tools"
SECTION_CONVERSATION = "conversation"
SECTION_LIMITS = "limits"

ADVANCED_SECTIONS: tuple[tuple[str, tuple[str, ...], bool], ...] = (
    (
        SECTION_RESPONSE,
        (
            CONF_MAX_TOKENS,
            CONF_TEMPERATURE,
            CONF_TOP_P,
            CONF_THINKING_ENABLED,
            CONF_REASONING_EFFORT,
        ),
        False,
    ),
    (SECTION_TOOLS, (CONF_MAX_TOOL_ITERATIONS, CONF_MAX_TOOL_RESULT_CHARS), True),
    (
        SECTION_CONVERSATION,
        (CONF_STRIP_MARKDOWN, CONF_INCLUDE_USER_CONTEXT, CONF_MAX_HISTORY_ROUNDS),
        True,
    ),
    (SECTION_LIMITS, (CONF_REQUEST_TIMEOUT, CONF_VISION_ENABLED), True),
)


def _advanced_field(key: str, options: Mapping[str, Any]) -> tuple[Any, Any]:
    """Return the (marker, selector) pair for one advanced setting."""
    if key == CONF_MAX_TOKENS:
        return (
            vol.Optional(
                key,
                default=coerce_max_tokens(options.get(key, RECOMMENDED_MAX_TOKENS)),
            ),
            NumberSelector(
                NumberSelectorConfig(
                    min=1, max=MAX_TOKENS_UPPER_BOUND, mode="box", step=1
                )
            ),
        )
    if key == CONF_TEMPERATURE:
        return (
            vol.Optional(key, default=RECOMMENDED_TEMPERATURE),
            NumberSelector(
                NumberSelectorConfig(min=0, max=2, step=0.05, mode="slider")
            ),
        )
    if key == CONF_TOP_P:
        return (
            vol.Optional(key, default=RECOMMENDED_TOP_P),
            NumberSelector(
                NumberSelectorConfig(min=0, max=1, step=0.05, mode="slider")
            ),
        )
    if key == CONF_THINKING_ENABLED:
        return (
            vol.Optional(
                key, default=options.get(key, DEFAULT_THINKING_ENABLED)
            ),
            BooleanSelector(),
        )
    if key == CONF_REASONING_EFFORT:
        return (
            vol.Optional(
                key, default=options.get(key, RECOMMENDED_REASONING_EFFORT)
            ),
            SelectSelector(
                SelectSelectorConfig(
                    options=[
                        SelectOptionDict(label=value, value=value)
                        for value, _ in REASONING_EFFORT_SELECT
                    ],
                    translation_key=CONF_REASONING_EFFORT,
                )
            ),
        )
    if key == CONF_MAX_TOOL_ITERATIONS:
        return (
            vol.Optional(
                key,
                default=coerce_max_tool_iterations(
                    options.get(key, RECOMMENDED_MAX_TOOL_ITERATIONS)
                ),
            ),
            NumberSelector(
                NumberSelectorConfig(
                    min=1, max=MAX_TOOL_ITERATIONS_UPPER_BOUND, mode="box", step=1
                )
            ),
        )
    if key == CONF_MAX_TOOL_RESULT_CHARS:
        return (
            vol.Optional(
                key,
                default=coerce_max_tool_result_chars(
                    options.get(key, RECOMMENDED_MAX_TOOL_RESULT_CHARS)
                ),
            ),
            NumberSelector(
                NumberSelectorConfig(
                    min=0,
                    max=MAX_TOOL_RESULT_CHARS_UPPER_BOUND,
                    mode="box",
                    step=500,
                )
            ),
        )
    if key == CONF_MAX_HISTORY_ROUNDS:
        return (
            vol.Optional(
                key,
                default=coerce_max_history_rounds(
                    options.get(key, RECOMMENDED_MAX_HISTORY_ROUNDS)
                ),
            ),
            NumberSelector(
                NumberSelectorConfig(
                    min=0, max=MAX_HISTORY_ROUNDS_UPPER_BOUND, mode="box", step=1
                )
            ),
        )
    if key == CONF_REQUEST_TIMEOUT:
        return (
            vol.Optional(
                key,
                default=coerce_request_timeout(
                    options.get(key, RECOMMENDED_REQUEST_TIMEOUT)
                ),
            ),
            NumberSelector(
                NumberSelectorConfig(
                    min=REQUEST_TIMEOUT_LOWER_BOUND,
                    max=REQUEST_TIMEOUT_UPPER_BOUND,
                    mode="box",
                    step=5,
                    unit_of_measurement="s",
                )
            ),
        )
    if key == CONF_STRIP_MARKDOWN:
        return (
            vol.Optional(key, default=options.get(key, DEFAULT_STRIP_MARKDOWN)),
            BooleanSelector(),
        )
    if key == CONF_INCLUDE_USER_CONTEXT:
        return (
            vol.Optional(
                key, default=options.get(key, DEFAULT_INCLUDE_USER_CONTEXT)
            ),
            BooleanSelector(),
        )
    if key == CONF_VISION_ENABLED:
        return (
            vol.Optional(key, default=options.get(key, DEFAULT_VISION_ENABLED)),
            BooleanSelector(),
        )
    raise ValueError(f"no selector defined for {key}")


def _flatten_sections(user_input: dict[str, Any]) -> dict[str, Any]:
    """Undo the nesting a sectioned form returns, so a subentry stays flat."""
    flat: dict[str, Any] = {}
    for value in user_input.values():
        if isinstance(value, dict):
            flat.update(value)
    return flat


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


def _api_key_selector() -> TextSelector:
    return TextSelector(
        TextSelectorConfig(
            type=TextSelectorType.PASSWORD,
            autocomplete="current-password",
        )
    )


def _base_url_selector() -> TextSelector:
    return TextSelector(
        TextSelectorConfig(
            type=TextSelectorType.URL,
            autocomplete="url",
        )
    )


def get_user_step_schema() -> vol.Schema:
    """Schema for initial config (API key, URL, model, optional Brave key)."""
    return vol.Schema(
        {
            vol.Required(CONF_API_KEY): _api_key_selector(),
            vol.Optional(CONF_BASE_URL, default=DEEPSEEK_API_BASE_URL): _base_url_selector(),
            vol.Optional(CONF_BRAVE_API_KEY): _api_key_selector(),
            vol.Optional(
                CONF_CHAT_MODEL, default=RECOMMENDED_CHAT_MODEL
            ): _chat_model_selector(),
        }
    )


STEP_REAUTH_DATA_SCHEMA = vol.Schema(
    {
        vol.Required(CONF_API_KEY): _api_key_selector(),
    }
)


def get_reconfigure_step_schema(entry: ConfigEntry) -> vol.Schema:
    """Schema for reconfigure (DeepSeek key, base URL, optional Brave key).

    Brave key: leave empty to keep the current key; enter ``-`` to remove it
    (clears web search LLM API registration after reload).
    """
    return vol.Schema(
        {
            vol.Required(CONF_API_KEY): _api_key_selector(),
            vol.Optional(
                CONF_BASE_URL,
                default=entry.data.get(CONF_BASE_URL, DEEPSEEK_API_BASE_URL),
            ): _base_url_selector(),
            vol.Optional(CONF_BRAVE_API_KEY): _api_key_selector(),
        }
    )


async def validate_input(hass: HomeAssistant, data: dict[str, Any]) -> None:
    """Validate the user input allows us to connect."""
    base_url = data.get(CONF_BASE_URL, DEEPSEEK_API_BASE_URL)
    if base_url:
        base_url = base_url.strip()
    if not base_url:
        base_url = DEEPSEEK_API_BASE_URL

    # The OpenAI client wraps Home Assistant's shared httpx client, which HA owns
    # and closes on shutdown; closing it here would only trigger a framework
    # warning without releasing anything, so the client is left for GC.
    client = openai.AsyncOpenAI(
        api_key=data[CONF_API_KEY],
        base_url=base_url,
        http_client=get_async_client(hass),
    )
    await async_probe_deepseek_client(client)


async def async_validate_reconfigure_input(
    hass: HomeAssistant,
    user_input: dict[str, Any],
    *,
    current_base_url: str,
) -> tuple[dict[str, str], dict[str, Any] | None]:
    """Validate API key and base URL for reconfigure (config or options flow)."""
    errors: dict[str, str] = {}
    base_url = user_input.get(CONF_BASE_URL, current_base_url)
    if isinstance(base_url, str):
        base_url = base_url.strip()
    if not base_url:
        errors[CONF_BASE_URL] = "url_required"
        return errors, None

    validate_data = {
        CONF_API_KEY: user_input[CONF_API_KEY],
        CONF_BASE_URL: base_url,
    }
    try:
        await validate_input(hass, validate_data)
    except openai.APIConnectionError:
        errors["base"] = "cannot_connect"
    except openai.AuthenticationError:
        errors["base"] = "invalid_auth"
    except openai.APIStatusError as err:
        if err.status_code in (401, 403):
            errors["base"] = "invalid_auth"
        else:
            LOGGER.error("DeepSeek API status error during reconfigure: %s", err)
            errors["base"] = "api_error"
    except openai.OpenAIError as e:
        LOGGER.error("DeepSeek API error during reconfigure: %s", e)
        errors["base"] = "api_error"
    except Exception:
        LOGGER.exception("Unexpected exception during reconfigure")
        errors["base"] = "unknown"
    else:
        return {}, {
            CONF_API_KEY: user_input[CONF_API_KEY],
            CONF_BASE_URL: base_url,
        }

    return errors, None


class DeepSeekConfigFlow(ConfigFlow, domain=DOMAIN):
    """Handle a config flow for DeepSeek Conversation."""

    VERSION = 3

    def _async_update_entry_and_abort(
        self, entry: ConfigEntry, **kwargs: Any
    ) -> ConfigFlowResult:
        """Update entry data; reload is owned by the conversation update listener.

        Deliberately not ``async_update_reload_and_abort``: combining that with an
        update listener is what Home Assistant warns breaks in 2026.12.
        """
        return self.async_update_and_abort(entry, **kwargs)

    def _async_show_user_form(
        self,
        user_input: dict[str, Any] | None = None,
        errors: dict[str, str] | None = None,
    ) -> ConfigFlowResult:
        """Render the setup form, keeping whatever the user already typed.

        Without the suggested values a rejected key or model empties every field
        and the whole form has to be filled in again.
        """
        schema = get_user_step_schema()
        if user_input:
            schema = self.add_suggested_values_to_schema(schema, user_input)
        return self.async_show_form(
            step_id="user", data_schema=schema, errors=errors
        )

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Handle the initial step."""
        if user_input is None:
            return self._async_show_user_form()

        errors: dict[str, str] = {}

        if is_retired_chat_model(
            user_input.get(CONF_CHAT_MODEL),
            base_url=user_input.get(CONF_BASE_URL, DEEPSEEK_API_BASE_URL),
        ):
            errors[CONF_CHAT_MODEL] = "model_retired"
            return self._async_show_user_form(user_input, errors)

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
            brave_key = (user_input.get(CONF_BRAVE_API_KEY) or "").strip()
            if brave_key:
                entry_data[CONF_BRAVE_API_KEY] = brave_key
                LOGGER.debug(
                    "[Debug config_flow]: Brave Search key set on initial setup"
                )
            # The model picked here seeds both agents; each can be changed
            # afterwards, and more agents added, without touching the key.
            conversation_options = {**RECOMMENDED_CONVERSATION_OPTIONS}
            ai_task_options = {**RECOMMENDED_AI_TASK_OPTIONS}
            if model := user_input.get(CONF_CHAT_MODEL):
                conversation_options[CONF_CHAT_MODEL] = model
                ai_task_options[CONF_CHAT_MODEL] = model

            return self.async_create_entry(
                title="DeepSeek",
                data=entry_data,
                subentries=[
                    {
                        "subentry_type": SUBENTRY_TYPE_CONVERSATION,
                        "data": conversation_options,
                        "title": DEFAULT_CONVERSATION_NAME,
                        "unique_id": None,
                    },
                    {
                        "subentry_type": SUBENTRY_TYPE_AI_TASK,
                        "data": ai_task_options,
                        "title": DEFAULT_AI_TASK_NAME,
                        "unique_id": None,
                    },
                ],
            )

        return self._async_show_user_form(user_input, errors)

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
                return self._async_update_entry_and_abort(
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
        reconfigure_entry = self._get_reconfigure_entry()
        current_base_url = reconfigure_entry.data.get(
            CONF_BASE_URL, DEEPSEEK_API_BASE_URL
        )

        if user_input is not None:
            errors, data_updates = await async_validate_reconfigure_input(
                self.hass,
                user_input,
                current_base_url=current_base_url,
            )
            if data_updates is not None:
                brave_raw = (user_input.get(CONF_BRAVE_API_KEY) or "").strip()
                if brave_raw == "-":
                    # Full data replace to drop Brave key (data_updates alone only merges).
                    new_data = {**reconfigure_entry.data, **data_updates}
                    new_data.pop(CONF_BRAVE_API_KEY, None)
                    LOGGER.debug(
                        "[Debug config_flow]: reconfigure removing Brave Search key"
                    )
                    # Update only; conversation update listener schedules reload
                    # when entry.data changes (avoids HA 2026.12 listener+reload warning).
                    return self._async_update_entry_and_abort(
                        reconfigure_entry,
                        data=new_data,
                    )
                if brave_raw:
                    data_updates[CONF_BRAVE_API_KEY] = brave_raw
                    LOGGER.debug(
                        "[Debug config_flow]: reconfigure updating Brave Search key"
                    )
                # Empty Brave field: keep existing key (do not include in data_updates).
                LOGGER.debug(
                    "[Debug config_flow]: reconfigure successful; listener will reload"
                )
                return self._async_update_entry_and_abort(
                    reconfigure_entry,
                    data_updates=data_updates,
                )
        else:
            errors = {}

        return self.async_show_form(
            step_id="reconfigure",
            data_schema=get_reconfigure_step_schema(reconfigure_entry),
            errors=errors,
        )

    @classmethod
    @callback
    def async_get_supported_subentry_types(
        cls, config_entry: ConfigEntry
    ) -> dict[str, type[ConfigSubentryFlow]]:
        """Agents that can be added to this entry, both driven by one flow."""
        return {
            SUBENTRY_TYPE_CONVERSATION: DeepSeekSubentryFlowHandler,
            SUBENTRY_TYPE_AI_TASK: DeepSeekSubentryFlowHandler,
        }


class DeepSeekSubentryFlowHandler(ConfigSubentryFlow):
    """Add or reconfigure one agent on a DeepSeek config entry.

    Two steps: the first asks what the agent is for, the second only opens when
    the recommended settings are switched off. An agent left on the recommended
    settings stores just those first answers - everything else resolves to the
    defaults in const.py as it is read, so later changes to a default reach
    agents that never overrode it.
    """

    options: dict[str, Any]
    _name: str | None = None

    @property
    def _is_new(self) -> bool:
        """Whether this flow is adding an agent rather than editing one."""
        return self.source == "user"

    @property
    def _is_conversation(self) -> bool:
        """Whether this agent talks to people rather than to an automation."""
        return self._subentry_type == SUBENTRY_TYPE_CONVERSATION

    def _base_url(self) -> str:
        return self._get_entry().data.get(CONF_BASE_URL, DEEPSEEK_API_BASE_URL)

    def _default_name(self) -> str:
        return (
            DEFAULT_CONVERSATION_NAME
            if self._is_conversation
            else DEFAULT_AI_TASK_NAME
        )

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Add an agent."""
        self.options = dict(
            RECOMMENDED_CONVERSATION_OPTIONS
            if self._is_conversation
            else RECOMMENDED_AI_TASK_OPTIONS
        )
        return await self.async_step_init()

    async def async_step_reconfigure(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Edit an existing agent."""
        self.options = dict(self._get_reconfigure_subentry().data)
        return await self.async_step_init()

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Ask what the agent is for: prompt, tools and model."""
        if self._get_entry().state is not ConfigEntryState.LOADED:
            return self.async_abort(reason="entry_not_loaded")

        options = self.options
        errors: dict[str, str] = {}

        # Drop APIs that no longer exist - web search after the Brave key was
        # removed, say - so the form does not offer a value it cannot save.
        available_apis = {api.id for api in llm.async_get_apis(self.hass)}
        if selected := _normalize_llm_hass_api(options.get(CONF_LLM_HASS_API)):
            options[CONF_LLM_HASS_API] = [
                api for api in selected if api in available_apis
            ]

        if user_input is not None:
            if is_retired_chat_model(
                user_input.get(CONF_CHAT_MODEL), base_url=self._base_url()
            ):
                errors[CONF_CHAT_MODEL] = "model_retired"
            else:
                self._name = user_input.pop(CONF_NAME, None)
                normalized = _normalize_llm_hass_api(user_input.get(CONF_LLM_HASS_API))
                user_input.pop(CONF_LLM_HASS_API, None)
                options.update(user_input)
                if normalized is None:
                    # No API selected means no control over the home.
                    options.pop(CONF_LLM_HASS_API, None)
                else:
                    options[CONF_LLM_HASS_API] = normalized

                if options.get(CONF_RECOMMENDED):
                    return self._async_save()
                return await self.async_step_advanced()

        return self.async_show_form(
            step_id="init",
            data_schema=self.add_suggested_values_to_schema(
                vol.Schema(self._init_schema()), options
            ),
            errors=errors,
        )

    async def async_step_advanced(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Ask for the settings the recommended defaults otherwise decide."""
        if user_input is not None:
            self.options.update(_flatten_sections(user_input))
            return self._async_save()

        return self.async_show_form(
            step_id="advanced",
            data_schema=self.add_suggested_values_to_schema(
                vol.Schema(self._advanced_schema()), self._advanced_suggested()
            ),
        )

    def _sections(self) -> tuple[tuple[str, tuple[str, ...], bool], ...]:
        """Return the sections this agent kind actually has."""
        return tuple(
            group
            for group in ADVANCED_SECTIONS
            if group[0] != SECTION_CONVERSATION or self._is_conversation
        )

    def _advanced_schema(self) -> VolDictType:
        """Second step, grouped so it reads as four short lists."""
        return {
            vol.Required(name): section(
                vol.Schema(
                    dict(_advanced_field(key, self.options) for key in keys)
                ),
                {"collapsed": collapsed},
            )
            for name, keys, collapsed in self._sections()
        }

    def _advanced_suggested(self) -> dict[str, dict[str, Any]]:
        """Suggested values, nested the way a sectioned form expects them."""
        return {
            name: {key: self.options[key] for key in keys if key in self.options}
            for name, keys, _collapsed in self._sections()
        }

    @callback
    def _async_save(self) -> SubentryFlowResult:
        """Create the agent, or write the changes back to an existing one."""
        if self.options.get(CONF_RECOMMENDED):
            # Back on the recommended settings: forget the overrides, or the
            # agent would keep running on values its form no longer shows.
            self.options = recommended_agent_options(self.options)
        if self._is_new:
            return self.async_create_entry(
                title=(self._name or "").strip() or self._default_name(),
                data=self.options,
            )
        return self.async_update_and_abort(
            self._get_entry(),
            self._get_reconfigure_subentry(),
            data=self.options,
        )

    def _init_schema(self) -> VolDictType:
        """First step: what this agent is for."""
        hass_apis: list[SelectOptionDict] = [
            SelectOptionDict(label=api.name, value=api.id)
            for api in llm.async_get_apis(self.hass)
        ]

        schema: VolDictType = {}
        if self._is_new:
            schema[vol.Required(CONF_NAME, default=self._default_name())] = str

        schema.update(
            {
                vol.Optional(CONF_PROMPT): TemplateSelector(),
                vol.Optional(CONF_LLM_HASS_API): SelectSelector(
                    SelectSelectorConfig(options=hass_apis, multiple=True)
                ),
                vol.Optional(CONF_CHAT_MODEL): _chat_model_selector(),
                vol.Required(CONF_RECOMMENDED, default=True): BooleanSelector(),
            }
        )
        return schema
