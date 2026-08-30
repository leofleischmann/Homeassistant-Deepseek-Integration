"""Config flow for DeepSeek Conversation integration.

Flow control only: which step follows which, and what to do with the answers.
The forms themselves are in ``flow_schemas.py``, and reaching the API to check
a set of credentials is ``client.py``'s job.
"""

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
from homeassistant.helpers.selector import (  # pyright: ignore[reportMissingImports]
    BooleanSelector,
    SelectOptionDict,
    SelectSelector,
    SelectSelectorConfig,
    TemplateSelector,
)
from homeassistant.helpers.typing import VolDictType  # pyright: ignore[reportMissingImports]

from .client import async_validate_credentials
from .const import (
    CONF_BASE_URL,
    CONF_BRAVE_API_KEY,
    CONF_CHAT_MODEL,
    CONF_PROMPT,
    CONF_RECOMMENDED,
    DEEPSEEK_API_BASE_URL,
    DEFAULT_AI_TASK_NAME,
    DEFAULT_CONVERSATION_NAME,
    DOMAIN,
    LOGGER,
    RECOMMENDED_AI_TASK_OPTIONS,
    RECOMMENDED_CONVERSATION_OPTIONS,
    SUBENTRY_TYPE_AI_TASK,
    SUBENTRY_TYPE_CONVERSATION,
)
from .flow_schemas import (
    advanced_field,
    ADVANCED_SECTIONS,
    chat_model_selector,
    flatten_sections,
    get_reconfigure_step_schema,
    get_user_step_schema,
    normalize_llm_hass_api,
    SECTION_CONVERSATION,
    STEP_REAUTH_DATA_SCHEMA,
)
from .models import is_retired_chat_model
from .options import recommended_agent_options


async def _async_check_credentials(
    hass: HomeAssistant, data: dict[str, Any], *, context: str
) -> dict[str, str]:
    """Check a key and base URL, and return the form errors - empty if they work.

    Initial setup, reauth and reconfigure all ask the same question and turn the
    same failures into the same error keys, so they share this. ``context`` only
    names the step in the log line.
    """
    try:
        await async_validate_credentials(hass, data)
    except openai.APIConnectionError:
        return {"base": "cannot_connect"}
    except openai.AuthenticationError:
        return {"base": "invalid_auth"}
    except openai.APIStatusError as err:
        if err.status_code in (401, 403):
            return {"base": "invalid_auth"}
        LOGGER.error("DeepSeek API status error during %s: %s", context, err)
        return {"base": "api_error"}
    except openai.OpenAIError as err:
        LOGGER.error("DeepSeek API error during %s: %s", context, err)
        return {"base": "api_error"}
    except Exception:
        LOGGER.exception("Unexpected exception during %s", context)
        return {"base": "unknown"}
    return {}


async def async_validate_reconfigure_input(
    hass: HomeAssistant,
    user_input: dict[str, Any],
    *,
    current_base_url: str,
) -> tuple[dict[str, str], dict[str, Any] | None]:
    """Validate API key and base URL for reconfigure (config or options flow)."""
    base_url = user_input.get(CONF_BASE_URL, current_base_url)
    if isinstance(base_url, str):
        base_url = base_url.strip()
    if not base_url:
        return {CONF_BASE_URL: "url_required"}, None

    validate_data = {
        CONF_API_KEY: user_input[CONF_API_KEY],
        CONF_BASE_URL: base_url,
    }
    if errors := await _async_check_credentials(
        hass, validate_data, context="reconfigure"
    ):
        return errors, None
    return {}, validate_data


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

        if is_retired_chat_model(
            user_input.get(CONF_CHAT_MODEL),
            base_url=user_input.get(CONF_BASE_URL, DEEPSEEK_API_BASE_URL),
        ):
            return self._async_show_user_form(
                user_input, {CONF_CHAT_MODEL: "model_retired"}
            )

        if errors := await _async_check_credentials(
            self.hass, user_input, context="validation"
        ):
            return self._async_show_user_form(user_input, errors)

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
            errors = await _async_check_credentials(
                self.hass, validate_data, context="reauth"
            )
            if not errors:
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
        if selected := normalize_llm_hass_api(options.get(CONF_LLM_HASS_API)):
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
                normalized = normalize_llm_hass_api(user_input.get(CONF_LLM_HASS_API))
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
            self.options.update(flatten_sections(user_input))
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
                    dict(advanced_field(key, self.options) for key in keys)
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
                vol.Optional(CONF_CHAT_MODEL): chat_model_selector(),
                vol.Required(CONF_RECOMMENDED, default=True): BooleanSelector(),
            }
        )
        return schema
