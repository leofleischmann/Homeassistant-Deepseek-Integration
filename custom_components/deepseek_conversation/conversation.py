"""Conversation support for DeepSeek: the Assist agent entity.

One entity per conversation subentry. Its job is what happens either side of
the API loop in ``chat_session.py``: work out who is talking and from where,
hand Home Assistant the prompt and the tools to expose, and turn the finished
chat log back into an ``IntentResponse``.
"""

from __future__ import annotations

from typing import Literal

from homeassistant.components import assist_pipeline, conversation  # pyright: ignore[reportMissingImports]
from homeassistant.config_entries import ConfigSubentry  # pyright: ignore[reportMissingImports]
from homeassistant.const import CONF_LLM_HASS_API, MATCH_ALL  # pyright: ignore[reportMissingImports]
from homeassistant.core import HomeAssistant  # pyright: ignore[reportMissingImports]
from homeassistant.exceptions import HomeAssistantError  # pyright: ignore[reportMissingImports]
from homeassistant.helpers import intent  # pyright: ignore[reportMissingImports]
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback  # pyright: ignore[reportMissingImports]

from .chat_messages import final_speech_from_chat_log
from .chat_session import async_handle_chat_log
from .const import (
    CONF_BASE_URL,
    CONF_INCLUDE_USER_CONTEXT,
    CONF_PROMPT,
    CONF_STRIP_MARKDOWN,
    CONF_THINKING_ENABLED,
    DEEPSEEK_API_BASE_URL,
    DEFAULT_INCLUDE_USER_CONTEXT,
    DEFAULT_STRIP_MARKDOWN,
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_THINKING_ENABLED,
    DOMAIN,
    LOGGER,
    SUBENTRY_TYPE_CONVERSATION,
)
from .markdown_strip import strip_markdown
from .types import (
    agent_device_info,
    agent_subentries,
    DeepSeekConfigEntry,
)
from .user_context import (
    async_build_speaker_context,
    merge_extra_system_prompt,
    strip_speaker_block,
)
from .vision import conversation_entity_features_for_options
from .agent_tools import agent_llm_api


def _intent_error_result(
    *,
    language: str,
    conversation_id: str,
    message: str,
    code: intent.IntentResponseErrorCode = intent.IntentResponseErrorCode.UNKNOWN,
) -> conversation.ConversationResult:
    """Build a ConversationResult that surfaces an error to the user.

    Centralised so the three failure paths in ``_async_handle_message`` - no
    client, context preparation, and anything the API loop raises - don't each
    rebuild an ``IntentResponse``.
    """
    intent_response = intent.IntentResponse(language=language)
    intent_response.async_set_error(code, message)
    return conversation.ConversationResult(
        response=intent_response, conversation_id=conversation_id
    )


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: DeepSeekConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up one conversation entity per conversation subentry."""
    for subentry in agent_subentries(config_entry, SUBENTRY_TYPE_CONVERSATION):
        async_add_entities(
            [DeepSeekConversationEntity(config_entry, subentry)],
            config_subentry_id=subentry.subentry_id,
        )


class DeepSeekConversationEntity(
    conversation.ConversationEntity, conversation.AbstractConversationAgent
):
    """DeepSeek conversation agent."""
    _attr_has_entity_name = True
    _attr_name = None
    _attr_supports_streaming = True

    def __init__(
        self, entry: DeepSeekConfigEntry, subentry: ConfigSubentry
    ) -> None:
        """Initialize one agent from its subentry."""
        self.entry = entry
        self.subentry = subentry
        self._attr_unique_id = subentry.subentry_id
        self._attr_device_info = agent_device_info(entry, subentry)
        options = subentry.data
        self._attr_supported_features = conversation_entity_features_for_options(
            options,
            has_control=bool(options.get(CONF_LLM_HASS_API)),
            base_url=entry.data.get(CONF_BASE_URL, DEEPSEEK_API_BASE_URL),
        )

    @property
    def supported_languages(self) -> list[str] | Literal["*"]:
        return MATCH_ALL

    async def async_added_to_hass(self) -> None:
        await super().async_added_to_hass()
        # async_migrate_engine may not be available in all Home Assistant versions
        if hasattr(assist_pipeline, 'async_migrate_engine'):
            try:
                assist_pipeline.async_migrate_engine(
                    self.hass, "conversation", self.entry.entry_id, self.entity_id
                )
            except Exception as e:
                LOGGER.warning("Failed to migrate assist pipeline engine: %s", e)
        conversation.async_set_agent(self.hass, self.entry, self)

    async def async_will_remove_from_hass(self) -> None:
        conversation.async_unset_agent(self.hass, self.entry)
        await super().async_will_remove_from_hass()

    async def _async_handle_message(
        self,
        user_input: conversation.ConversationInput,
        chat_log: conversation.ChatLog,
    ) -> conversation.ConversationResult:
        """Handle a message using DeepSeek."""
        options = self.subentry.data
        runtime = self.entry.runtime_data
        if runtime is None or runtime.client is None:
            LOGGER.error("DeepSeek client not available in runtime_data.")
            return _intent_error_result(
                language=user_input.language,
                conversation_id=chat_log.conversation_id,
                message="DeepSeek client not available",
                code=intent.IntentResponseErrorCode.FAILED_TO_HANDLE,
            )
        thinking_on = options.get(CONF_THINKING_ENABLED, DEFAULT_THINKING_ENABLED)

        # Who is talking, and from where. The preamble defines the template
        # variables; the facts block is appended last (after the exposed-entity
        # list) so the cacheable part of the system prompt stays speaker-agnostic.
        llm_context = user_input.as_llm_context(DOMAIN)
        speaker = await async_build_speaker_context(self.hass, llm_context)
        user_llm_prompt = (options.get(CONF_PROMPT) or "").strip() or DEFAULT_SYSTEM_PROMPT

        speaker_block = (
            speaker.facts_prompt()
            if options.get(CONF_INCLUDE_USER_CONTEXT, DEFAULT_INCLUDE_USER_CONTEXT)
            else None
        )
        # On follow-up turns ConversationInput carries no extra prompt and HA
        # falls back to the value it persisted last turn - which already had our
        # block appended. Recover the caller's own part before re-appending.
        caller_extra = user_input.extra_system_prompt
        if caller_extra is None:
            caller_extra = strip_speaker_block(chat_log.extra_system_prompt)

        LOGGER.debug(
            "[Debug conversation]: speaker user=%s area=%s facts=%s",
            speaker.has_user,
            speaker.has_location,
            bool(speaker_block),
        )

        try:
            await chat_log.async_provide_llm_data(
                llm_context=llm_context,
                user_llm_hass_api=agent_llm_api(
                    self.hass, self.entry, options, options.get(CONF_LLM_HASS_API)
                ),
                user_llm_prompt=speaker.apply_to_prompt(user_llm_prompt),
                user_extra_system_prompt=merge_extra_system_prompt(
                    caller_extra, speaker_block
                ),
            )
        except HomeAssistantError as err:
            # Not just ConverseError (which is one): Home Assistant wraps a
            # failing API id in that, but an API object we hand it is called
            # directly, so whatever it raises arrives here unwrapped.
            LOGGER.error("Error during chat_log.async_provide_llm_data: %s", err)
            return _intent_error_result(
                language=user_input.language,
                conversation_id=chat_log.conversation_id,
                message=f"Error preparing context: {err}",
                code=intent.IntentResponseErrorCode.FAILED_TO_HANDLE,
            )

        try:
            await async_handle_chat_log(
                self.hass,
                self.entry,
                chat_log,
                options=options,
                agent_id=user_input.agent_id,
                usage_source="assist",
                strip_markdown_output=bool(
                    options.get(CONF_STRIP_MARKDOWN, DEFAULT_STRIP_MARKDOWN)
                ),
            )
        except HomeAssistantError as err:
            return _intent_error_result(
                language=user_input.language,
                conversation_id=chat_log.conversation_id,
                message=str(err),
                code=intent.IntentResponseErrorCode.FAILED_TO_HANDLE,
            )

        # --- Construct final response ---
        intent_response = intent.IntentResponse(language=user_input.language)
        speech_text = final_speech_from_chat_log(
            chat_log.content, thinking_enabled=bool(thinking_on)
        )
        if speech_text:
            LOGGER.debug(
                "[Debug conversation]: final speech after tool loop (%d chars): %.120s%s",
                len(speech_text),
                speech_text,
                "…" if len(speech_text) > 120 else "",
            )
        else:
            LOGGER.warning(
                "DeepSeek: empty speech after tool loop; tail=%s",
                [(type(c).__name__, getattr(c, "role", None)) for c in chat_log.content[-6:]],
            )
        
        if options.get(CONF_STRIP_MARKDOWN, DEFAULT_STRIP_MARKDOWN):
            speech_text = strip_markdown(speech_text)

        intent_response.async_set_speech(speech_text)

        return conversation.ConversationResult(
            response=intent_response,
            conversation_id=chat_log.conversation_id,
            continue_conversation=chat_log.continue_conversation,
        )
