"""Conversation support for DeepSeek."""

from collections.abc import AsyncGenerator, Callable
import json
# Removed Literal import as it might not be strictly needed now
from typing import Any, AsyncGenerator, Callable, Literal, cast, Optional, Union, Dict, List

import openai
# Import necessary types for chat completions
from openai import AsyncStream
from openai.types.chat import ChatCompletionChunk
from openai.types.chat.chat_completion import ChatCompletionMessage, ChatCompletion
from openai.types.chat.chat_completion_message_tool_call import ChatCompletionMessageToolCall

# --- Import voluptuous_openapi ---
from voluptuous_openapi import convert  # pyright: ignore[reportMissingImports]
# --- End Import ---
import voluptuous as vol # Keep for basic schema validation if needed  # pyright: ignore[reportMissingImports]

from homeassistant.components import assist_pipeline, conversation  # pyright: ignore[reportMissingImports]
from homeassistant.config_entries import ConfigEntry  # pyright: ignore[reportMissingImports]
# --- Import CONF_LLM_HASS_API ---
from homeassistant.const import MATCH_ALL, CONF_LLM_HASS_API  # pyright: ignore[reportMissingImports]
# --- End Import ---
from homeassistant.core import HomeAssistant  # pyright: ignore[reportMissingImports]
# --- Import HomeAssistantError --- Needed for exception handling
from homeassistant.exceptions import HomeAssistantError  # pyright: ignore[reportMissingImports]
# --- End Import ---
from homeassistant.helpers import device_registry as dr, intent, llm  # pyright: ignore[reportMissingImports]
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback  # pyright: ignore[reportMissingImports]

# Use the specific type alias if defined, otherwise generic ConfigEntry
# from . import DeepSeekConfigEntry
type DeepSeekConfigEntry = ConfigEntry

# Updated imports from const
from .const import (
    CONF_CHAT_MODEL,
    CONF_MAX_TOKENS,
    CONF_PROMPT, # Keep CONF_PROMPT for options access
    CONF_TEMPERATURE,
    CONF_TOP_P,
    DOMAIN, # Use updated domain
    LOGGER, # Use the logger from const
    RECOMMENDED_CHAT_MODEL,
    RECOMMENDED_MAX_TOKENS,
    RECOMMENDED_TEMPERATURE,
    RECOMMENDED_TOP_P,
)

# Max number of back and forth with the LLM for tool usage
MAX_TOOL_ITERATIONS = 10


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: DeepSeekConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up conversation entities."""
    if not hasattr(config_entry, 'runtime_data') or config_entry.runtime_data is None:
        LOGGER.error("DeepSeek client not initialized in config entry.")
        return

    agent = DeepSeekConversationEntity(config_entry)
    async_add_entities([agent])


# --- Tool Formatting (Keep if using tools with DeepSeek) ---
def _format_tool(
    tool: llm.Tool, custom_serializer: Callable[[Any], Any] | None
) -> Dict[str, Any]:
    """Format tool specification for OpenAI-compatible tool format."""
    # --- Use voluptuous_openapi.convert for parameters ---
    try:
        # Pass the voluptuous schema directly to convert
        parameters = convert(tool.parameters, custom_serializer=custom_serializer)
    except Exception as e:
        LOGGER.error("Error converting tool parameters for %s: %s", tool.name, e)
        # Fallback or decide how to handle conversion errors
        parameters = {"type": "object", "properties": {}} # Empty schema on error
    # --- End Use ---

    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": parameters, # Use the converted JSON schema
        }
    }
# --- End Tool Formatting ---


# --- Message Conversion (Adapted for chat.completions) ---
# --- MODIFIED: Removed system_prompt argument ---
def _convert_content_to_messages(
    content_list: list[conversation.Content],
) -> list[dict[str, Any]]:
    """Convert conversation history (excluding system prompt) to DeepSeek API message format."""
    messages = []
    # --- REMOVED: Explicit system prompt addition ---

    for content in content_list:
        role: Optional[Literal["user", "assistant", "tool"]] = None
        message_content: str | list[dict[str, Any]] | None = None
        tool_calls: list[ChatCompletionMessageToolCall] | None = None
        tool_call_id: str | None = None

        # --- ADDED: Skip system messages potentially added by async_update_llm_data ---
        if isinstance(content, conversation.SystemContent):
             continue
        # --- End ADDED ---

        if isinstance(content, conversation.UserContent):
            role = "user"
            message_content = content.content
        elif isinstance(content, conversation.AssistantContent):
            role = "assistant"
            message_content = content.content
            if content.tool_calls:
                formatted_tool_calls = []
                for tc in content.tool_calls:
                    arguments_str = json.dumps(tc.tool_args) if not isinstance(tc.tool_args, str) else tc.tool_args
                    formatted_tool_calls.append(
                         ChatCompletionMessageToolCall(
                            id=tc.id,
                            function=dict(name=tc.tool_name, arguments=arguments_str),
                            type="function"
                        )
                    )
                tool_calls = formatted_tool_calls
        elif isinstance(content, conversation.ToolResultContent):
            role = "tool"
            message_content = json.dumps(content.tool_result)
            tool_call_id = content.tool_call_id

        if role:
            msg: Dict[str, Any] = {"role": role}
            if message_content:
                msg["content"] = message_content
            if tool_calls:
                if role == "assistant":
                    msg["content"] = msg.get("content")
                msg["tool_calls"] = [tc.model_dump(exclude_unset=True) if hasattr(tc, 'model_dump') else tc for tc in tool_calls]
            if tool_call_id:
                msg["tool_call_id"] = tool_call_id
            messages.append(msg)

    return messages
# --- End Message Conversion ---


# --- Stream Transformation (Adapted for ChatCompletionChunk) ---
async def _transform_stream(
    chat_log: conversation.ChatLog,
    result: AsyncStream[ChatCompletionChunk],
) -> AsyncGenerator[conversation.AssistantContentDeltaDict, None]:
    """Transform a DeepSeek delta stream (ChatCompletionChunk) into HA format."""
    current_tool_calls: list[dict] = []
    current_tool_call_args_buffer: dict[int, str] = {}
    role: Optional[Literal["assistant"]] = None
    full_response_log = [] # --- DEBUG: Log full stream ---

    async for chunk in result:
        # --- DEBUG: Log each chunk ---
        LOGGER.debug("DeepSeek Stream Chunk: %s", chunk.model_dump_json(indent=2))
        full_response_log.append(chunk.model_dump())
        # --- END DEBUG ---

        delta = chunk.choices[0].delta if chunk.choices else None
        finish_reason = chunk.choices[0].finish_reason if chunk.choices else None

        if not delta:
            continue

        if delta.role:
            if delta.role == "assistant":
                role = delta.role
                yield {"role": role}
            else:
                LOGGER.warning("Unexpected role in stream delta: %s", delta.role)

        if delta.content:
            yield {"content": delta.content}

        if delta.tool_calls:
            LOGGER.debug("Received Tool Call Chunk: %s", delta.tool_calls)
            for tool_call_chunk in delta.tool_calls:
                if tool_call_chunk.index is None:
                    LOGGER.warning("Tool call chunk missing index: %s", tool_call_chunk)
                    continue
                index = tool_call_chunk.index
                if index >= len(current_tool_calls):
                    current_tool_calls.extend([{}] * (index - len(current_tool_calls) + 1))
                    function_name = tool_call_chunk.function.name if tool_call_chunk.function else None
                    if tool_call_chunk.id and tool_call_chunk.type and function_name:
                        current_tool_calls[index] = {
                            "id": tool_call_chunk.id,
                            "type": tool_call_chunk.type,
                            "function": {"name": function_name, "arguments": ""}
                        }
                        current_tool_call_args_buffer[index] = ""
                        LOGGER.debug("Tool Call Start Detected: Index=%d, ID=%s, Name=%s", index, tool_call_chunk.id, function_name)
                    else:
                         LOGGER.warning("Incomplete tool call start info in chunk: %s", tool_call_chunk)
                if tool_call_chunk.function and tool_call_chunk.function.arguments and index in current_tool_call_args_buffer:
                    current_tool_call_args_buffer[index] += tool_call_chunk.function.arguments

        if finish_reason:
            LOGGER.debug("Stream Finish Reason: %s", finish_reason)
            LOGGER.debug("Final Tool Args Buffer: %s", current_tool_call_args_buffer)
            LOGGER.debug("Final Current Tool Calls: %s", current_tool_calls)
            if finish_reason == "tool_calls":
                tool_inputs = []
                for index, args_str in current_tool_call_args_buffer.items():
                    if index < len(current_tool_calls) and current_tool_calls[index]:
                        tool_call_info = current_tool_calls[index]
                        if "function" in tool_call_info and "name" in tool_call_info["function"]:
                            try:
                                LOGGER.debug("Attempting to parse args for %s: %s", tool_call_info["function"]["name"], args_str)
                                tool_args = json.loads(args_str) if args_str else {}
                                tool_inputs.append(
                                    llm.ToolInput(
                                        id=tool_call_info["id"],
                                        tool_name=tool_call_info["function"]["name"],
                                        tool_args=tool_args,
                                    )
                                )
                                LOGGER.debug("Successfully parsed tool input: %s", tool_inputs[-1])
                            except json.JSONDecodeError as e:
                                LOGGER.error(
                                    "Failed to decode tool arguments for %s: %s. Error: %s",
                                    tool_call_info["function"]["name"], args_str, e
                                )
                        else:
                             LOGGER.warning("Missing function info for tool call at index %d", index)
                if tool_inputs:
                    yield {"tool_calls": tool_inputs}
                current_tool_calls = []
                current_tool_call_args_buffer = {}
            elif finish_reason == "stop":
                pass
            elif finish_reason == "length":
                raise HomeAssistantError("max_token")
            elif finish_reason == "content_filter":
                 raise HomeAssistantError("content_filter")
            else:
                 raise HomeAssistantError(f"finish_reason_{finish_reason}")

# --- End Stream Transformation ---


class DeepSeekConversationEntity(
    conversation.ConversationEntity, conversation.AbstractConversationAgent
):
    """DeepSeek conversation agent."""
    _attr_has_entity_name = True
    _attr_name = None

    def __init__(self, entry: DeepSeekConfigEntry) -> None:
        """Initialize the agent."""
        self.entry = entry
        self._attr_unique_id = entry.entry_id
        self._attr_device_info = dr.DeviceInfo(
            identifiers={(DOMAIN, entry.entry_id)},
            name=entry.title,
            manufacturer="DeepSeek",
            model="DeepSeek API",
            entry_type=dr.DeviceEntryType.SERVICE,
        )
        if self.entry.options.get(CONF_LLM_HASS_API):
            self._attr_supported_features = (
                conversation.ConversationEntityFeature.CONTROL
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
        self.entry.async_on_unload(
            self.entry.add_update_listener(self._async_entry_update_listener)
        )

    async def async_will_remove_from_hass(self) -> None:
        conversation.async_unset_agent(self.hass, self.entry)
        await super().async_will_remove_from_hass()

    async def _async_handle_message(
        self,
        user_input: conversation.ConversationInput,
        chat_log: conversation.ChatLog,
    ) -> conversation.ConversationResult:
        """Handle a message using DeepSeek."""
        options = self.entry.options
        if not hasattr(self.entry, 'runtime_data') or not isinstance(self.entry.runtime_data, openai.AsyncClient):
             LOGGER.error("DeepSeek client not available in runtime_data.")
             intent_response = intent.IntentResponse(language=user_input.language)
             intent_response.async_set_error(
                  intent.IntentResponseErrorCode.UNKNOWN, "DeepSeek client not available"
             )
             return conversation.ConversationResult(
                 response=intent_response, conversation_id=chat_log.conversation_id
             )
        client: openai.AsyncClient = self.entry.runtime_data
        model = options.get(CONF_CHAT_MODEL, RECOMMENDED_CHAT_MODEL)

        # --- Reinstate async_update_llm_data call ---
        try:
            await chat_log.async_update_llm_data(
                DOMAIN,
                user_input,
                options.get(CONF_LLM_HASS_API), # Pass selected API key
                options.get(CONF_PROMPT), # Pass RAW prompt template
            )
        except conversation.ConverseError as err:
             # Handle potential errors during prompt rendering/context management
             LOGGER.error("Error during chat_log.async_update_llm_data: %s", err)
             intent_response = intent.IntentResponse(language=user_input.language)
             intent_response.async_set_error(
                 intent.IntentResponseErrorCode.UNKNOWN, f"Error preparing context: {err}"
             )
             return conversation.ConversationResult(
                 response=intent_response, conversation_id=chat_log.conversation_id
             )
        # --- End reinstate ---

        # --- Prepare tools if HASS API is available in chat_log ---
        tools: list[Dict[str, Any]] | None = None
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None
        hass_api_key = options.get(CONF_LLM_HASS_API) # Still useful for logging

        if chat_log.llm_api: # Check the object populated by async_update_llm_data
            active_llm_api = chat_log.llm_api
            try:
                 # --- Use _format_tool which now includes conversion ---
                 tools = [
                     _format_tool(tool, active_llm_api.custom_serializer)
                     for tool in active_llm_api.tools
                 ]
                 # --- End Use ---
                 tool_choice = "auto"
                 # Log only tool names to avoid serialization errors
                 tool_names = [t.get("function", {}).get("name", "unknown") for t in tools]
                 LOGGER.debug("Sending tools to DeepSeek (from chat_log.llm_api): %s", tool_names)
            except Exception as e:
                 # Log error during tool formatting/conversion
                 LOGGER.error("Error formatting tools: %s", e, exc_info=True)
                 tools = None # Ensure tools are None if formatting failed
                 tool_choice = None
        elif hass_api_key:
             LOGGER.warning("HASS API '%s' selected in options, but chat_log.llm_api is None after async_update_llm_data. Tools cannot be sent.", hass_api_key)
        # --- End Tool Prep ---


        # --- Convert chat history (NOW EXCLUDES system prompt) ---
        messages = _convert_content_to_messages(chat_log.content)
        LOGGER.debug("Sending messages to DeepSeek (excluding system): %s", json.dumps(messages, indent=2))
        # --- End Convert ---

        # To prevent infinite loops with tools
        for _iteration in range(MAX_TOOL_ITERATIONS):
            model_args: Dict[str, Any] = {
                "model": model,
                "messages": messages, # Pass history without explicit system prompt
                "max_tokens": options.get(CONF_MAX_TOKENS, RECOMMENDED_MAX_TOKENS),
                "top_p": options.get(CONF_TOP_P, RECOMMENDED_TOP_P),
                "temperature": options.get(CONF_TEMPERATURE, RECOMMENDED_TEMPERATURE),
                "stream": True,
            }
            if tools:
                model_args["tools"] = tools
            if tool_choice:
                 model_args["tool_choice"] = tool_choice

            LOGGER.debug("Model arguments for DeepSeek: %s", model_args) # This might fail again if tools aren't fully serializable by default logger

            try:
                result = await client.chat.completions.create(**model_args)
            except openai.RateLimitError as err:
                LOGGER.warning("Rate limited by DeepSeek: %s", err)
                intent_response = intent.IntentResponse(language=user_input.language)
                intent_response.async_set_error(intent.IntentResponseErrorCode.UNKNOWN, "Rate limited by DeepSeek API")
                return conversation.ConversationResult(response=intent_response, conversation_id=chat_log.conversation_id)
            except openai.APIConnectionError as err:
                 LOGGER.error("Connection error talking to DeepSeek: %s", err)
                 intent_response = intent.IntentResponse(language=user_input.language)
                 intent_response.async_set_error(intent.IntentResponseErrorCode.UNKNOWN, "Connection error with DeepSeek API")
                 return conversation.ConversationResult(response=intent_response, conversation_id=chat_log.conversation_id)
            # --- Catch potential TypeError during request encoding ---
            except TypeError as err:
                 LOGGER.error("TypeError during DeepSeek API call (likely tool serialization): %s", err, exc_info=True)
                 intent_response = intent.IntentResponse(language=user_input.language)
                 intent_response.async_set_error(intent.IntentResponseErrorCode.UNKNOWN, f"Failed to send request: {err}")
                 return conversation.ConversationResult(response=intent_response, conversation_id=chat_log.conversation_id)
            # --- End Catch ---
            except openai.OpenAIError as err:
                LOGGER.error("Error talking to DeepSeek: %s", err)
                intent_response = intent.IntentResponse(language=user_input.language)
                intent_response.async_set_error(intent.IntentResponseErrorCode.UNKNOWN, f"DeepSeek API error: {err}")
                return conversation.ConversationResult(response=intent_response, conversation_id=chat_log.conversation_id)

            # Process the stream and update chat log
            try:
                async for content_delta in chat_log.async_add_delta_content_stream(
                    user_input.agent_id, _transform_stream(chat_log, result)
                ):
                    pass # Handled by chat_log internally
            except HomeAssistantError as e:
                 LOGGER.error("Error processing DeepSeek stream: %s", e)
                 intent_response = intent.IntentResponse(language=user_input.language)
                 error_code = intent.IntentResponseErrorCode.UNKNOWN
                 error_msg = str(e)
                 if str(e) == "max_token": error_msg = "Response truncated by token limit"
                 elif str(e) == "content_filter": error_msg = "Response blocked by content filter"
                 intent_response.async_set_error(error_code, error_msg)
                 return conversation.ConversationResult(response=intent_response, conversation_id=chat_log.conversation_id)

            # --- Rebuild messages for next iteration (using updated chat_log.content) ---
            messages = _convert_content_to_messages(chat_log.content)
            # --- End Rebuild ---

            if not chat_log.unresponded_tool_results:
                LOGGER.debug("Iteration %d finished. No unresponded tool results.", _iteration + 1)
                break
            else:
                 LOGGER.debug("Iteration %d finished. Unresponded tool results found, preparing next iteration.", _iteration + 1)

        else: # Max iterations reached
            LOGGER.warning("Max tool iterations reached for conversation %s", chat_log.conversation_id)
            intent_response = intent.IntentResponse(language=user_input.language)
            intent_response.async_set_error(intent.IntentResponseErrorCode.UNKNOWN, "Maximum tool iterations reached")
            return conversation.ConversationResult(response=intent_response, conversation_id=chat_log.conversation_id)

        # --- Construct final response ---
        intent_response = intent.IntentResponse(language=user_input.language)
        last_assistant_message = next(
            (msg for msg in reversed(chat_log.content) if isinstance(msg, conversation.AssistantContent)), None
        )
        speech_text = last_assistant_message.content if last_assistant_message else ""
        intent_response.async_set_speech(speech_text or "")

        return conversation.ConversationResult(
            response=intent_response,
            conversation_id=chat_log.conversation_id,
            continue_conversation=chat_log.continue_conversation,
        )
        # --- End final response construction ---

    async def _async_entry_update_listener(
        self, hass: HomeAssistant, entry: ConfigEntry
    ) -> None:
        """Handle options update."""
        await hass.config_entries.async_reload(entry.entry_id)

