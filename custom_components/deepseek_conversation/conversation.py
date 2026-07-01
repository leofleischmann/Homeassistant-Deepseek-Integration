"""Conversation support for DeepSeek."""

from __future__ import annotations

from collections.abc import AsyncGenerator, Callable
import datetime
import json
import re
from typing import Any, Literal

import openai
from openai import AsyncStream
from openai.types.chat import ChatCompletionChunk
from openai.types.chat.chat_completion_message_tool_call import ChatCompletionMessageToolCall
from voluptuous_openapi import convert  # pyright: ignore[reportMissingImports]

from homeassistant.components import assist_pipeline, conversation  # pyright: ignore[reportMissingImports]
from homeassistant.const import CONF_LLM_HASS_API, MATCH_ALL  # pyright: ignore[reportMissingImports]
from homeassistant.core import HomeAssistant  # pyright: ignore[reportMissingImports]
from homeassistant.exceptions import HomeAssistantError  # pyright: ignore[reportMissingImports]
from homeassistant.helpers import device_registry as dr, intent, llm  # pyright: ignore[reportMissingImports]
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback  # pyright: ignore[reportMissingImports]

from .api_errors import openai_exception_user_message
from .const import (
    build_chat_completion_args,
    CONF_CHAT_MODEL,
    CONF_PROMPT,
    CONF_STRIP_MARKDOWN,
    CONF_THINKING_ENABLED,
    DEFAULT_STRIP_MARKDOWN,
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_THINKING_ENABLED,
    DOMAIN,
    LOGGER,
    RECOMMENDED_CHAT_MODEL,
)
from .types import DeepSeekConfigEntry
from .usage_metrics import CompletionUsage, completion_usage_from_api

# Max number of back and forth with the LLM for tool usage
MAX_TOOL_ITERATIONS = 10


def _format_tool(
    tool: llm.Tool, custom_serializer: Callable[[Any], Any] | None
) -> dict[str, Any] | None:
    """Format one HA LLM tool for OpenAI-compatible ``tools`` array.

    Returns ``None`` when ``voluptuous_openapi.convert`` fails so callers never
    send an empty schema (which causes opaque API errors). See ``_format_tools_for_api``.
    """
    try:
        parameters = convert(tool.parameters, custom_serializer=custom_serializer)
    except Exception as err:
        LOGGER.warning(
            "[Debug conversation]: Skipping tool %s - parameter schema conversion "
            "failed: %s",
            tool.name,
            err,
        )
        return None

    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": parameters,
        },
    }


def _format_tools_for_api(
    tools: list[llm.Tool],
    custom_serializer: Callable[[Any], Any] | None,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Convert HA tools for the chat API; return (formatted, skipped names)."""
    formatted: list[dict[str, Any]] = []
    skipped: list[str] = []
    for tool in tools:
        payload = _format_tool(tool, custom_serializer)
        if payload is None:
            skipped.append(tool.name)
        else:
            formatted.append(payload)
    return formatted, skipped


def _intent_error_result(
    *,
    language: str,
    conversation_id: str,
    message: str,
    code: intent.IntentResponseErrorCode = intent.IntentResponseErrorCode.UNKNOWN,
) -> conversation.ConversationResult:
    """Build a ConversationResult that surfaces an error to the user.

    Centralised so the seven distinct API/stream error paths in
    ``_async_handle_message`` don't each rebuild an ``IntentResponse``.
    """
    intent_response = intent.IntentResponse(language=language)
    intent_response.async_set_error(code, message)
    return conversation.ConversationResult(
        response=intent_response, conversation_id=conversation_id
    )


def _classify_openai_error(
    err: BaseException,
) -> tuple[intent.IntentResponseErrorCode, str]:
    """Map an OpenAI/DeepSeek SDK error to an HA intent error code + message.

    Avoids the previous behaviour of returning ``UNKNOWN`` for every error
    type, which made it impossible for Assist / TTS to differentiate between
    e.g. network outages and bad requests.
    """
    msg = openai_exception_user_message(err)
    if isinstance(err, openai.AuthenticationError):
        return (
            intent.IntentResponseErrorCode.FAILED_TO_HANDLE,
            "Authentication failed — check the DeepSeek API key",
        )
    if isinstance(err, openai.RateLimitError):
        return (
            intent.IntentResponseErrorCode.FAILED_TO_HANDLE,
            "Rate limited by DeepSeek API",
        )
    if isinstance(err, openai.APIConnectionError):
        return (
            intent.IntentResponseErrorCode.FAILED_TO_HANDLE,
            "Connection error with DeepSeek API",
        )
    if isinstance(err, (openai.BadRequestError, openai.APIStatusError, openai.OpenAIError)):
        return intent.IntentResponseErrorCode.FAILED_TO_HANDLE, msg
    return intent.IntentResponseErrorCode.UNKNOWN, msg


class _HAJSONEncoder(json.JSONEncoder):
    """JSON encoder that handles HA types not supported by the stdlib encoder.

    Home Assistant's LLM tool results (e.g. IntentResponseDict.speech_slots)
    can contain ``datetime.time`` / ``datetime.date`` / ``datetime.datetime``
    objects.  The stdlib ``json`` module raises a ``TypeError`` for these;
    this encoder serialises them as ISO-8601 strings instead.

    For any other unknown type, a ``str()`` representation is used as a
    safe fallback so that serialisation never crashes the pipeline.
    """

    def default(self, obj: Any) -> Any:  # noqa: ANN401
        if isinstance(obj, (datetime.time, datetime.date, datetime.datetime)):
            return obj.isoformat()
        # Fallback: convert unknown types to string instead of crashing.
        try:
            return super().default(obj)
        except TypeError:
            LOGGER.debug(
                "_HAJSONEncoder: falling back to str() for unserializable type %s",
                type(obj).__name__,
            )
            return str(obj)


def _strip_markdown(text: str) -> str:
    """Strip common markdown formatting for TTS readability."""
    if not text:
        return text
    
    # Remove code block formatting (```python)
    text = re.sub(r'```[a-z]*\n?', '', text)
    # Remove inline code formatting
    text = text.replace('`', '')
    
    # Remove blockquotes
    text = re.sub(r'(?m)^\s*>\s+', '', text)
    # Remove headings
    text = re.sub(r'(?m)^#{1,6}\s+', '', text)
    
    # Remove bold/italic (asterisks and underscores)
    text = re.sub(r'(?<!\w)\*\*(?!\s)(.+?)(?<!\s)\*\*(?!\w)', r'\1', text)
    text = re.sub(r'(?<!\w)\*(?!\s)(.+?)(?<!\s)\*(?!\w)', r'\1', text)
    text = re.sub(r'(?<!\w)__(?!\s)(.+?)(?<!\s)__(?!\w)', r'\1', text)
    text = re.sub(r'(?<!\w)_(?!\s)(.+?)(?<!\s)_(?!\w)', r'\1', text)
    
    # Remove strikethrough
    text = re.sub(r'(?<!\w)~~(?!\s)(.+?)(?<!\s)~~(?!\w)', r'\1', text)
    
    # Remove images
    text = re.sub(r'!\[(.*?)\]\(.*?\)', r'\1', text)
    # Remove links (replace with just the text)
    text = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', text)
    
    # Remove list formatting
    text = re.sub(r'(?m)^\s*[-*+]\s+', '', text)
    
    # Remove arrows
    text = text.replace('→', '').replace('->', '')
    
    return text.strip()


def _is_deepseek_reasoner_model(model: str) -> bool:
    """True for deepseek-reasoner (CoT must not be replayed in request history)."""
    return "reasoner" in (model or "").lower()


def _include_assistant_reasoning_in_request(
    *,
    model: str,
    thinking_enabled: bool,
    has_tool_calls: bool,
) -> bool:
    """Whether to attach reasoning_content for an assistant message.

    deepseek-reasoner: never send reasoning in the messages array (API 400).
    Thinking mode (non-reasoner): only on assistant turns that issued tool_calls;
    plain answers do not need CoT in context (DeepSeek thinking-mode guide).
    """
    if _is_deepseek_reasoner_model(model):
        return False
    if not thinking_enabled:
        return False
    return has_tool_calls


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


def _convert_content_to_messages(
    content_list: list[conversation.Content],
    *,
    model: str,
    thinking_enabled: bool,
) -> list[dict[str, Any]]:
    """Convert conversation history to DeepSeek API message format."""
    messages: list[dict[str, Any]] = []

    for content in content_list:
        role: Literal["user", "assistant", "tool", "system"] | None = None
        message_content: str | list[dict[str, Any]] | None = None
        tool_calls: list[ChatCompletionMessageToolCall] | None = None
        tool_call_id: str | None = None

        if isinstance(content, conversation.SystemContent):
            role = "system"
            message_content = content.content

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
            message_content = json.dumps(content.tool_result, cls=_HAJSONEncoder)
            tool_call_id = content.tool_call_id

        if role:
            msg: dict[str, Any] = {"role": role}
            if message_content:
                msg["content"] = message_content
            if isinstance(content, conversation.AssistantContent):
                thinking = getattr(content, "thinking_content", None)
                if thinking and _include_assistant_reasoning_in_request(
                    model=model,
                    thinking_enabled=thinking_enabled,
                    has_tool_calls=bool(content.tool_calls),
                ):
                    msg["reasoning_content"] = thinking
            if tool_calls:
                if role == "assistant":
                    msg["content"] = msg.get("content")
                msg["tool_calls"] = [tc.model_dump(exclude_unset=True) if hasattr(tc, 'model_dump') else tc for tc in tool_calls]
            if tool_call_id:
                msg["tool_call_id"] = tool_call_id
            messages.append(msg)

    return messages


def _final_speech_from_chat_log(
    content_list: list[conversation.Content], *, thinking_enabled: bool
) -> str:
    """Pick text for IntentResponse after tool rounds.

    Skip assistant turns that only issued tool_calls (preamble); the final answer
    is normally the next assistant message after tool results.
    """
    for msg in reversed(content_list):
        if not isinstance(msg, conversation.AssistantContent):
            continue
        if msg.tool_calls:
            continue
        raw = msg.content
        if isinstance(raw, str) and raw.strip():
            return raw.strip()
    if thinking_enabled:
        for msg in reversed(content_list):
            if not isinstance(msg, conversation.AssistantContent):
                continue
            if msg.tool_calls:
                continue
            think = getattr(msg, "thinking_content", None)
            if isinstance(think, str) and think.strip():
                LOGGER.debug(
                    "[Debug conversation]: using thinking_content as speech fallback "
                    "(no assistant text in content after tools)"
                )
                return think.strip()
    return ""


def _yield_assistant_text_deltas(
    *,
    role_emitted: bool,
    content_delta: str | None,
    reasoning_delta: str | None,
) -> tuple[list[conversation.AssistantContentDeltaDict], bool]:
    """Build HA chat_log deltas for streamed assistant text.

    Never send ``content: ""`` — empty strings are falsy in HA and the Assist UI.
    """
    deltas: list[conversation.AssistantContentDeltaDict] = []
    if not role_emitted and (content_delta or reasoning_delta):
        first: conversation.AssistantContentDeltaDict = {"role": "assistant"}
        if content_delta:
            first["content"] = content_delta
        if reasoning_delta:
            first["thinking_content"] = reasoning_delta
        deltas.append(first)
        role_emitted = True
    else:
        if content_delta:
            deltas.append({"content": content_delta})
        if reasoning_delta:
            deltas.append({"thinking_content": reasoning_delta})
    return deltas, role_emitted


def _prime_assist_ui_after_tool_results(chat_log: conversation.ChatLog) -> None:
    """Prepare stock ``ha-assist-chat`` for the next stream after tool results.

    Assist sets ``currentDeltaRole`` to ``tool_result`` per tool-result delta.
    After parallel tools the role stays on ``tool_result`` until reset; follow-up
    thinking/content deltas are ignored unless ``currentDeltaRole === "assistant"``.

    Call only after ``async_add_delta_content_stream`` finished and
    ``unresponded_tool_results`` is true. Listener-only; does not touch
    ``chat_log.content``. Each follow-up API round also yields ``role: assistant``
    at stream start (Google Gemini pattern).
    """
    if chat_log.delta_listener is None:
        return
    LOGGER.debug(
        "[Debug conversation]: priming Assist UI after tool results (listener only)"
    )
    chat_log.delta_listener(chat_log, {"role": "assistant"})
    # Nudge main text and thinking so ha-assist-chat calls requestUpdate even
    # before the next API stream arrives (ZWS is invisible in the UI).
    chat_log.delta_listener(chat_log, {"content": "\u200b"})
    chat_log.delta_listener(chat_log, {"thinking_content": "\u200b"})


def _stream_delta_text(delta: Any, field: str) -> str | None:
    """Read a streamed text field from ChoiceDelta.

    DeepSeek may send ``reasoning_content`` / ``content`` in JSON while the OpenAI
    SDK model does not map them to attributes; they then appear only in
    ``model_extra``. Home Assistant's chat log only appends ``content`` when it is
    truthy (``if delta_content := ...``), so missing string parts never reach the UI.
    """

    def _normalize(raw: Any) -> str | None:
        if raw is None:
            return None
        if isinstance(raw, str):
            return raw or None
        if isinstance(raw, list):
            parts: list[str] = []
            for item in raw:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict) and item.get("type") == "text":
                    t = item.get("text")
                    if isinstance(t, str):
                        parts.append(t)
            merged = "".join(parts)
            return merged or None
        return None

    extra = (getattr(delta, "model_extra", None) or {}).get(field)
    for candidate in (getattr(delta, field, None), extra):
        got = _normalize(candidate)
        if got is not None:
            return got
    return None


async def _transform_stream(
    chat_log: conversation.ChatLog,
    result: AsyncStream[ChatCompletionChunk],
    *,
    thinking_enabled: bool = False,
    usage_events: list[CompletionUsage] | None = None,
) -> AsyncGenerator[conversation.AssistantContentDeltaDict, None]:
    """Transform a DeepSeek delta stream (ChatCompletionChunk) into HA format.

    Every API round starts with ``{"role": "assistant"}`` (same as Google Gemini
    ``_transform_stream`` per iteration). That resets Assist ``currentDeltaRole``
    after tool-result deltas so follow-up thinking/content is not dropped.
    """
    current_tool_calls: list[dict[str, Any]] = []
    current_tool_call_args_buffer: dict[int, str] = {}
    role: Literal["assistant"] | None = None
    role_emitted = False
    LOGGER.debug("[Debug conversation]: yielding stream delta: {'role': 'assistant'}")
    yield {"role": "assistant"}
    role_emitted = True
    async for chunk in result:
        parsed_usage = completion_usage_from_api(getattr(chunk, "usage", None))
        if parsed_usage is not None:
            if usage_events is not None:
                usage_events.append(parsed_usage)
            LOGGER.debug(
                "[Debug usage_metrics]: stream usage chunk prompt=%d completion=%d",
                parsed_usage.prompt_tokens,
                parsed_usage.completion_tokens,
            )

        if not chunk.choices:
            continue
        choice0 = chunk.choices[0]
        delta = choice0.delta
        finish_reason = choice0.finish_reason

        # Never skip terminal chunks: ``finish_reason`` may be set when ``delta`` is
        # missing or an empty object (OpenAI-compatible streams); tool_calls must still
        # be finalized.
        if delta is not None:
            if delta.role and delta.role != "assistant":
                LOGGER.warning("Unexpected role in stream delta: %s", delta.role)
            elif delta.role == "assistant":
                role = delta.role

            reasoning_delta = _stream_delta_text(delta, "reasoning_content")
            content_delta = _stream_delta_text(delta, "content")
            if not thinking_enabled and reasoning_delta:
                LOGGER.debug(
                    "[Debug conversation]: dropping reasoning_content stream "
                    "(thinking_enabled is false)"
                )
                reasoning_delta = None
            if content_delta and not (getattr(delta, "content", None) or ""):
                LOGGER.debug(
                    "Stream delta: using content from model_extra (attr empty or unset)"
                )

            text_deltas, role_emitted = _yield_assistant_text_deltas(
                role_emitted=role_emitted,
                content_delta=content_delta,
                reasoning_delta=reasoning_delta,
            )
            for text_delta in text_deltas:
                LOGGER.debug("[Debug conversation]: yielding stream delta: %s", text_delta)
                yield text_delta

        if delta is not None and delta.tool_calls:
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
                    if not role_emitted:
                        # Tool-only iteration (no content/thinking streamed):
                        # bind role to the tool_calls delta so chat_log starts
                        # an assistant message instead of dropping the call.
                        yield {"role": "assistant", "tool_calls": tool_inputs}
                        role_emitted = True
                    else:
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


class DeepSeekConversationEntity(
    conversation.ConversationEntity, conversation.AbstractConversationAgent
):
    """DeepSeek conversation agent."""
    _attr_has_entity_name = True
    _attr_name = None
    _attr_supports_streaming = True

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
        self._sync_entity_attributes_from_entry(entry)

    def _sync_entity_attributes_from_entry(self, entry: DeepSeekConfigEntry) -> None:
        """Refresh entity flags from options without a config-entry reload.

        Assist reads prompt/model/temperature from ``entry.options`` each turn.
        Reload is only needed for connection data (base_url, API key); see
        config_flow.py (base_url) and reauth (API key).
        """
        if entry.options.get(CONF_LLM_HASS_API):
            self._attr_supported_features = (
                conversation.ConversationEntityFeature.CONTROL
            )
        else:
            self._attr_supported_features = conversation.ConversationEntityFeature(0)

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
        runtime = self.entry.runtime_data
        if runtime is None or runtime.client is None:
            LOGGER.error("DeepSeek client not available in runtime_data.")
            return _intent_error_result(
                language=user_input.language,
                conversation_id=chat_log.conversation_id,
                message="DeepSeek client not available",
                code=intent.IntentResponseErrorCode.FAILED_TO_HANDLE,
            )
        client: openai.AsyncClient = runtime.client
        model = options.get(CONF_CHAT_MODEL, RECOMMENDED_CHAT_MODEL)

        try:
            await chat_log.async_provide_llm_data(
                llm_context=user_input.as_llm_context(DOMAIN),
                user_llm_hass_api=options.get(CONF_LLM_HASS_API),
                user_llm_prompt=(options.get(CONF_PROMPT) or "").strip()
                or DEFAULT_SYSTEM_PROMPT,
                user_extra_system_prompt=user_input.extra_system_prompt,
            )
        except conversation.ConverseError as err:
            LOGGER.error("Error during chat_log.async_provide_llm_data: %s", err)
            return _intent_error_result(
                language=user_input.language,
                conversation_id=chat_log.conversation_id,
                message=f"Error preparing context: {err}",
                code=intent.IntentResponseErrorCode.FAILED_TO_HANDLE,
            )

        tools: list[dict[str, Any]] | None = None
        tool_choice: str | dict[str, Any] | None = None
        hass_api_key = options.get(CONF_LLM_HASS_API)

        if chat_log.llm_api:
            active_llm_api = chat_log.llm_api
            registered = list(active_llm_api.tools)
            tools, skipped_tools = _format_tools_for_api(
                registered, active_llm_api.custom_serializer
            )
            if skipped_tools:
                LOGGER.warning(
                    "[Debug conversation]: %d of %d tool(s) skipped (schema "
                    "conversion failed): %s",
                    len(skipped_tools),
                    len(registered),
                    ", ".join(skipped_tools),
                )
            if tools:
                tool_choice = "auto"
                tool_names = [
                    t.get("function", {}).get("name", "unknown") for t in tools
                ]
                LOGGER.debug(
                    "Sending tools to DeepSeek (from chat_log.llm_api): %s",
                    tool_names,
                )
            elif registered:
                LOGGER.error(
                    "[Debug conversation]: All %d tool(s) failed schema conversion; "
                    "cannot call Home Assistant tools",
                    len(registered),
                )
                return _intent_error_result(
                    language=user_input.language,
                    conversation_id=chat_log.conversation_id,
                    message=(
                        "Home Assistant tools could not be prepared for the API. "
                        "Check the log for skipped tool names."
                    ),
                    code=intent.IntentResponseErrorCode.FAILED_TO_HANDLE,
                )
        elif hass_api_key:
            LOGGER.warning(
                "HASS API '%s' selected in options, but chat_log.llm_api is None "
                "after async_provide_llm_data. Tools cannot be sent.",
                hass_api_key,
            )

        thinking_on = options.get(CONF_THINKING_ENABLED, DEFAULT_THINKING_ENABLED)

        initial_messages = _convert_content_to_messages(
            chat_log.content, model=model, thinking_enabled=thinking_on
        )
        LOGGER.debug(
            "Sending messages to DeepSeek: %s",
            json.dumps(initial_messages, indent=2, cls=_HAJSONEncoder),
        )

        max_iterations_reached = False
        current_messages = list(initial_messages)
        for _iteration in range(MAX_TOOL_ITERATIONS):
            model_args = build_chat_completion_args(
                model=model,
                messages=current_messages,
                options=options,
                stream=True,
                tools=tools,
                tool_choice=tool_choice,
            )
            LOGGER.debug("Model arguments for DeepSeek: %s", model_args)

            try:
                result = await client.chat.completions.create(**model_args)
            except openai.AuthenticationError as err:
                LOGGER.error("DeepSeek API key rejected: %s", err)
                self.entry.async_start_reauth(self.hass)
                code, message = _classify_openai_error(err)
                return _intent_error_result(
                    language=user_input.language,
                    conversation_id=chat_log.conversation_id,
                    message=message,
                    code=code,
                )
            except (
                openai.RateLimitError,
                openai.APIConnectionError,
                openai.BadRequestError,
                openai.APIStatusError,
                openai.OpenAIError,
            ) as err:
                LOGGER.error("DeepSeek API error: %s", err)
                code, message = _classify_openai_error(err)
                return _intent_error_result(
                    language=user_input.language,
                    conversation_id=chat_log.conversation_id,
                    message=message,
                    code=code,
                )
            except TypeError as err:
                LOGGER.error(
                    "TypeError during DeepSeek API call (likely tool serialization): %s",
                    err,
                    exc_info=True,
                )
                return _intent_error_result(
                    language=user_input.language,
                    conversation_id=chat_log.conversation_id,
                    message=f"Failed to send request: {err}",
                    code=intent.IntentResponseErrorCode.FAILED_TO_HANDLE,
                )

            try:
                iteration_usage: list[CompletionUsage] = []
                async for _ in chat_log.async_add_delta_content_stream(
                    user_input.agent_id,
                    _transform_stream(
                        chat_log,
                        result,
                        thinking_enabled=bool(thinking_on),
                        usage_events=iteration_usage,
                    ),
                ):
                    pass
                for usage in iteration_usage:
                    runtime.usage.record(usage, source="assist")
            except HomeAssistantError as err:
                LOGGER.error("Error processing DeepSeek stream: %s", err)
                error_msg = str(err)
                if error_msg == "max_token":
                    error_msg = "Response truncated by token limit"
                elif error_msg == "content_filter":
                    error_msg = "Response blocked by content filter"
                return _intent_error_result(
                    language=user_input.language,
                    conversation_id=chat_log.conversation_id,
                    message=error_msg,
                    code=intent.IntentResponseErrorCode.FAILED_TO_HANDLE,
                )
            except (
                openai.RateLimitError,
                openai.APIConnectionError,
                openai.APIStatusError,
                openai.OpenAIError,
            ) as err:
                LOGGER.error("DeepSeek SDK error during streaming: %s", err)
                code, message = _classify_openai_error(err)
                return _intent_error_result(
                    language=user_input.language,
                    conversation_id=chat_log.conversation_id,
                    message=message,
                    code=code,
                )

            if not chat_log.unresponded_tool_results:
                LOGGER.debug("Iteration %d finished. No tool calls.", _iteration + 1)
                break

            LOGGER.debug(
                "Iteration %d finished. Tool calls in flight, preparing next iteration.",
                _iteration + 1,
            )
            _prime_assist_ui_after_tool_results(chat_log)
            current_messages = _convert_content_to_messages(
                chat_log.content, model=model, thinking_enabled=thinking_on
            )
        else:
            max_iterations_reached = True

        if max_iterations_reached:
            LOGGER.warning(
                "Max tool iterations reached for conversation %s",
                chat_log.conversation_id,
            )
            return _intent_error_result(
                language=user_input.language,
                conversation_id=chat_log.conversation_id,
                message="Maximum tool iterations reached",
                code=intent.IntentResponseErrorCode.FAILED_TO_HANDLE,
            )

        # --- Construct final response ---
        intent_response = intent.IntentResponse(language=user_input.language)
        speech_text = _final_speech_from_chat_log(
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
            speech_text = _strip_markdown(speech_text)

        intent_response.async_set_speech(speech_text)

        return conversation.ConversationResult(
            response=intent_response,
            conversation_id=chat_log.conversation_id,
            continue_conversation=chat_log.continue_conversation,
        )

    async def _async_entry_update_listener(
        self, hass: HomeAssistant, entry: DeepSeekConfigEntry
    ) -> None:
        """Apply option changes in memory without reloading the config entry."""
        self.entry = entry
        self._sync_entity_attributes_from_entry(entry)
        self.async_write_ha_state()
        LOGGER.debug(
            "[Debug conversation]: Options applied in-memory (no reload); "
            "base_url/API key changes trigger reload via config_flow/reauth"
        )

