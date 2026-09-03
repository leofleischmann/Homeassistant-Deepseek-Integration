"""Translating between Home Assistant's chat log and the API message array.

Home Assistant keeps a turn as typed ``conversation.Content`` objects; the API
wants a flat list of role/content dicts. Both directions live here: building
the request from a chat log, and picking the answer back out of it once the
tool loop has finished.

Image attachments are deliberately not part of the conversion. They are
encoded once into the last user message by
``async_apply_attachments_to_last_user_message`` and then reused across every
tool round, instead of being re-read and re-encoded each time the message
array is rebuilt.
"""

from __future__ import annotations

import datetime
import json
from typing import Any, Literal

from openai.types.chat.chat_completion_message_tool_call import ChatCompletionMessageToolCall

from homeassistant.components import conversation  # pyright: ignore[reportMissingImports]
from homeassistant.core import HomeAssistant  # pyright: ignore[reportMissingImports]

from .const import LOGGER
from .context_trim import format_tool_result_content
from .vision import async_user_message_content, latest_user_attachments


class HAJSONEncoder(json.JSONEncoder):
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
                "HAJSONEncoder: falling back to str() for unserializable type %s",
                type(obj).__name__,
            )
            return str(obj)


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


def convert_content_to_messages(
    content_list: list[conversation.Content],
    *,
    model: str,
    thinking_enabled: bool,
    options: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Convert conversation history to DeepSeek API message format.

    Text only: image attachments are applied separately to the last user message
    (see ``async_apply_attachments_to_last_user_message``) so they are encoded once.
    """
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
            message_content = format_tool_result_content(
                content.tool_result,
                json_encoder=HAJSONEncoder,
                options=options or {},
                tool_name=content.tool_name,
            )
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
                msg["tool_calls"] = [
                    tc.model_dump(exclude_unset=True) for tc in tool_calls
                ]
            if tool_call_id:
                msg["tool_call_id"] = tool_call_id
            if "tool_calls" not in msg and not msg.get("content"):
                # "Invalid assistant message: content or tool_calls must be
                # set" - and because the whole chat log is rebuilt for every
                # request, one such message fails every later turn of that
                # conversation, not just the round that produced it. An
                # assistant turn can end up empty when it carried only
                # reasoning that is not replayed (see
                # _include_assistant_reasoning_in_request); there is nothing
                # left to send, so leave it out entirely. For the other roles
                # an empty string is a message, and dropping one would break
                # the user/tool pairing the API checks.
                if role == "assistant":
                    LOGGER.debug(
                        "[Debug conversation]: skipping an assistant message with "
                        "neither content nor tool calls"
                    )
                    continue
                msg["content"] = ""
            messages.append(msg)

    return messages


async def async_apply_attachments_to_last_user_message(
    hass: HomeAssistant,
    content_list: list[conversation.Content],
    messages: list[dict[str, Any]],
) -> None:
    """Encode the current turn's image attachments into the last user message.

    Mirrors the stock OpenAI/Ollama integrations: attachments are encoded once
    here and reused across all tool rounds (``messages`` is extended, not rebuilt,
    in ``_async_handle_message``) instead of re-read and re-encoded every round.
    Raises ``HomeAssistantError`` if a file cannot be read as an image.
    """
    attachments = latest_user_attachments(content_list)
    if not attachments:
        return
    for message in reversed(messages):
        if message.get("role") != "user":
            continue
        text = message.get("content")
        message["content"] = await async_user_message_content(
            hass, text if isinstance(text, str) else "", attachments
        )
        LOGGER.debug(
            "[Debug vision]: encoded %d attachment(s) into the last user message",
            len(attachments),
        )
        return


def final_speech_from_chat_log(
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
