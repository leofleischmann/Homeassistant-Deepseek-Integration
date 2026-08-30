"""Turning one DeepSeek delta stream into Home Assistant chat-log deltas.

One stream per API round. Home Assistant forwards every delta it is handed
straight to the UI and to text-to-speech, which is what makes this the only
place some things can happen: markdown has to be stripped here rather than on
the finished answer, or the spoken reply keeps its asterisks, and an empty
``content`` must never be yielded, because Assist treats it as falsy and drops
the message.

The two ``*_http_version`` / ``*_unexpected_reasoning`` helpers report on the
connection and on an endpoint that ignores the thinking flag; both write to the
config entry's runtime data, which is why they take it rather than reading it.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator, Callable
import json
from typing import Any

from openai import AsyncStream
from openai.types.chat import ChatCompletionChunk

from homeassistant.components import conversation  # pyright: ignore[reportMissingImports]
from homeassistant.exceptions import HomeAssistantError  # pyright: ignore[reportMissingImports]
from homeassistant.helpers import llm  # pyright: ignore[reportMissingImports]

from .const import LOGGER
from .markdown_strip import StreamingMarkdownStripper
from .usage_metrics import CompletionUsage, completion_usage_from_api


def record_http_version(runtime: Any, result: Any) -> None:
    """Remember the negotiated HTTP version of the streaming response.

    HTTP/1.1 means every API round pays a fresh TCP+TLS handshake, because the
    OpenAI SDK closes the streaming response at ``[DONE]`` without draining it,
    so httpx cannot return the connection to the pool. Over HTTP/2 only the
    stream is closed and the connection is reused. Surfaced in the debug report
    so slow round trips can be attributed correctly.
    """
    if runtime is None:
        return
    version = getattr(getattr(result, "response", None), "http_version", None)
    if not version or version == getattr(runtime, "http_version", None):
        return
    runtime.http_version = version
    LOGGER.debug(
        "[Debug conversation]: negotiated %s with the API endpoint%s",
        version,
        ""
        if version.startswith("HTTP/2")
        else " (HTTP/1.1: each API round re-establishes the TLS connection)",
    )


def warn_unexpected_reasoning(runtime: Any, *, model: str, base_url: str) -> None:
    """Warn once per config entry when reasoning arrives although it is off.

    ``build_chat_completion_args`` sends ``extra_body.thinking = disabled`` for
    DeepSeek model ids, and V4 defaults to reasoning **enabled** when the field
    is absent. A gateway that drops unknown ``extra_body`` keys therefore leaves
    reasoning on: the tokens are generated and billed, and the user waits for
    them, but ``transform_stream`` discards the text so nothing is visible.
    That silent latency is worth one warning.
    """
    if runtime is None or getattr(runtime, "warned_unexpected_reasoning", False):
        return
    runtime.warned_unexpected_reasoning = True
    LOGGER.warning(
        "DeepSeek returned reasoning_content although reasoning is disabled in "
        "the options (model=%s, base_url=%s). The API endpoint appears to ignore "
        "the 'thinking: disabled' flag, so reasoning tokens are still generated, "
        "billed and waited for while being discarded. Check the reasoning_tokens "
        "sensor; if it keeps rising, this endpoint does not support switching "
        "reasoning off and responses will be slower than expected.",
        model,
        base_url,
    )


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


async def transform_stream(
    result: AsyncStream[ChatCompletionChunk],
    *,
    thinking_enabled: bool = False,
    usage_events: list[CompletionUsage] | None = None,
    on_unexpected_reasoning: Callable[[], None] | None = None,
    markdown_stripper: StreamingMarkdownStripper | None = None,
) -> AsyncGenerator[conversation.AssistantContentDeltaDict, None]:
    """Transform a DeepSeek delta stream (ChatCompletionChunk) into HA format.

    One stream per API round. The first chunk that carries text or a tool call
    also carries ``role`` so Home Assistant starts a fresh assistant message
    (same pattern as the stock Ollama integration); ending the stream lets HA
    finalize the message and run any pending tool calls.

    ``on_unexpected_reasoning`` is invoked once per stream if the API sends
    ``reasoning_content`` while reasoning is switched off — see
    ``warn_unexpected_reasoning``.

    ``markdown_stripper`` removes formatting from the text deltas as they pass
    through. It has to happen here: Home Assistant forwards every delta to the
    UI and to text-to-speech immediately, so stripping the finished answer only
    ever fixed the transcript, never what was spoken. Reasoning text is left
    alone - it is displayed, not read out.
    """
    current_tool_calls: list[dict[str, Any]] = []
    current_tool_call_args_buffer: dict[int, str] = {}
    role_emitted = False
    reported_unexpected_reasoning = False
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

            reasoning_delta = _stream_delta_text(delta, "reasoning_content")
            content_delta = _stream_delta_text(delta, "content")
            if not thinking_enabled and reasoning_delta:
                LOGGER.debug(
                    "[Debug conversation]: dropping reasoning_content stream "
                    "(thinking_enabled is false)"
                )
                reasoning_delta = None
                if not reported_unexpected_reasoning:
                    reported_unexpected_reasoning = True
                    if on_unexpected_reasoning is not None:
                        on_unexpected_reasoning()
            if content_delta and not (getattr(delta, "content", None) or ""):
                LOGGER.debug(
                    "Stream delta: using content from model_extra (attr empty or unset)"
                )
            if markdown_stripper is not None and content_delta:
                # May return nothing: the stripper holds text back until a point
                # a markdown construct cannot reach across. flush() releases it.
                content_delta = markdown_stripper.feed(content_delta) or None

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
                    if tool_call_chunk.id and function_name:
                        current_tool_calls[index] = {
                            "id": tool_call_chunk.id,
                            # Several OpenAI-compatible gateways leave "type" out
                            # of the opening chunk. Requiring it dropped the whole
                            # tool call, so the model asked to switch a light and
                            # nothing happened, with no error anywhere.
                            "type": tool_call_chunk.type or "function",
                            "function": {"name": function_name, "arguments": ""}
                        }
                        current_tool_call_args_buffer[index] = ""
                        LOGGER.debug("Tool Call Start Detected: Index=%d, ID=%s, Name=%s", index, tool_call_chunk.id, function_name)
                    else:
                         LOGGER.warning("Incomplete tool call start info in chunk: %s", tool_call_chunk)
                if tool_call_chunk.function and tool_call_chunk.function.arguments and index in current_tool_call_args_buffer:
                    current_tool_call_args_buffer[index] += tool_call_chunk.function.arguments

        if finish_reason:
            # Release held-back text before any tool_calls delta, so the
            # assistant message keeps the order the model produced.
            if markdown_stripper is not None and (tail := markdown_stripper.flush()):
                text_deltas, role_emitted = _yield_assistant_text_deltas(
                    role_emitted=role_emitted,
                    content_delta=tail,
                    reasoning_delta=None,
                )
                for text_delta in text_deltas:
                    yield text_delta
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

    # Some gateways end a stream without ever sending a finish_reason. Nothing
    # the stripper is still holding may be lost; flush() is empty if the
    # terminal chunk above already drained it.
    if markdown_stripper is not None and (tail := markdown_stripper.flush()):
        text_deltas, role_emitted = _yield_assistant_text_deltas(
            role_emitted=role_emitted,
            content_delta=tail,
            reasoning_delta=None,
        )
        for text_delta in text_deltas:
            yield text_delta
