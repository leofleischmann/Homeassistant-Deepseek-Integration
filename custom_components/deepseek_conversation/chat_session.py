"""The shared API loop: drive chat completions against a Home Assistant ChatLog.

Both entities run on this. The conversation agent (``conversation.py``) and the
AI Task entity (``ai_task.py``) differ in how they prepare the chat log and what
they do with the answer, not in how they talk to the API - so the loop that
sends a round, streams the reply, runs whatever tools the model asked for and
sends the next round lives here rather than in either platform.

It also owns the tool schema conversion, because a tool that cannot be
converted must never reach the request as an empty schema.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
import json
from typing import Any

import openai

from homeassistant.components import conversation  # pyright: ignore[reportMissingImports]
from homeassistant.const import CONF_LLM_HASS_API  # pyright: ignore[reportMissingImports]
from homeassistant.core import HomeAssistant  # pyright: ignore[reportMissingImports]
from homeassistant.exceptions import HomeAssistantError  # pyright: ignore[reportMissingImports]
from homeassistant.helpers import llm  # pyright: ignore[reportMissingImports]

from .api_errors import api_error_user_message
from .chat_messages import (
    async_apply_attachments_to_last_user_message,
    convert_content_to_messages,
    HAJSONEncoder,
)
from .const import (
    CONF_BASE_URL,
    CONF_CHAT_MODEL,
    CONF_MAX_TOOL_ITERATIONS,
    CONF_THINKING_ENABLED,
    DEEPSEEK_API_BASE_URL,
    DEFAULT_THINKING_ENABLED,
    LOGGER,
    RECOMMENDED_CHAT_MODEL,
    RECOMMENDED_MAX_TOOL_ITERATIONS,
    RESPONSE_FORMAT_JSON_OBJECT,
)
from .context_trim import trim_messages_for_api
from .markdown_strip import StreamingMarkdownStripper
from .openapi_schema import render_openapi_schema
from .options import coerce_max_tool_iterations, request_timeout_from_options
from .request_builder import build_chat_completion_args
from .stream_transform import (
    record_http_version,
    transform_stream,
    warn_unexpected_reasoning,
)
from .structured_output import (
    append_structure_guidance_to_last_user_message,
    build_response_format_for_schema,
)
from .types import DeepSeekConfigEntry
from .usage_metrics import CompletionUsage
from .vision import (
    latest_user_attachments,
    raise_if_vision_unsupported,
    vision_enabled_in_options,
)


def _format_tool(
    tool: llm.Tool, custom_serializer: Callable[[Any], Any] | None
) -> dict[str, Any] | None:
    """Format one HA LLM tool for OpenAI-compatible ``tools`` array.

    Returns ``None`` when the parameter schema cannot be rendered, so callers
    never send an empty schema (which causes opaque API errors) nor an object
    the SDK cannot serialise. See ``_format_tools_for_api``.
    """
    try:
        parameters = render_openapi_schema(
            tool.parameters, custom_serializer=custom_serializer
        )
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


async def async_handle_chat_log(
    hass: HomeAssistant,
    entry: DeepSeekConfigEntry,
    chat_log: conversation.ChatLog,
    *,
    options: Mapping[str, Any],
    agent_id: str,
    force_json: bool = False,
    response_schema: dict[str, Any] | None = None,
    usage_source: str = "assist",
    strip_markdown_output: bool = False,
) -> None:
    """Drive DeepSeek streaming chat completions against an HA ``ChatLog``.

    Shared by the conversation agent and the AI Task entity. Loops until the
    model stops requesting tools or ``max_tool_iterations`` is hit. When
    ``force_json`` is true, sets ``response_format`` for structured AI Task
    output (``json_object`` on official DeepSeek, ``json_schema`` on custom
    gateways when ``response_schema`` is provided).

    ``strip_markdown_output`` removes formatting from the streamed text. Only
    Assist asks for it: an AI Task result is consumed by an automation, and for
    a structured task stripping would break the JSON.

    ``options`` are the calling agent's settings - a subentry's data, or the
    first agent's settings for the entry-wide actions. The entry itself only
    carries the credentials.
    """
    runtime = entry.runtime_data
    if runtime is None or runtime.client is None:
        LOGGER.error("DeepSeek client not available in runtime_data.")
        raise HomeAssistantError("DeepSeek client not available")

    client: openai.AsyncClient = runtime.client
    model = options.get(CONF_CHAT_MODEL, RECOMMENDED_CHAT_MODEL)
    base_url = entry.data.get(CONF_BASE_URL, DEEPSEEK_API_BASE_URL)

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
            raise HomeAssistantError(
                "Home Assistant tools could not be prepared for the API. "
                "Check the log for skipped tool names."
            )
    elif hass_api_key and usage_source == "assist":
        LOGGER.warning(
            "HASS API '%s' selected in options, but chat_log.llm_api is None "
            "after async_provide_llm_data. Tools cannot be sent.",
            hass_api_key,
        )

    thinking_on = bool(options.get(CONF_THINKING_ENABLED, DEFAULT_THINKING_ENABLED))
    api_options: dict[str, Any] = dict(options)
    if force_json:
        # Structured AI Task output must land in ``content``; thinking mode can
        # leave the final answer in reasoning_content only (see generate_content).
        thinking_on = False
        api_options[CONF_THINKING_ENABLED] = False

    attachments = latest_user_attachments(chat_log.content)
    if attachments:
        if not vision_enabled_in_options(options):
            raise HomeAssistantError(
                "Images are switched off for this agent. Turn on "
                "'Allow images' in its settings to send attachments."
            )
        raise_if_vision_unsupported(model, base_url=base_url)

    initial_messages = convert_content_to_messages(
        chat_log.content,
        model=model,
        thinking_enabled=thinking_on,
        options=options,
    )
    await async_apply_attachments_to_last_user_message(
        hass, chat_log.content, initial_messages
    )
    if response_schema is not None:
        append_structure_guidance_to_last_user_message(
            initial_messages, response_schema
        )
    LOGGER.debug(
        "Sending messages to DeepSeek: %s",
        json.dumps(initial_messages, indent=2, cls=HAJSONEncoder),
    )

    max_tool_iterations = coerce_max_tool_iterations(
        options.get(CONF_MAX_TOOL_ITERATIONS, RECOMMENDED_MAX_TOOL_ITERATIONS)
    )
    # Read per turn, not per client: options apply without a config entry reload.
    # This is a read timeout, so it bounds the gap between two stream chunks -
    # a long answer that keeps streaming is never cut off, a stalled endpoint is.
    # Bound once here; every round of the tool loop reuses this view of the
    # client, which shares the connection pool with the entry's client.
    stream_timeout = request_timeout_from_options(options)
    bounded_client = client.with_options(timeout=stream_timeout)
    LOGGER.debug(
        "[Debug conversation]: max_tool_iterations=%d force_json=%s usage_source=%s "
        "stream_timeout=%.0fs",
        max_tool_iterations,
        force_json,
        usage_source,
        stream_timeout,
    )

    response_format: dict[str, Any] | None = None
    if force_json:
        if response_schema is not None:
            response_format = build_response_format_for_schema(
                response_schema,
                base_url=base_url,
            )
        else:
            response_format = {"type": RESPONSE_FORMAT_JSON_OBJECT}

    all_usage: list[CompletionUsage] = []
    messages = initial_messages

    def _report_unexpected_reasoning() -> None:
        warn_unexpected_reasoning(runtime, model=model, base_url=base_url)

    try:
        for _iteration in range(max_tool_iterations):
            messages_for_api = trim_messages_for_api(messages, options=api_options)
            model_args = build_chat_completion_args(
                model=model,
                messages=messages_for_api,
                options=api_options,
                stream=True,
                tools=tools,
                tool_choice=tool_choice,
                response_format=response_format,
            )
            LOGGER.debug("Model arguments for DeepSeek: %s", model_args)
            result = await bounded_client.chat.completions.create(**model_args)
            record_http_version(runtime, result)
            new_contents = [
                content
                async for content in chat_log.async_add_delta_content_stream(
                    agent_id,
                    transform_stream(
                        result,
                        thinking_enabled=thinking_on,
                        usage_events=all_usage,
                        on_unexpected_reasoning=_report_unexpected_reasoning,
                        markdown_stripper=(
                            StreamingMarkdownStripper()
                            if strip_markdown_output and not force_json
                            else None
                        ),
                    ),
                )
            ]

            if not chat_log.unresponded_tool_results:
                LOGGER.debug("Iteration %d finished. No tool calls.", _iteration + 1)
                break

            LOGGER.debug(
                "Iteration %d finished. Tool results in, extending messages.",
                _iteration + 1,
            )
            messages.extend(
                convert_content_to_messages(
                    new_contents,
                    model=model,
                    thinking_enabled=thinking_on,
                    options=options,
                )
            )
        else:
            LOGGER.warning(
                "Max tool iterations (%d) reached for conversation %s",
                max_tool_iterations,
                chat_log.conversation_id,
            )
            raise HomeAssistantError("Maximum tool iterations reached")

    except openai.AuthenticationError as err:
        # Its own clause only because a rejected key also has to start reauth;
        # the message is the one api_error_user_message gives for this type.
        LOGGER.error("DeepSeek API key rejected: %s", err)
        entry.async_start_reauth(hass)
        raise HomeAssistantError(api_error_user_message(err)) from err
    except (
        openai.RateLimitError,
        openai.APIConnectionError,
        openai.BadRequestError,
        openai.APIStatusError,
        openai.OpenAIError,
    ) as err:
        LOGGER.error("DeepSeek API error: %s", err)
        raise HomeAssistantError(api_error_user_message(err)) from err
    except TypeError as err:
        LOGGER.error(
            "TypeError during DeepSeek API call (likely tool serialization): %s",
            err,
            exc_info=True,
        )
        raise HomeAssistantError(f"Failed to send request: {err}") from err
    except HomeAssistantError:
        raise
    except Exception as err:
        LOGGER.error("Error processing DeepSeek stream: %s", err)
        error_msg = str(err)
        if error_msg == "max_token":
            raise HomeAssistantError("Response truncated by token limit") from err
        if error_msg == "content_filter":
            raise HomeAssistantError("Response blocked by content filter") from err
        raise HomeAssistantError(error_msg) from err
    finally:
        # Every round that produced a usage event was billed, whether or not a
        # later round failed. Recording only on success silently undercounted
        # exactly the expensive turns: a long tool loop that hits the iteration
        # cap, or an API error several rounds in.
        for usage in all_usage:
            runtime.usage.record(usage, source=usage_source)
