"""Structured AI Task output helpers.

Converts Home Assistant ``structure:`` schemas (voluptuous) to OpenAPI JSON
Schema and guides the model when the API only supports ``json_object`` (official
DeepSeek). Used by ``ai_task.py`` and ``conversation.py`` (``force_json``).
"""

from __future__ import annotations

from collections.abc import Callable
import json
import logging
from typing import Any

import voluptuous as vol
from voluptuous_openapi import convert

from homeassistant.components import conversation  # pyright: ignore[reportMissingImports]
from homeassistant.helpers import llm  # pyright: ignore[reportMissingImports]

from .const import DEEPSEEK_API_BASE_URL, RESPONSE_FORMAT_JSON_OBJECT
from .vision import is_official_deepseek_api_base_url

_LOGGER = logging.getLogger(__name__)


def structure_to_openapi_schema(
    structure: vol.Schema,
    *,
    custom_serializer: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Convert an HA AI Task structure schema to OpenAPI/JSON Schema."""
    return convert(
        structure,
        custom_serializer=custom_serializer or llm.selector_serializer,
    )


def build_response_format_for_schema(
    schema: dict[str, Any],
    *,
    base_url: str | None,
) -> dict[str, Any]:
    """Pick API response_format for structured output.

    Official DeepSeek only supports ``json_object``; custom OpenAI-compatible
    gateways may accept ``json_schema``.
    """
    if is_official_deepseek_api_base_url(base_url):
        _LOGGER.debug(
            "[Debug structured_output]: using json_object for base_url=%r",
            base_url or DEEPSEEK_API_BASE_URL,
        )
        return {"type": RESPONSE_FORMAT_JSON_OBJECT}

    _LOGGER.debug(
        "[Debug structured_output]: using json_schema for custom base_url=%r",
        base_url,
    )
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "ai_task_response",
            "strict": True,
            "schema": schema,
        },
    }


def structure_schema_for_task(
    chat_log: conversation.ChatLog,
    structure: vol.Schema,
) -> dict[str, Any]:
    """Convert an AI Task ``structure`` to OpenAPI/JSON Schema for the API call."""
    custom_serializer = (
        chat_log.llm_api.custom_serializer
        if chat_log.llm_api
        else llm.selector_serializer
    )
    schema = structure_to_openapi_schema(
        structure, custom_serializer=custom_serializer
    )
    _LOGGER.debug(
        "[Debug structured_output]: built schema with %d top-level properties",
        len(schema.get("properties", {}))
        if isinstance(schema.get("properties"), dict)
        else 0,
    )
    return schema


def append_structure_guidance_to_last_user_message(
    messages: list[dict[str, Any]],
    schema: dict[str, Any],
) -> None:
    """Append JSON schema guidance to the last user API message.

    ``UserContent`` in the chat log is frozen; mirror ``_apply_attachments_to_last_user_message``
    and patch the converted messages instead.
    """
    suffix = _structure_guidance_suffix(schema)
    for message in reversed(messages):
        if message.get("role") != "user":
            continue
        text = message.get("content")
        if not isinstance(text, str):
            continue
        message["content"] = text.rstrip() + suffix
        _LOGGER.debug(
            "[Debug structured_output]: appended schema guidance (%d chars) "
            "to last user API message",
            len(suffix),
        )
        return
    _LOGGER.warning(
        "[Debug structured_output]: no user API message found; schema not injected"
    )

def _structure_guidance_suffix(schema: dict[str, Any]) -> str:
    example = json.dumps(_example_from_schema(schema), indent=2)
    schema_text = json.dumps(schema, indent=2)
    return (
        "\n\nYou must reply with a single JSON object only (no markdown fences). "
        "Use exactly these top-level field names and value types:\n"
        f"{schema_text}\n"
        f"Example JSON shape:\n{example}"
    )


def _example_from_schema(schema: dict[str, Any]) -> dict[str, Any]:
    if schema.get("type") != "object":
        return {}
    props = schema.get("properties", {})
    if not isinstance(props, dict):
        return {}
    return {
        key: _example_value_for_property(prop)
        for key, prop in props.items()
        if isinstance(prop, dict)
    }


def _example_value_for_property(prop: dict[str, Any]) -> Any:
    prop_type = prop.get("type")
    if prop_type == "string":
        return "example"
    if prop_type == "integer":
        return 0
    if prop_type == "number":
        return 0.0
    if prop_type == "boolean":
        return False
    if prop_type == "array":
        items = prop.get("items")
        if isinstance(items, dict):
            return [_example_value_for_property(items)]
        return []
    if prop_type == "object":
        return _example_from_schema(prop)
    if enum := prop.get("enum"):
        if isinstance(enum, list) and enum:
            return enum[0]
    return None
