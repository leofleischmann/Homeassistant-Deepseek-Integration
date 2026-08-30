"""Shared vision/image encoding for Assist and generate_content.

Used by chat_messages.py (UserContent.attachments from Assist / AI Task) and
services.py (generate_content filenames). Images are sent as OpenAI-style
``image_url`` content parts with a base64 ``data:`` URL.

Whether that is accepted depends on the **model**, not on the endpoint: the
official API serves ``deepseek-v4-flash-vision-exp`` for image input and rejects
``image_url`` parts everywhere else. A custom OpenAI-compatible gateway has a
catalogue we cannot know - a DeepSeek model name there may be routed to any
backend - so nothing is refused for those and the API answers for itself. Option
CONF_VISION_ENABLED gates the feature entirely; see flow_schemas.py.
"""

from __future__ import annotations

import base64
from collections.abc import Mapping
from mimetypes import guess_file_type
from pathlib import Path
from typing import Any

from homeassistant.components import ai_task, conversation  # pyright: ignore[reportMissingImports]
from homeassistant.core import HomeAssistant  # pyright: ignore[reportMissingImports]
from homeassistant.exceptions import HomeAssistantError  # pyright: ignore[reportMissingImports]

from .const import (
    CONF_CHAT_MODEL,
    CONF_VISION_ENABLED,
    DEFAULT_VISION_ENABLED,
    LOGGER,
    RECOMMENDED_CHAT_MODEL,
    VISION_CHAT_MODEL,
    VISION_CHAT_MODELS,
)
from .models import is_official_deepseek_api_base_url, normalize_model_id


def model_supports_vision(model: str | None, *, base_url: str | None = None) -> bool:
    """Whether ``model`` accepts image input on the endpoint in ``base_url``.

    Only the official API has a catalogue worth checking against, and it takes
    images on exactly one model. A custom gateway is left alone on purpose:
    refusing an id we merely recognise would break a setup that maps, say,
    ``deepseek-chat`` onto a multimodal backend, and the cost of being wrong the
    other way is one clear API error (see api_errors.py).
    """
    if not is_official_deepseek_api_base_url(base_url):
        return True
    return normalize_model_id(model) in VISION_CHAT_MODELS


def raise_if_vision_unsupported(model: str | None, *, base_url: str | None) -> None:
    """Fail fast before encoding images when the model cannot accept them."""
    if model_supports_vision(model, base_url=base_url):
        return
    LOGGER.debug(
        "[Debug vision]: blocked image input for model=%r base_url=%r",
        model,
        base_url,
    )
    name = (model or "").strip() or "the configured model"
    raise HomeAssistantError(
        f"The model {name} does not accept image input. On the official DeepSeek "
        f"API only {VISION_CHAT_MODEL} takes images - set it as this agent's "
        "model, or point the base URL at a multimodal OpenAI-compatible gateway "
        "(integration card, Reconfigure)."
    )


# ConversationEntityFeature has no attachment flag yet; getattr lets the entity
# advertise one the day Home Assistant adds it, without breaking until then.
# ai_task is not guarded: this integration sets up Platform.AI_TASK and ai_task.py
# imports the component outright, so a core without it never gets this far.
CONVERSATION_SUPPORT_ATTACHMENTS = getattr(
    conversation.ConversationEntityFeature, "SUPPORT_ATTACHMENTS", None
)
AI_TASK_SUPPORT_ATTACHMENTS = ai_task.AITaskEntityFeature.SUPPORT_ATTACHMENTS


def vision_enabled_in_options(options: Mapping[str, Any]) -> bool:
    """Whether image input is allowed for this config entry."""
    return bool(options.get(CONF_VISION_ENABLED, DEFAULT_VISION_ENABLED))


def vision_available(options: Mapping[str, Any], *, base_url: str | None) -> bool:
    """Whether this entry can actually send images right now.

    Both halves have to hold: the option is on **and** the configured model
    accepts images. Advertising SUPPORT_ATTACHMENTS on a text-only model gives
    the user an attachment button whose every use ends in an error.
    """
    if not vision_enabled_in_options(options):
        return False
    return model_supports_vision(
        options.get(CONF_CHAT_MODEL, RECOMMENDED_CHAT_MODEL), base_url=base_url
    )


def conversation_entity_features_for_options(
    options: Mapping[str, Any],
    *,
    has_control: bool,
    base_url: str | None,
) -> conversation.ConversationEntityFeature:
    """Build ``ConversationEntityFeature`` flags from entry options."""
    features = conversation.ConversationEntityFeature(0)
    if has_control:
        features |= conversation.ConversationEntityFeature.CONTROL
    if (
        vision_available(options, base_url=base_url)
        and CONVERSATION_SUPPORT_ATTACHMENTS is not None
    ):
        features |= CONVERSATION_SUPPORT_ATTACHMENTS
    return features


def ai_task_entity_features_for_options(
    options: Mapping[str, Any],
    *,
    base_url: str | None,
) -> ai_task.AITaskEntityFeature:
    """Build ``AITaskEntityFeature`` flags from entry options."""
    features = ai_task.AITaskEntityFeature.GENERATE_DATA
    if vision_available(options, base_url=base_url):
        features |= AI_TASK_SUPPORT_ATTACHMENTS
    return features


def encode_file_path(file_path: str | Path) -> tuple[str, str, int]:
    """Return ``(mime_type, base64_data, raw_byte_count)`` for a local file."""
    path = Path(file_path)
    mime_type, _ = guess_file_type(str(path))
    if mime_type is None:
        mime_type = "application/octet-stream"
    raw = path.read_bytes()
    return mime_type, base64.b64encode(raw).decode("utf-8"), len(raw)


def image_url_content_part(mime_type: str, base64_data: str) -> dict[str, Any]:
    """Build one DeepSeek/OpenAI ``image_url`` content part."""
    return {
        "type": "image_url",
        "image_url": {"url": f"data:{mime_type};base64,{base64_data}"},
    }


def latest_user_attachments(
    content_list: list[conversation.Content],
) -> list[conversation.Attachment] | None:
    """Return image attachments from the most recent user turn, if any.

    Only the current request's attachments matter: earlier attachments are
    already encoded into prior API rounds, so scanning the whole history would
    re-encode them every turn and could block a later text-only follow-up turn.
    """
    for content in reversed(content_list):
        if isinstance(content, conversation.UserContent):
            return content.attachments or None
    return None


def _normalize_mime_type(file_path: Path, mime_type: str | None) -> str:
    if mime_type:
        return mime_type
    guessed, _ = guess_file_type(str(file_path))
    return guessed or "application/octet-stream"


def _image_part_from_path(
    file_path: Path, mime_type: str | None
) -> tuple[dict[str, Any], int]:
    resolved_mime = _normalize_mime_type(file_path, mime_type)
    if not resolved_mime.startswith("image/"):
        raise HomeAssistantError(
            f"Only image attachments are supported, got {resolved_mime} for `{file_path}`"
        )
    # Prefer the HA-provided/resolved mime over encode_file_path's extension guess,
    # so an attachment with an authoritative mime_type but no/odd file extension is
    # still sent with the correct image type in the data URL.
    _guessed_mime, base64_data, byte_count = encode_file_path(file_path)
    return image_url_content_part(resolved_mime, base64_data), byte_count


def _read_image_parts_from_paths(
    files: list[tuple[Path, str | None]],
    strict: bool,
) -> tuple[list[dict[str, Any]], int]:
    parts: list[dict[str, Any]] = []
    total_bytes = 0
    for file_path, mime_type in files:
        if not file_path.exists():
            message = f"`{file_path}` does not exist"
            if strict:
                raise HomeAssistantError(message)
            LOGGER.warning("[Debug vision]: %s", message)
            continue
        resolved_mime = _normalize_mime_type(file_path, mime_type)
        if not resolved_mime.startswith("image/"):
            message = (
                f"Skipping `{file_path}`: unsupported type {resolved_mime} "
                "(only image/* is supported)"
            )
            if strict:
                raise HomeAssistantError(
                    f"Only image attachments are supported, got {resolved_mime}"
                )
            LOGGER.warning("[Debug vision]: %s", message)
            continue
        part, byte_count = _image_part_from_path(file_path, mime_type)
        parts.append(part)
        total_bytes += byte_count
    return parts, total_bytes


async def async_image_parts_from_paths(
    hass: HomeAssistant,
    files: list[tuple[Path, str | None]],
    *,
    strict: bool,
) -> list[dict[str, Any]]:
    """Encode local files to DeepSeek image content parts.

    ``strict=True`` (Assist): raise on missing or non-image files.
    ``strict=False`` (generate_content service): log and skip disallowed files.
    """
    parts, total_bytes = await hass.async_add_executor_job(
        _read_image_parts_from_paths, files, strict
    )
    if parts:
        LOGGER.debug(
            "[Debug vision]: %d attachments, %d total bytes",
            len(parts),
            total_bytes,
        )
    return parts


async def async_image_parts_from_attachments(
    hass: HomeAssistant,
    attachments: list[conversation.Attachment],
) -> list[dict[str, Any]]:
    """Encode HA ``Attachment`` objects from Assist / AI Task chat logs."""
    files = [(attachment.path, attachment.mime_type) for attachment in attachments]
    return await async_image_parts_from_paths(hass, files, strict=True)


async def async_image_parts_from_filenames(
    hass: HomeAssistant,
    filenames: list[str],
) -> list[dict[str, Any]]:
    """Encode ``generate_content`` ``filenames`` paths.

    Raises ``HomeAssistantError`` when paths were given but no image could be
    encoded (missing file, disallowed path, or non-image type). Used from services.py.
    """
    files: list[tuple[Path, str | None]] = []
    disallowed: list[str] = []
    for filename in filenames:
        if not hass.config.is_allowed_path(filename):
            disallowed.append(filename)
            LOGGER.warning(
                "[Debug vision]: cannot read %s, path not allowed; "
                "adjust allowlist_external_dirs in configuration.yaml",
                filename,
            )
            continue
        files.append((Path(filename), None))

    if disallowed and not files:
        raise HomeAssistantError(
            "No filename path is allowed for Home Assistant to read. Add the "
            "directory to allowlist_external_dirs in configuration.yaml "
            f"(blocked: {', '.join(disallowed)})"
        )

    parts, total_bytes = await hass.async_add_executor_job(
        _read_image_parts_from_paths, files, False
    )
    if not parts:
        paths = ", ".join(str(path) for path, _ in files) or ", ".join(filenames)
        raise HomeAssistantError(
            "Could not read any image from filenames. Check that each file "
            f"exists and is an image (JPEG, PNG, …): {paths}"
        )

    LOGGER.debug(
        "[Debug vision]: %d attachment(s) from filenames, %d total bytes",
        len(parts),
        total_bytes,
    )
    return parts


async def async_user_message_content(
    hass: HomeAssistant,
    text: str,
    attachments: list[conversation.Attachment] | None,
) -> str | list[dict[str, Any]]:
    """Build ``content`` for a DeepSeek user message (plain text or parts array)."""
    if not attachments:
        return text

    parts: list[dict[str, Any]] = []
    if text.strip():
        parts.append({"type": "text", "text": text})
    parts.extend(await async_image_parts_from_attachments(hass, attachments))
    if not parts:
        raise HomeAssistantError("Image attachment could not be read")
    return parts
