"""Speaker identity for the system prompt.

Home Assistant resolves exactly one speaker variable when it renders an agent's
system prompt: ``user_name`` (see ``ChatLog._async_expand_prompt_template``).
Two things make that insufficient here:

* Voice satellites run the pipeline with a bare ``Context()``, so there is no
  ``user_id`` to resolve and ``user_name`` is ``None`` — a prompt containing
  ``{{ user_name }}`` then renders the literal string ``None``. That is the
  exact setup (a household talking to satellites) this is meant to serve.
* There is no way to reach the user id, the matching ``person`` entity or the
  area of the satellite that was spoken to.

This module resolves the speaker once per turn and exposes the result twice:

``SpeakerContext.jinja_preamble()``
    ``{% set %}`` statements prepended to the configured prompt so
    ``{{ user_id }}``, ``{{ user_area }}`` and friends are real template
    variables. Home Assistant renders the combined string, so there is no
    second render pass and a user name can never inject Jinja — values are
    embedded as JSON string literals, which Jinja parses as plain strings.

``SpeakerContext.facts_prompt()``
    A short plain-text block for people who do not want to write a template.
    It is passed through ``user_extra_system_prompt``, which Home Assistant
    appends **last**, after the API prompt with the exposed-entity list. That
    placement keeps the large, speaker-independent part of the system prompt
    identical for everyone, which matters for DeepSeek's prefix cache.

Unresolved values are always empty strings, never ``None``, so ``{% if
user_name %}`` works and nothing can render as ``None``.
"""

from __future__ import annotations

from dataclasses import dataclass
import json

from homeassistant.core import Context, HomeAssistant  # pyright: ignore[reportMissingImports]
from homeassistant.exceptions import TemplateError  # pyright: ignore[reportMissingImports]
from homeassistant.helpers import (  # pyright: ignore[reportMissingImports]
    area_registry as ar,
    device_registry as dr,
    floor_registry as fr,
    llm,
    template,
)

from .const import DOMAIN, LOGGER

PERSON_DOMAIN = "person"

#: Template variables provided to the system prompt, in preamble order.
#: Every name here is always defined; unknown values are empty strings.
USER_CONTEXT_VARS: tuple[str, ...] = (
    "user_id",
    "user_name",
    "user_is_admin",
    "person_entity_id",
    "person_name",
    "person_state",
    "device_id",
    "device_name",
    "user_area",
    "user_floor",
)

#: First line of :meth:`SpeakerContext.facts_prompt`. Also the marker used by
#: :func:`strip_speaker_block` to recover a caller's own extra system prompt
#: from the combined value Home Assistant persists on the chat log.
SPEAKER_BLOCK_HEADER = "Speaker context (provided by Home Assistant):"

_SPEAKER_BLOCK_GUIDANCE = (
    "Use this to personalise replies and to resolve possessives such as "
    '"my room". Never invent a name or a room that is not listed above.'
)

#: Presence states that carry no information worth spending tokens on.
_UNUSABLE_STATES = frozenset({"unknown", "unavailable", ""})

#: ``person`` states read as identifiers; phrase them for a prompt instead.
#: Anything else is a zone name and is used as-is ("at Work").
_PRESENCE_PHRASES = {"home": "at home", "not_home": "away from home"}


@dataclass(frozen=True, slots=True)
class SpeakerContext:
    """Who is talking, and from where.

    ``values`` holds one entry per :data:`USER_CONTEXT_VARS`, always a string.
    """

    values: dict[str, str]

    @property
    def has_user(self) -> bool:
        """Whether Home Assistant identified the speaking user."""
        return bool(self.values.get("user_id"))

    @property
    def has_location(self) -> bool:
        """Whether the satellite that was spoken to is assigned to an area."""
        return bool(self.values.get("user_area"))

    def jinja_preamble(self) -> str:
        """Return ``{% set %}`` statements defining every context variable.

        Kept to a single line without trailing whitespace control so that line
        numbers in Home Assistant's template error messages still match the
        prompt the user actually typed.
        """
        return "".join(
            "{%- set " + name + " = " + _jinja_literal(self.values.get(name, "")) + " %}"
            for name in USER_CONTEXT_VARS
        )

    def apply_to_prompt(self, prompt: str) -> str:
        """Return ``prompt`` with the variable preamble prepended."""
        return self.jinja_preamble() + prompt

    def facts_prompt(self) -> str:
        """Return a plain-text speaker block, or ``""`` when nothing is known."""
        lines: list[str] = []

        if name := self.values.get("user_name"):
            lines.append(f"- The person speaking is {name}.")
        if presence := _presence_phrase(self.values.get("person_state", "")):
            lines.append(f"- They are currently {presence}.")

        area = self.values.get("user_area")
        device_name = self.values.get("device_name")
        if area and device_name:
            location = f"- They are talking to {device_name}, which is in {area}"
            if floor := self.values.get("user_floor"):
                location += f" on {floor}"
            lines.append(location + ".")
        elif area:
            lines.append(f"- They are talking to a device in {area}.")

        if not lines:
            return ""

        return "\n".join([SPEAKER_BLOCK_HEADER, *lines, _SPEAKER_BLOCK_GUIDANCE])


EMPTY_SPEAKER_CONTEXT = SpeakerContext(values={name: "" for name in USER_CONTEXT_VARS})


def _presence_phrase(state: str) -> str:
    """Render a ``person`` state as prose, or ``""`` when it says nothing."""
    if state in _UNUSABLE_STATES:
        return ""
    return _PRESENCE_PHRASES.get(state, f"at {state}")


def _jinja_literal(value: str) -> str:
    """Return ``value`` as a Jinja string literal that cannot break out.

    ``json.dumps`` escaping is a subset of the escaping Jinja's lexer accepts,
    so ``{{``/``{%`` inside a name stay inert text. ``ensure_ascii=False`` is
    deliberate: with ``True`` a non-BMP character is emitted as a surrogate
    pair, which Jinja's ``unicode-escape`` decoding turns into lone surrogates.
    """
    return json.dumps(value, ensure_ascii=False)


async def async_build_speaker_context(
    hass: HomeAssistant, llm_context: llm.LLMContext
) -> SpeakerContext:
    """Resolve the speaker behind ``llm_context``.

    Never raises: a registry lookup that fails degrades to an empty value so a
    conversation is never lost over prompt decoration.
    """
    values = {name: "" for name in USER_CONTEXT_VARS}

    try:
        await _async_add_user(hass, llm_context, values)
        _add_device_location(hass, llm_context, values)
    except Exception:  # noqa: BLE001 - never fail a conversation over this
        LOGGER.exception("[Debug user_context]: failed to resolve speaker context")

    return SpeakerContext(values=values)


async def _async_add_user(
    hass: HomeAssistant, llm_context: llm.LLMContext, values: dict[str, str]
) -> None:
    """Fill in user and person fields from the context's user id."""
    context = llm_context.context
    user_id = context.user_id if context else None
    if not user_id:
        return

    values["user_id"] = user_id

    if user := await hass.auth.async_get_user(user_id):
        values["user_name"] = user.name or ""
        values["user_is_admin"] = "true" if user.is_admin else ""

    for state in hass.states.async_all(PERSON_DOMAIN):
        if state.attributes.get("user_id") != user_id:
            continue
        values["person_entity_id"] = state.entity_id
        values["person_name"] = state.attributes.get("friendly_name") or ""
        values["person_state"] = state.state
        # A user maps to at most one person entity.
        break


def _add_device_location(
    hass: HomeAssistant, llm_context: llm.LLMContext, values: dict[str, str]
) -> None:
    """Fill in device, area and floor for the satellite that was spoken to."""
    device_id = llm_context.device_id
    if not device_id:
        return

    values["device_id"] = device_id

    device = dr.async_get(hass).async_get(device_id)
    if device is None:
        return

    values["device_name"] = device.name_by_user or device.name or ""

    if not device.area_id:
        return

    area = ar.async_get(hass).async_get_area(device.area_id)
    if area is None:
        return

    values["user_area"] = area.name

    if area.floor_id and (floor := fr.async_get(hass).async_get_floor(area.floor_id)):
        values["user_floor"] = floor.name


async def async_render_standalone_prompt(
    hass: HomeAssistant, prompt: str, call_context: Context | None
) -> str:
    """Render the configured prompt outside a chat log, for ``generate_content``.

    That service talks to the API directly, so nothing renders the prompt for
    it: without this a configured ``{{ user_name }}`` would reach the model as
    literal text. The variables match the Assist path, with ``ha_name`` and
    ``llm_context`` supplied here because Home Assistant only provides those
    when it renders a chat log's prompt itself.

    A broken template degrades to the unrendered prompt rather than failing the
    service call - the same prompt already reached the API unrendered before.
    """
    llm_context = llm.LLMContext(
        platform=DOMAIN,
        context=call_context,
        language=None,
        assistant=DOMAIN,
        device_id=None,
    )
    speaker = await async_build_speaker_context(hass, llm_context)

    try:
        return template.Template(
            speaker.apply_to_prompt(prompt), hass
        ).async_render(
            {
                "ha_name": hass.config.location_name,
                "user_name": speaker.values["user_name"] or None,
                "llm_context": llm_context,
            },
            parse_result=False,
        )
    except TemplateError:
        LOGGER.exception(
            "[Debug user_context]: generate_content prompt template failed; "
            "sending it unrendered"
        )
        return prompt


def strip_speaker_block(text: str | None) -> str | None:
    """Return ``text`` without a previously appended speaker block.

    Home Assistant persists the combined ``extra_system_prompt`` on the chat log
    and reuses it on later turns, when ``ConversationInput.extra_system_prompt``
    is ``None``. Without this the caller's own extra prompt would be lost as
    soon as the speaker block replaced it.
    """
    if not text:
        return None
    return text.split(SPEAKER_BLOCK_HEADER, 1)[0].rstrip() or None


def merge_extra_system_prompt(
    caller_extra: str | None, speaker_block: str | None
) -> str | None:
    """Combine a caller's extra system prompt with the speaker block."""
    parts = [part for part in (caller_extra, speaker_block) if part]
    return "\n".join(parts) if parts else None
