"""Rendering a voluptuous schema as an OpenAPI schema, whichever library core uses.

Home Assistant moved this conversion from ``voluptuous_openapi.convert`` to
``probatio.to_openapi``. The two take the same arguments and behave the same
way with one exception that matters enormously: each asks the caller's
``custom_serializer`` first and returns whatever it answers, unless that answer
is **its own** ``UNSUPPORTED`` sentinel, which it recognises by identity.

That identity check is why they cannot be mixed. Calling one converter with a
serializer built against the other compares two different sentinels, finds them
different, and returns the foreign one as if it were a finished schema. Every
tool then reached the API as ``"parameters": UNSUPPORTED`` and the whole request
died inside the SDK's ``json.dumps`` with ``Object of type _Unsupported is not
JSON serializable`` - so no tool worked at all, on any model, with nothing in
the log naming the cause.

Two rules keep that from happening again, and neither needs to know which
libraries exist:

* the serializer is wrapped so that any answer which is **not a schema** means
  "defer", translated into the sentinel the converter in use recognises. Every
  library's marker is an opaque singleton, so this covers the two that exist
  today and any third one equally, without a list to keep up to date;
* a rendered result is refused unless it survives ``json.dumps`` - the same call
  the SDK makes. A converter asks the serializer for every node and returns that
  answer where the node was, so a marker can be buried inside an otherwise
  ordinary schema, where checking only the top level would not see it.

Converters are then simply tried in turn, core's current one first, which also
settles the separate question of which library understands the schema object
core handed us.
"""

from __future__ import annotations

from collections.abc import Callable
import json
from typing import Any, NamedTuple

from .const import LOGGER


class SchemaConversionError(RuntimeError):
    """A schema could not be rendered as OpenAPI by any installed converter."""


class _Converter(NamedTuple):
    """One installed renderer and the sentinel its serializer contract uses."""

    name: str
    render: Callable[..., Any]
    unsupported: Any


def _available_converters() -> list[_Converter]:
    """The converters this installation has, core's current one first.

    Import errors are the normal case rather than a fault: a core on the old
    library has no ``probatio``, and a core on the new one has no reason to
    still ship ``voluptuous_openapi``.
    """
    converters: list[_Converter] = []
    try:
        from probatio import UNSUPPORTED, to_openapi  # pyright: ignore[reportMissingImports]  # noqa: PLC0415

        converters.append(_Converter("probatio.to_openapi", to_openapi, UNSUPPORTED))
    except ImportError:
        pass
    try:
        from voluptuous_openapi import UNSUPPORTED, convert  # pyright: ignore[reportMissingImports]  # noqa: PLC0415

        converters.append(
            _Converter("voluptuous_openapi.convert", convert, UNSUPPORTED)
        )
    except ImportError:
        pass
    return converters


#: Resolved once at import; the installed set cannot change while HA runs.
CONVERTERS = _available_converters()


def _translating(
    custom_serializer: Callable[..., Any] | None, unsupported: Any
) -> Callable[..., Any] | None:
    """Wrap a serializer so "I cannot render this" is said in the right dialect.

    The contract both libraries define is: answer with a schema object, or with
    the sentinel meaning "you handle it". A schema object is always a dict, so
    anything else is the second answer however it is spelled - the other
    library's sentinel, a future third one, or a ``None`` from a serializer that
    reads the contract differently. All of them become the sentinel this
    converter checks for, which is the only value it will not hand back as a
    schema.
    """
    if custom_serializer is None:
        return None

    def translate(schema: Any) -> Any:
        answer = custom_serializer(schema)
        return answer if isinstance(answer, dict) else unsupported

    return translate


def _unsendable(rendered: Any) -> str | None:
    """Why ``rendered`` must not be sent, or ``None`` if it may be.

    ``json.dumps`` is the same call the OpenAI SDK makes on the finished
    request, so a schema that passes here reaches the API intact - and one that
    does not costs a skipped tool with its name in the log, rather than a whole
    request failing on a ``TypeError`` raised deep inside the SDK.
    """
    if not isinstance(rendered, dict):
        return f"returned {rendered!r} instead of a schema"
    try:
        json.dumps(rendered)
    except (TypeError, ValueError) as err:
        return f"returned a schema that cannot be sent as JSON ({err})"
    return None


def render_openapi_schema(
    schema: Any,
    *,
    custom_serializer: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Return ``schema`` as an OpenAPI schema dict.

    Raises ``SchemaConversionError`` if no installed converter produced one, so
    a caller can skip the tool rather than send something the SDK cannot
    serialise.
    """
    problems: list[str] = []
    for converter in CONVERTERS:
        try:
            rendered = converter.render(
                schema,
                custom_serializer=_translating(
                    custom_serializer, converter.unsupported
                ),
            )
        except Exception as err:  # noqa: BLE001 - any failure means "try the next one"
            problems.append(f"{converter.name} raised {type(err).__name__}: {err}")
            continue
        if (refusal := _unsendable(rendered)) is not None:
            problems.append(f"{converter.name} {refusal}")
            continue
        if problems:
            LOGGER.debug(
                "[Debug conversation]: rendered the schema with %s after %s",
                converter.name,
                "; ".join(problems),
            )
        return rendered

    if not CONVERTERS:
        problems.append("neither probatio nor voluptuous_openapi is installed")
    raise SchemaConversionError("; ".join(problems))
