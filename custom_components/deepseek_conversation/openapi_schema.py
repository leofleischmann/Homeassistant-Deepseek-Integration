"""Rendering a voluptuous schema as an OpenAPI schema, whichever library core uses.

Home Assistant moved this conversion from ``voluptuous_openapi.convert`` to
``probatio.to_openapi``. The two take the same arguments and behave the same
way with one exception that matters enormously: each asks the caller's
``custom_serializer`` first and then compares its answer against **its own**
``UNSUPPORTED`` sentinel, by identity.

That identity check is why they cannot be mixed. Calling the old converter on a
core that hands out the new serializer compares ``probatio.UNSUPPORTED`` against
``voluptuous_openapi.UNSUPPORTED``, finds them different, and returns the
foreign sentinel as if it were a finished schema. Every tool then reached the
API as ``"parameters": UNSUPPORTED`` and the whole request died inside the SDK's
``json.dumps`` with ``Object of type _Unsupported is not JSON serializable`` -
so no tool worked at all, on any model, with nothing in the log naming the
cause.

Two things make that impossible here rather than one. The serializer is wrapped
so that *any* library's "I cannot render this" answer is translated into the
sentinel the converter actually being used checks for, which removes the
mismatch at its source; and a result that is not a dict is refused whatever
produced it, so a sentinel can never reach the request even if some future
library brings a third one. Converters are then simply tried in turn, core's
current one first, which also settles the separate question of which library
understands the schema object core handed us.
"""

from __future__ import annotations

from collections.abc import Callable
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

#: Every "cannot render this" marker we know, for recognising a foreign one.
KNOWN_UNSUPPORTED = tuple(converter.unsupported for converter in CONVERTERS)


def _is_unsupported(value: Any) -> bool:
    """Whether a serializer's answer is any library's UNSUPPORTED marker."""
    return any(value is sentinel for sentinel in KNOWN_UNSUPPORTED)


def _translating(
    custom_serializer: Callable[..., Any] | None, unsupported: Any
) -> Callable[..., Any] | None:
    """Wrap a serializer so its "unsupported" answer is one this converter knows.

    Without this the answer is compared by identity against a sentinel from a
    different library, silently succeeds, and the marker object is returned as
    the schema. A real schema is passed through untouched.
    """
    if custom_serializer is None:
        return None

    def translate(schema: Any) -> Any:
        answer = custom_serializer(schema)
        return unsupported if _is_unsupported(answer) else answer

    return translate


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
        if isinstance(rendered, dict):
            if problems:
                LOGGER.debug(
                    "[Debug conversation]: rendered the schema with %s after %s",
                    converter.name,
                    "; ".join(problems),
                )
            return rendered
        problems.append(f"{converter.name} returned {rendered!r} instead of a schema")

    if not CONVERTERS:
        problems.append("neither probatio nor voluptuous_openapi is installed")
    raise SchemaConversionError("; ".join(problems))
