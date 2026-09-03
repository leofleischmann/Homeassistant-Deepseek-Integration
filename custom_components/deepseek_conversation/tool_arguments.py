"""Reading the ``arguments`` string of a streamed tool call.

A model that emits invalid JSON here used to cost the user the whole turn. The
call was dropped with nothing but a log line, no tool ran, and Assist fell
silent after whatever preamble had already been spoken - "I'm checking the
weather outside", and then nothing. Worse, the assistant turn that was left in
the chat log carried neither text nor tool calls, which the API rejects on the
next request of that conversation.

Two things prevent that now. Malformations models actually produce are
repaired here, and anything left is raised as ``ToolArgumentsError`` so
``stream_transform`` can hand the failure back to the model as a tool result
and let it try the call again, instead of swallowing it.

The repair is deliberately narrow. Guessing at a half-written command is how a
question about the balcony temperature turns into the wrong light being
switched, so only rewrites with a single possible reading are applied:

* a bareword where JSON wants a string (``{"domain": sensor}``), which is the
  malformation that opened issue #33,
* single-quoted strings and Python's ``True`` / ``False`` / ``None``,
* a trailing comma before ``}`` or ``]``,
* a markdown code fence around the whole object,
* arguments encoded as a JSON string containing the real JSON.

Anything else is reported rather than guessed at.
"""

from __future__ import annotations

import json
import re
from typing import Any

from .const import LOGGER

#: How much of the raw text an error message repeats. Long enough to recognise
#: the call in the log, short enough not to flood it - and it travels back to
#: the model as a tool result, where it is charged as input tokens.
_MAX_RAW_IN_ERROR = 200

_STRUCTURAL = "{}[],:"
_QUOTES = "\"'"
_NUMBER = re.compile(r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?")
_PYTHON_LITERALS = {"True": "true", "False": "false", "None": "null"}
_FENCE = re.compile(r"\A```[a-zA-Z0-9_-]*\s*(?P<body>.*?)\s*```\Z", re.DOTALL)


class ToolArgumentsError(ValueError):
    """The arguments of a tool call could not be read, even after repair."""


def parse_tool_arguments(raw: str | None) -> dict[str, Any]:
    """Return the arguments of a tool call as a dict.

    Empty arguments are an empty dict: a tool that takes no parameters is
    called with ``""`` or ``"{}"`` depending on the endpoint.

    Raises ``ToolArgumentsError`` when the text is not a JSON object and the
    repair pass could not make it one.
    """
    text = (raw or "").strip()
    if not text:
        return {}

    candidates = [(text, "")]
    if (repaired := _repair(text)) != text:
        candidates.append((repaired, "repaired"))

    for candidate, note in candidates:
        try:
            value = json.loads(candidate)
        except ValueError:
            continue
        # Some endpoints send the arguments JSON-encoded a second time, so the
        # first decode yields the string that holds the actual object.
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except ValueError:
                pass
        if isinstance(value, dict):
            if note:
                LOGGER.debug(
                    "[Debug conversation]: repaired malformed tool arguments %s -> %s",
                    _excerpt(text),
                    _excerpt(candidate),
                )
            return value
        # Valid JSON of the wrong shape - a list, a number, a bare string.
        # Repairing cannot turn that into arguments, so stop here.
        break

    raise ToolArgumentsError(
        f"the model sent arguments that are not a JSON object: {_excerpt(text)}"
    )


def _excerpt(text: str) -> str:
    """Shorten text for a log line or an error handed back to the model."""
    if len(text) <= _MAX_RAW_IN_ERROR:
        return text
    return f"{text[:_MAX_RAW_IN_ERROR]}… ({len(text)} characters)"


def _repair(text: str) -> str:
    """Apply the narrow rewrites described in the module docstring."""
    if (fenced := _FENCE.match(text)) is not None:
        text = fenced.group("body")
    return _requote(text)


def _requote(text: str) -> str:
    """Rewrite unquoted barewords, single quotes and trailing commas.

    One left-to-right scan that tracks whether it is inside a string, so a
    ``{`` or a bareword-looking run inside a value is left untouched. An
    unterminated string is copied as-is and simply fails to parse afterwards,
    which is the outcome we want: no guessing.
    """
    out: list[str] = []
    index = 0
    length = len(text)
    while index < length:
        char = text[index]
        if char in _QUOTES:
            end = _end_of_string(text, index)
            chunk = text[index:end]
            out.append(chunk if char == '"' else json.dumps(chunk[1:-1] if len(chunk) > 1 else ""))
            index = end
            continue
        if char in _STRUCTURAL:
            if char in "}]":
                _drop_trailing_comma(out)
            out.append(char)
            index += 1
            continue
        if char.isspace():
            out.append(char)
            index += 1
            continue
        end = index
        while (
            end < length
            and text[end] not in _STRUCTURAL
            and text[end] not in _QUOTES
            and not text[end].isspace()
        ):
            end += 1
        out.append(_as_json_literal(text[index:end]))
        index = end
    return "".join(out)


def _end_of_string(text: str, start: int) -> int:
    """Index just past the string starting at ``start``; ``len(text)`` if unterminated."""
    quote = text[start]
    index = start + 1
    while index < len(text):
        if text[index] == "\\":
            index += 2
            continue
        if text[index] == quote:
            return index + 1
        index += 1
    return len(text)


def _drop_trailing_comma(out: list[str]) -> None:
    """Remove a comma that now sits directly before a closing bracket."""
    index = len(out) - 1
    while index >= 0 and out[index].isspace():
        index -= 1
    if index >= 0 and out[index] == ",":
        del out[index]


def _as_json_literal(token: str) -> str:
    """Turn one unquoted run into valid JSON, quoting it when it is a bareword."""
    if token in ("true", "false", "null"):
        return token
    if token in _PYTHON_LITERALS:
        return _PYTHON_LITERALS[token]
    if _NUMBER.fullmatch(token):
        return token
    return json.dumps(token)
