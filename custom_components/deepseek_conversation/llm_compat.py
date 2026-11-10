"""Home Assistant's tool-result types, across the cores this integration supports.

The README promises Home Assistant 2026.1 and later, and how a tool result
travels changed inside that range. Up to 2026.9 it is a bare JSON object
everywhere: what a tool returns, what ``ToolResultContent.tool_result`` holds
and what a ``tool_result`` delta carries. 2026.10 introduced ``llm.ToolResult``
and moved all three onto it - the old spellings still work there, but core
reports each use and refuses them from 2027.11.

Every question of the form "which shape does this core want" is answered here,
one small function per question, and each asks core for the feature rather than
for its version number. No other module carries such a check.

Affects / influenced by:
- ``chat_messages.py`` reads a result out of the chat log with
  ``tool_result_data`` before sending it back to the API.
- ``stream_transform.py`` reports unreadable tool arguments to the model with
  ``tool_result_delta``.
- ``web_search.py`` answers with ``tool_result``.
- ``tests/test_real_home_assistant.py`` pins all three against whichever core
  is installed; the stubbed suite only ever sees the pre-2026.10 shapes.
"""

from __future__ import annotations

from typing import Any

from homeassistant.components import conversation  # pyright: ignore[reportMissingImports]
from homeassistant.helpers import llm  # pyright: ignore[reportMissingImports]


def tool_result(data: dict[str, Any], *, error: bool = False) -> Any:
    """Wrap a tool payload in whatever this core expects back from a tool.

    ``Any`` rather than a union because naming ``llm.ToolResult`` in the
    signature would be a dangling reference on exactly the cores that predate
    it. ``error`` is dropped there too: a bare object has nowhere to put it,
    and the payload's own ``error`` key is what the model reads either way.
    """
    if (result_type := getattr(llm, "ToolResult", None)) is None:
        return data
    return result_type(data=data, error=error)


def tool_result_data(content: Any) -> Any:
    """The payload of a chat log's ``ToolResultContent``, on any core.

    From 2026.10 the content carries an ``llm.ToolResult`` in ``result`` and
    keeps ``tool_result`` only as a property that reports every access. Asking
    for ``result`` first is what keeps that report out of the log; the cores
    before it have no such attribute and hold the payload in ``tool_result``
    itself.
    """
    if (result := getattr(content, "result", None)) is not None:
        return result.data
    return content.tool_result


def _delta_carries_result() -> bool:
    """Whether this core's chat log reads ``result`` from a tool result delta.

    Asked of the delta's own type. The stubbed suite has no such type at all,
    which answers the same as an old core: no.
    """
    delta_type = getattr(conversation, "ToolResultContentDeltaDict", None)
    keys = getattr(delta_type, "__required_keys__", frozenset()) | getattr(
        delta_type, "__optional_keys__", frozenset()
    )
    return "result" in keys


def tool_result_delta(
    *, tool_call_id: str, tool_name: str, data: dict[str, Any], error: bool = False
) -> dict[str, Any]:
    """A tool result delta in the shape this core's chat log reads.

    Used for results this integration writes itself instead of letting core
    run the tool - today the answer to a call whose arguments would not parse.
    """
    delta: dict[str, Any] = {
        "role": "tool_result",
        "tool_call_id": tool_call_id,
        "tool_name": tool_name,
    }
    if _delta_carries_result():
        delta["result"] = tool_result(data, error=error)
    else:
        delta["tool_result"] = data
    return delta
