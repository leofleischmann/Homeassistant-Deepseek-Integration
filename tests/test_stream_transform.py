"""Tests for the two pure helpers in stream_transform.

Both sit on the path every streamed token takes to the user, and both fail
silently when they are wrong: text that is dropped here is simply never spoken
and never shown, with nothing in the log to say so. That is what makes them
worth covering even though the module around them talks to the API.

Pure standard library, so they run without Home Assistant installed::

    python -m unittest discover -s tests
"""

from __future__ import annotations

import unittest

from integration_modules import load

stream_transform = load("stream_transform", api_stubs=True)

delta_text = stream_transform._stream_delta_text
yield_deltas = stream_transform._yield_assistant_text_deltas


class _Delta:
    """Stand-in for the SDK's ChoiceDelta.

    ``model_extra`` is where the OpenAI SDK parks JSON fields its model does
    not declare - which is where DeepSeek's text sometimes turns up.
    """

    def __init__(self, model_extra: dict[str, object] | None = None, **attrs: object) -> None:
        self.model_extra = model_extra
        for key, value in attrs.items():
            setattr(self, key, value)


class TestStreamDeltaText(unittest.TestCase):
    def test_reads_the_declared_attribute(self) -> None:
        self.assertEqual(delta_text(_Delta(content="hello"), "content"), "hello")
        self.assertEqual(
            delta_text(_Delta(reasoning_content="why"), "reasoning_content"), "why"
        )

    def test_nothing_to_say_is_none_not_empty_string(self) -> None:
        """HA appends content only when truthy, so "" and None are the same thing."""
        self.assertIsNone(delta_text(_Delta(content=""), "content"))
        self.assertIsNone(delta_text(_Delta(content=None), "content"))
        self.assertIsNone(delta_text(_Delta(), "content"))

    def test_falls_back_to_model_extra(self) -> None:
        """A gateway can send a field the SDK model does not map to an attribute."""
        self.assertEqual(
            delta_text(_Delta(model_extra={"content": "hello"}), "content"), "hello"
        )
        self.assertEqual(
            delta_text(_Delta(content="", model_extra={"content": "hello"}), "content"),
            "hello",
        )
        self.assertEqual(
            delta_text(
                _Delta(content=None, model_extra={"content": "hello"}), "content"
            ),
            "hello",
        )

    def test_the_attribute_wins_when_both_carry_text(self) -> None:
        self.assertEqual(
            delta_text(
                _Delta(content="attr", model_extra={"content": "extra"}), "content"
            ),
            "attr",
        )

    def test_a_content_part_list_is_joined(self) -> None:
        self.assertEqual(
            delta_text(
                _Delta(content=[{"type": "text", "text": "one "}, "two"]), "content"
            ),
            "one two",
        )

    def test_non_text_parts_are_skipped(self) -> None:
        self.assertEqual(
            delta_text(
                _Delta(
                    content=[
                        {"type": "image_url", "image_url": {"url": "x"}},
                        {"type": "text", "text": "kept"},
                        {"type": "text"},
                    ]
                ),
                "content",
            ),
            "kept",
        )

    def test_a_list_with_no_text_is_none(self) -> None:
        self.assertIsNone(delta_text(_Delta(content=[]), "content"))
        self.assertIsNone(
            delta_text(_Delta(content=[{"type": "image_url"}]), "content")
        )

    def test_unusable_types_are_none(self) -> None:
        for value in (7, {"content": "x"}, object()):
            with self.subTest(value=type(value).__name__):
                self.assertIsNone(delta_text(_Delta(content=value), "content"))

    def test_a_missing_model_extra_is_not_an_error(self) -> None:
        self.assertIsNone(delta_text(_Delta(model_extra=None), "content"))


class TestYieldAssistantTextDeltas(unittest.TestCase):
    """The role must ride along on the first delta that carries text."""

    def test_first_text_delta_opens_the_assistant_message(self) -> None:
        deltas, emitted = yield_deltas(
            role_emitted=False, content_delta="hi", reasoning_delta=None
        )
        self.assertEqual(deltas, [{"role": "assistant", "content": "hi"}])
        self.assertTrue(emitted)

    def test_reasoning_alone_also_opens_it(self) -> None:
        deltas, emitted = yield_deltas(
            role_emitted=False, content_delta=None, reasoning_delta="why"
        )
        self.assertEqual(deltas, [{"role": "assistant", "thinking_content": "why"}])
        self.assertTrue(emitted)

    def test_both_kinds_share_the_opening_delta(self) -> None:
        deltas, emitted = yield_deltas(
            role_emitted=False, content_delta="hi", reasoning_delta="why"
        )
        self.assertEqual(
            deltas, [{"role": "assistant", "content": "hi", "thinking_content": "why"}]
        )
        self.assertTrue(emitted)

    def test_an_empty_delta_does_not_burn_the_role(self) -> None:
        """Otherwise the first real text arrives without a role and is dropped."""
        deltas, emitted = yield_deltas(
            role_emitted=False, content_delta=None, reasoning_delta=None
        )
        self.assertEqual(deltas, [])
        self.assertFalse(emitted)

    def test_later_deltas_carry_no_role(self) -> None:
        deltas, emitted = yield_deltas(
            role_emitted=True, content_delta="more", reasoning_delta=None
        )
        self.assertEqual(deltas, [{"content": "more"}])
        self.assertTrue(emitted)

    def test_later_deltas_split_text_and_reasoning(self) -> None:
        deltas, _emitted = yield_deltas(
            role_emitted=True, content_delta="more", reasoning_delta="why"
        )
        self.assertEqual(deltas, [{"content": "more"}, {"thinking_content": "why"}])

    def test_an_empty_string_is_never_yielded(self) -> None:
        """Assist reads a falsy content as "no message" and drops the turn."""
        self.assertEqual(
            yield_deltas(role_emitted=True, content_delta="", reasoning_delta="")[0], []
        )
        deltas, emitted = yield_deltas(
            role_emitted=False, content_delta="", reasoning_delta=""
        )
        self.assertEqual(deltas, [])
        self.assertFalse(emitted)


class _ToolCallChunk:
    """Stand-in for one entry of ``ChoiceDelta.tool_calls``.

    A gateway is free to spread ``id``, the function name and the arguments
    over as many chunks as it likes, and to send any subset in any one of
    them - which is exactly what these tests build.
    """

    def __init__(
        self,
        index: int | None,
        *,
        id: str | None = None,
        name: str | None = None,
        arguments: str | None = None,
        type: str | None = None,
    ) -> None:
        self.index = index
        self.id = id
        self.type = type
        self.function = (
            _ToolCallFunction(name, arguments)
            if name is not None or arguments is not None
            else None
        )


class _ToolCallFunction:
    def __init__(self, name: str | None, arguments: str | None) -> None:
        self.name = name
        self.arguments = arguments


def _assemble(*chunks: _ToolCallChunk) -> dict[int, dict[str, object]]:
    """Run the chunks through the merge step the stream performs per delta."""
    calls: dict[int, dict[str, object]] = {}
    for chunk in chunks:
        stream_transform._merge_tool_call_chunk(calls, chunk)
    return calls


class TestMergeToolCallChunk(unittest.TestCase):
    """Assembling one tool call out of however many chunks it arrives in."""

    def test_arguments_are_concatenated_in_order(self) -> None:
        calls = _assemble(
            _ToolCallChunk(0, id="call_1", name="GetLiveContext"),
            _ToolCallChunk(0, arguments='{"dom'),
            _ToolCallChunk(0, arguments='ain": "sensor"}'),
        )
        self.assertEqual(calls[0]["arguments"], '{"domain": "sensor"}')

    def test_an_id_arriving_late_still_belongs_to_the_call(self) -> None:
        """Requiring it in the opening chunk used to drop the whole call."""
        calls = _assemble(
            _ToolCallChunk(0, name="GetLiveContext"),
            _ToolCallChunk(0, id="call_1"),
            _ToolCallChunk(0, arguments="{}"),
        )
        self.assertEqual(calls[0]["id"], "call_1")
        self.assertEqual(calls[0]["name"], "GetLiveContext")

    def test_a_name_arriving_late_still_belongs_to_the_call(self) -> None:
        calls = _assemble(
            _ToolCallChunk(0, id="call_1"),
            _ToolCallChunk(0, name="GetLiveContext"),
            _ToolCallChunk(0, arguments='{"domain": "sensor"}'),
        )
        self.assertEqual(calls[0]["name"], "GetLiveContext")
        self.assertEqual(calls[0]["arguments"], '{"domain": "sensor"}')

    def test_arguments_before_the_name_are_not_lost(self) -> None:
        calls = _assemble(
            _ToolCallChunk(0, arguments='{"domain": '),
            _ToolCallChunk(0, id="call_1", name="GetLiveContext"),
            _ToolCallChunk(0, arguments='"sensor"}'),
        )
        self.assertEqual(calls[0]["arguments"], '{"domain": "sensor"}')

    def test_type_defaults_to_function(self) -> None:
        """Several OpenAI-compatible gateways leave it out entirely."""
        calls = _assemble(_ToolCallChunk(0, id="call_1", name="GetLiveContext"))
        self.assertEqual(calls[0]["type"], "function")

    def test_a_second_call_is_kept_apart_by_its_index(self) -> None:
        calls = _assemble(
            _ToolCallChunk(0, id="a", name="HassTurnOn", arguments='{"name": "lamp"}'),
            _ToolCallChunk(1, id="b", name="GetLiveContext", arguments="{}"),
        )
        self.assertEqual(calls[0]["name"], "HassTurnOn")
        self.assertEqual(calls[1]["name"], "GetLiveContext")

    def test_indexes_may_be_announced_out_of_order(self) -> None:
        """The list-based version silently discarded the lower index."""
        calls = _assemble(
            _ToolCallChunk(1, id="b", name="GetLiveContext"),
            _ToolCallChunk(0, id="a", name="HassTurnOn"),
            _ToolCallChunk(0, arguments='{"name": "lamp"}'),
            _ToolCallChunk(1, arguments="{}"),
        )
        self.assertEqual(calls[0]["id"], "a")
        self.assertEqual(calls[0]["arguments"], '{"name": "lamp"}')
        self.assertEqual(calls[1]["id"], "b")


class _ToolInput:
    """Stand-in for ``llm.ToolInput`` when Home Assistant is not installed.

    Mirrors core's field names and defaults, including ``external`` - the flag
    that keeps a call with unreadable arguments out of ``async_call_tool``.
    """

    def __init__(
        self,
        tool_name: str,
        tool_args: object,
        id: str = "generated-id",
        external: bool = False,
    ) -> None:
        self.tool_name = tool_name
        self.tool_args = tool_args
        self.id = id
        self.external = external


class _ToolInputWithoutExternal:
    """Core before ``ToolInput.external`` existed."""

    def __init__(
        self, tool_name: str, tool_args: object, id: str = "generated-id"
    ) -> None:
        self.tool_name = tool_name
        self.tool_args = tool_args
        self.id = id


class _UsingToolInput(unittest.TestCase):
    """Point the module at a ToolInput, whichever one the test needs."""

    tool_input: type = _ToolInput

    def setUp(self) -> None:
        self._original = getattr(stream_transform.llm, "ToolInput", None)
        stream_transform.llm.ToolInput = self.tool_input

    def tearDown(self) -> None:
        if self._original is None:
            del stream_transform.llm.ToolInput
        else:
            stream_transform.llm.ToolInput = self._original


class TestFinalizeToolCalls(_UsingToolInput):
    """Turning assembled calls into what the chat log is handed."""

    def test_a_well_formed_call_becomes_a_tool_input(self) -> None:
        calls = _assemble(
            _ToolCallChunk(0, id="call_1", name="GetLiveContext"),
            _ToolCallChunk(0, arguments='{"domain": "sensor"}'),
        )
        inputs, failures = stream_transform._finalize_tool_calls(calls)
        self.assertEqual(failures, [])
        self.assertEqual(len(inputs), 1)
        self.assertEqual(inputs[0].tool_name, "GetLiveContext")
        self.assertEqual(inputs[0].tool_args, {"domain": "sensor"})
        self.assertEqual(inputs[0].id, "call_1")
        self.assertFalse(inputs[0].external)

    def test_repairable_arguments_run_the_tool(self) -> None:
        """Issue #33's arguments: the call goes through instead of vanishing."""
        calls = _assemble(
            _ToolCallChunk(0, id="call_1", name="GetLiveContext"),
            _ToolCallChunk(0, arguments='{"domain": sensor, "name": "Balkon"}'),
        )
        inputs, failures = stream_transform._finalize_tool_calls(calls)
        self.assertEqual(failures, [])
        self.assertEqual(inputs[0].tool_args, {"domain": "sensor", "name": "Balkon"})
        self.assertFalse(inputs[0].external)

    def test_calls_come_back_in_index_order(self) -> None:
        calls = _assemble(
            _ToolCallChunk(1, id="b", name="GetLiveContext", arguments="{}"),
            _ToolCallChunk(0, id="a", name="HassTurnOn", arguments="{}"),
        )
        inputs, _failures = stream_transform._finalize_tool_calls(calls)
        self.assertEqual([call.id for call in inputs], ["a", "b"])

    def test_unreadable_arguments_are_reported_not_dropped(self) -> None:
        """The model has to learn about the failure, or the turn dead-ends."""
        calls = _assemble(
            _ToolCallChunk(0, id="call_1", name="GetLiveContext"),
            _ToolCallChunk(0, arguments='{"domain": '),
        )
        inputs, failures = stream_transform._finalize_tool_calls(calls)
        self.assertEqual(len(inputs), 1)
        self.assertEqual(len(failures), 1)
        self.assertEqual(failures[0]["role"], "tool_result")
        self.assertEqual(failures[0]["tool_call_id"], inputs[0].id)
        self.assertEqual(failures[0]["tool_name"], "GetLiveContext")
        self.assertEqual(failures[0]["tool_result"]["error"], "InvalidToolArguments")

    def test_a_call_with_unreadable_arguments_is_never_executed(self) -> None:
        """external is what stops core from calling it with empty arguments."""
        calls = _assemble(
            _ToolCallChunk(0, id="call_1", name="HassTurnOn"),
            _ToolCallChunk(0, arguments='{"name": '),
        )
        inputs, _failures = stream_transform._finalize_tool_calls(calls)
        self.assertTrue(inputs[0].external)

    def test_one_broken_call_does_not_take_the_others_with_it(self) -> None:
        calls = _assemble(
            _ToolCallChunk(0, id="a", name="HassTurnOn", arguments='{"name": "lamp"}'),
            _ToolCallChunk(1, id="b", name="GetLiveContext", arguments='{"domain": '),
        )
        inputs, failures = stream_transform._finalize_tool_calls(calls)
        self.assertEqual(len(inputs), 2)
        self.assertFalse(inputs[0].external)
        self.assertTrue(inputs[1].external)
        self.assertEqual(len(failures), 1)
        self.assertEqual(failures[0]["tool_call_id"], "b")

    def test_a_call_that_never_got_a_name_is_dropped(self) -> None:
        """Nothing can be called, and there is no id worth reporting against."""
        calls = _assemble(_ToolCallChunk(0, id="call_1", arguments="{}"))
        inputs, failures = stream_transform._finalize_tool_calls(calls)
        self.assertEqual(inputs, [])
        self.assertEqual(failures, [])

    def test_a_missing_id_is_left_to_core_to_generate(self) -> None:
        calls = _assemble(_ToolCallChunk(0, name="GetLiveContext", arguments="{}"))
        inputs, _failures = stream_transform._finalize_tool_calls(calls)
        self.assertEqual(inputs[0].id, "generated-id")


class TestFinalizeWithoutExternalSupport(_UsingToolInput):
    """A core release that cannot mark a call as not-to-be-executed."""

    tool_input = _ToolInputWithoutExternal

    def test_a_broken_call_is_dropped_rather_than_run_blind(self) -> None:
        calls = _assemble(
            _ToolCallChunk(0, id="a", name="HassTurnOn", arguments='{"name": '),
        )
        inputs, failures = stream_transform._finalize_tool_calls(calls)
        self.assertEqual(inputs, [])
        self.assertEqual(failures, [])

    def test_the_healthy_calls_are_unaffected(self) -> None:
        calls = _assemble(
            _ToolCallChunk(0, id="a", name="HassTurnOn", arguments='{"name": "lamp"}'),
            _ToolCallChunk(1, id="b", name="GetLiveContext", arguments='{"domain": '),
        )
        inputs, failures = stream_transform._finalize_tool_calls(calls)
        self.assertEqual([call.id for call in inputs], ["a"])
        self.assertEqual(failures, [])


if __name__ == "__main__":
    unittest.main()
