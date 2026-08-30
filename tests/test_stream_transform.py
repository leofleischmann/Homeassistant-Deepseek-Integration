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


if __name__ == "__main__":
    unittest.main()
