"""Tests for reading a streamed tool call's ``arguments`` string.

The repair pass decides whether a house command runs or not, in both
directions: too timid and a question about the balcony temperature dies
silently, too eager and the wrong light gets switched. So the cases below come
in two halves - what must be repaired, and what must be refused.

Pure standard library, so they run without Home Assistant installed::

    python -m unittest discover -s tests
"""

from __future__ import annotations

import unittest

from integration_modules import load

tool_arguments = load("tool_arguments")

parse = tool_arguments.parse_tool_arguments
ToolArgumentsError = tool_arguments.ToolArgumentsError


class TestWellFormedArguments(unittest.TestCase):
    def test_plain_json_object(self) -> None:
        self.assertEqual(parse('{"domain": "sensor"}'), {"domain": "sensor"})

    def test_no_arguments_is_an_empty_dict(self) -> None:
        """A tool without parameters is called with "" or "{}" by endpoint."""
        self.assertEqual(parse(""), {})
        self.assertEqual(parse(None), {})
        self.assertEqual(parse("   "), {})
        self.assertEqual(parse("{}"), {})

    def test_values_keep_their_types(self) -> None:
        self.assertEqual(
            parse('{"n": 3, "x": 1.5, "ok": true, "no": false, "gone": null}'),
            {"n": 3, "x": 1.5, "ok": True, "no": False, "gone": None},
        )

    def test_nested_structures_survive(self) -> None:
        self.assertEqual(
            parse('{"a": {"b": [1, 2]}, "c": "d"}'), {"a": {"b": [1, 2]}, "c": "d"}
        )


class TestRepairedArguments(unittest.TestCase):
    def test_bareword_value(self) -> None:
        """The malformation from issue #33: an unquoted string value."""
        self.assertEqual(
            parse('{"domain": sensor, "name": "OpenWeatherMap Temperatura"}'),
            {"domain": "sensor", "name": "OpenWeatherMap Temperatura"},
        )

    def test_bareword_key(self) -> None:
        self.assertEqual(parse('{domain: "sensor"}'), {"domain": "sensor"})

    def test_single_quotes(self) -> None:
        self.assertEqual(parse("{'domain': 'sensor'}"), {"domain": "sensor"})

    def test_python_literals(self) -> None:
        self.assertEqual(
            parse('{"a": True, "b": False, "c": None}'),
            {"a": True, "b": False, "c": None},
        )

    def test_trailing_comma(self) -> None:
        self.assertEqual(parse('{"domain": "sensor",}'), {"domain": "sensor"})
        self.assertEqual(parse('{"a": [1, 2,],}'), {"a": [1, 2]})

    def test_markdown_code_fence(self) -> None:
        self.assertEqual(parse('```json\n{"domain": "sensor"}\n```'), {"domain": "sensor"})

    def test_double_encoded_json(self) -> None:
        """Some endpoints send the object as a JSON string holding the JSON."""
        self.assertEqual(parse('"{\\"domain\\": \\"sensor\\"}"'), {"domain": "sensor"})

    def test_several_malformations_at_once(self) -> None:
        self.assertEqual(
            parse("{'domain': sensor, 'live': True,}"),
            {"domain": "sensor", "live": True},
        )


class TestArgumentsLeftAlone(unittest.TestCase):
    """A repair may never invent a reading; these have to fail instead."""

    def test_a_value_inside_a_string_is_not_touched(self) -> None:
        self.assertEqual(
            parse('{"name": "Lampe True, sensor"}'), {"name": "Lampe True, sensor"}
        )

    def test_braces_inside_a_string_are_not_structural(self) -> None:
        self.assertEqual(parse('{"name": "a {b} c"}'), {"name": "a {b} c"})

    def test_truncated_object_is_refused(self) -> None:
        with self.assertRaises(ToolArgumentsError):
            parse('{"domain": ')

    def test_unterminated_string_is_refused(self) -> None:
        with self.assertRaises(ToolArgumentsError):
            parse('{"domain": "sensor')

    def test_valid_json_of_the_wrong_shape_is_refused(self) -> None:
        for text in ('["sensor"]', '"sensor"', "42"):
            with self.subTest(text=text), self.assertRaises(ToolArgumentsError):
                parse(text)

    def test_prose_is_refused(self) -> None:
        with self.assertRaises(ToolArgumentsError):
            parse("I will check the balcony sensor for you")

    def test_the_error_repeats_what_arrived(self) -> None:
        """It travels back to the model, which has to see its own mistake."""
        with self.assertRaises(ToolArgumentsError) as caught:
            parse('{"domain": ')
        self.assertIn('{"domain":', str(caught.exception))

    def test_a_long_error_is_shortened(self) -> None:
        with self.assertRaises(ToolArgumentsError) as caught:
            parse('{"a": "' + "x" * 5000)
        self.assertLess(len(str(caught.exception)), 400)


if __name__ == "__main__":
    unittest.main()
