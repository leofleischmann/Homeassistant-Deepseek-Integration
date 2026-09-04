"""Tests for rendering a tool's parameter schema whichever library core uses.

Core swapped ``voluptuous_openapi`` for ``probatio`` here, and each compares
its caller's serializer answer against its own ``UNSUPPORTED`` sentinel by
identity — so the wrong pairing does not raise, it returns the *other*
library's sentinel as if it were a schema. That object reached the API inside
the tools array and killed every request in the SDK's ``json.dumps``. Two
guards are pinned down below: any serializer answer that is not a schema is
translated into the sentinel the converter in use expects, and a rendered
result is refused unless it survives ``json.dumps`` — at every depth, because
a marker can be buried inside an otherwise ordinary schema.

The converters are injected rather than installed, so this runs on a bare
Python without either library::

    python -m unittest discover -s tests
"""

from __future__ import annotations

import unittest

from integration_modules import load

openapi_schema = load("openapi_schema")

render = openapi_schema.render_openapi_schema
SchemaConversionError = openapi_schema.SchemaConversionError
Converter = openapi_schema._Converter


class _Sentinel:
    """Stand-in for a library's UNSUPPORTED marker."""

    def __init__(self, label: str) -> None:
        self.label = label

    def __repr__(self) -> str:
        return f"UNSUPPORTED<{self.label}>"


NEW_UNSUPPORTED = _Sentinel("new")
OLD_UNSUPPORTED = _Sentinel("old")


class _UsingConverters(unittest.TestCase):
    """Install a converter list for one test."""

    def setUp(self) -> None:
        self._converters = openapi_schema.CONVERTERS

    def tearDown(self) -> None:
        openapi_schema.CONVERTERS = self._converters

    def use(self, *converters: Converter) -> None:
        openapi_schema.CONVERTERS = list(converters)


class TestSentinelTranslation(_UsingConverters):
    """The heart of the bug: two libraries, two markers, one identity check."""

    def test_a_foreign_marker_is_translated_for_the_converter_in_use(self) -> None:
        def new_converter(schema, *, custom_serializer=None):
            # Behaves like the real thing: trusts anything that is not its own
            # marker and hands it straight back as the schema.
            answer = custom_serializer(schema)
            if answer is not NEW_UNSUPPORTED:
                return answer
            return {"type": "object", "properties": {}}

        self.use(Converter("new", new_converter, NEW_UNSUPPORTED))
        # A serializer from the *other* library. Untranslated this returns
        # OLD_UNSUPPORTED as the schema, which is the reported failure.
        rendered = render("x", custom_serializer=lambda schema: OLD_UNSUPPORTED)
        self.assertEqual(rendered, {"type": "object", "properties": {}})

    def test_a_marker_no_list_could_know_about_is_translated_too(self) -> None:
        """The rule is "not a schema means defer", so a third library is covered."""

        def new_converter(schema, *, custom_serializer=None):
            answer = custom_serializer(schema)
            if answer is not NEW_UNSUPPORTED:
                return answer
            return {"type": "object", "properties": {}}

        self.use(Converter("new", new_converter, NEW_UNSUPPORTED))
        rendered = render("x", custom_serializer=lambda schema: _Sentinel("third"))
        self.assertEqual(rendered, {"type": "object", "properties": {}})

    def test_a_serializer_that_answers_none_defers_instead_of_leaking(self) -> None:
        """Core's MergedAPI hands back the raw node when its serializers all pass."""

        def new_converter(schema, *, custom_serializer=None):
            answer = custom_serializer(schema)
            if answer is not NEW_UNSUPPORTED:
                return answer
            return {"type": "string"}

        self.use(Converter("new", new_converter, NEW_UNSUPPORTED))
        self.assertEqual(
            render("x", custom_serializer=lambda schema: None), {"type": "string"}
        )

    def test_a_real_answer_from_the_serializer_is_passed_through(self) -> None:
        def converter(schema, *, custom_serializer=None):
            answer = custom_serializer(schema)
            return answer if answer is not NEW_UNSUPPORTED else {"type": "object"}

        self.use(Converter("new", converter, NEW_UNSUPPORTED))
        rendered = render("x", custom_serializer=lambda schema: {"type": "string"})
        self.assertEqual(rendered, {"type": "string"})

    def test_no_serializer_stays_no_serializer(self) -> None:
        """Wrapping ``None`` would turn "no serializer" into one that returns None."""
        seen: list[object] = []

        def converter(schema, *, custom_serializer=None):
            seen.append(custom_serializer)
            return {"type": "object"}

        self.use(Converter("only", converter, NEW_UNSUPPORTED))
        render("x")
        self.assertEqual(seen, [None])


class TestRenderOpenapiSchema(_UsingConverters):
    def test_the_first_converter_that_returns_a_schema_wins(self) -> None:
        def good(schema, *, custom_serializer=None):
            return {"type": "object", "properties": {}}

        def never_called(schema, *, custom_serializer=None):
            raise AssertionError("the first converter already answered")

        self.use(
            Converter("good", good, NEW_UNSUPPORTED),
            Converter("second", never_called, OLD_UNSUPPORTED),
        )
        self.assertEqual(render("x"), {"type": "object", "properties": {}})

    def test_a_sentinel_is_never_accepted_as_a_schema(self) -> None:
        """The last line of defence, for a marker no translation knew about."""

        def leaky(schema, *, custom_serializer=None):
            return _Sentinel("third-party")

        self.use(Converter("leaky", leaky, NEW_UNSUPPORTED))
        with self.assertRaises(SchemaConversionError):
            render("x")

    def test_a_marker_buried_inside_the_schema_is_caught_as_well(self) -> None:
        """What checking only the top level missed: one unrenderable property.

        A converter asks the serializer per node and returns its answer where
        the node was, so the object arrives as a property value while the
        schema around it looks perfectly ordinary.
        """

        def leaky(schema, *, custom_serializer=None):
            return {
                "type": "object",
                "properties": {"name": {"type": "string"}, "area": _Sentinel("buried")},
            }

        self.use(Converter("leaky", leaky, NEW_UNSUPPORTED))
        with self.assertRaises(SchemaConversionError) as caught:
            render("x")
        self.assertIn("cannot be sent as JSON", str(caught.exception))

    def test_a_converter_that_buries_one_gives_way_to_one_that_does_not(self) -> None:
        def leaky(schema, *, custom_serializer=None):
            return {"properties": {"area": _Sentinel("buried")}}

        def clean(schema, *, custom_serializer=None):
            return {"properties": {"area": {"type": "string"}}}

        self.use(
            Converter("leaky", leaky, NEW_UNSUPPORTED),
            Converter("clean", clean, OLD_UNSUPPORTED),
        )
        self.assertEqual(render("x"), {"properties": {"area": {"type": "string"}}})

    def test_the_next_converter_is_tried_when_one_cannot_read_the_schema(self) -> None:
        """A core whose schema objects the newer library does not understand."""

        def wrong_library(schema, *, custom_serializer=None):
            raise TypeError("unhashable type: 'Schema'")

        def right_library(schema, *, custom_serializer=None):
            return {"type": "object"}

        self.use(
            Converter("wrong", wrong_library, NEW_UNSUPPORTED),
            Converter("right", right_library, OLD_UNSUPPORTED),
        )
        self.assertEqual(render("x"), {"type": "object"})

    def test_every_converter_failing_names_all_of_them(self) -> None:
        """The message has to say what was tried, or the cause stays invisible."""

        def broken(schema, *, custom_serializer=None):
            raise ValueError("cannot read this")

        def leaky(schema, *, custom_serializer=None):
            return _Sentinel("third-party")

        self.use(
            Converter("broken", broken, NEW_UNSUPPORTED),
            Converter("leaky", leaky, OLD_UNSUPPORTED),
        )
        with self.assertRaises(SchemaConversionError) as caught:
            render("x")
        message = str(caught.exception)
        self.assertIn("broken", message)
        self.assertIn("leaky", message)

    def test_no_converter_installed_is_reported_plainly(self) -> None:
        self.use()
        with self.assertRaises(SchemaConversionError) as caught:
            render("x")
        self.assertIn("neither probatio nor voluptuous_openapi", str(caught.exception))

    def test_an_empty_schema_is_a_valid_result(self) -> None:
        """A tool without parameters converts to {}, which must not read as failure."""

        def converter(schema, *, custom_serializer=None):
            return {}

        self.use(Converter("converter", converter, NEW_UNSUPPORTED))
        self.assertEqual(render("x"), {})


class TestAvailableConverters(unittest.TestCase):
    def test_core_s_current_library_is_preferred(self) -> None:
        """probatio first: on a modern core it is the one that speaks its schemas."""
        names = [converter.name for converter in openapi_schema._available_converters()]
        self.assertEqual(names, sorted(names, key=lambda n: n != "probatio.to_openapi"))

    def test_discovery_never_raises_when_nothing_is_installed(self) -> None:
        self.assertIsInstance(openapi_schema._available_converters(), list)

    def test_each_converter_carries_its_own_sentinel(self) -> None:
        for converter in openapi_schema._available_converters():
            with self.subTest(converter=converter.name):
                self.assertIsNotNone(converter.unsupported)


if __name__ == "__main__":
    unittest.main()
