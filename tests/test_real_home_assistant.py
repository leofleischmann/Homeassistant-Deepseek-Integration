"""Tests that need a real Home Assistant, and skip everywhere else.

The rest of the suite runs on a bare Python against stubs, which keeps CI fast
and also blind to a change in core itself - a swapped schema library once broke
every request, and no stubbed test could have seen it, because the stubs are
ours and the change was core's.

These cases close that gap by asking the real thing, whichever version is
installed: what they assert about core is discovered from core, so the same
suite is meaningful on the release before a change and the one after it.

They skip unless Home Assistant is importable, so CI stays green; build the
environment with::

    python scripts/ha_testenv.py                # latest
    python scripts/ha_testenv.py 2026.9.0       # or any release
    .venv-ha/Scripts/python -m unittest discover -s tests

``hatest.bat`` does both on Windows.
"""

from __future__ import annotations

import asyncio
import json
import pathlib
from types import MappingProxyType, SimpleNamespace
import unittest
from unittest import mock

from integration_modules import load

# Home Assistant first: on cores that ship a voluptuous replacement, importing
# it aliases ``voluptuous`` exactly as at runtime. Anything importing voluptuous
# before this would bind the wrong module, and core warns about it.
#
# The import is also the availability check, but only after ruling out the
# harness's own stub - by the time this module is imported another test has
# usually put one in ``sys.modules``, and failing against that would report a
# confusing missing-attribute error instead of "core is not installed".
_SETUP_HINT = "Home Assistant is not installed; build it with scripts/ha_testenv.py"
try:
    import importlib.util

    if importlib.util.find_spec("homeassistant.helpers.llm") is None:
        raise ImportError(_SETUP_HINT)

    from homeassistant.const import __version__ as HA_VERSION

    import voluptuous as vol
    from homeassistant.exceptions import HomeAssistantError
    from homeassistant.helpers import config_validation as cv, frame, llm, selector

    HA_REASON = ""
except (ImportError, ValueError) as err:  # pragma: no cover - the normal case in CI
    HA_VERSION = ""
    HA_REASON = _SETUP_HINT if str(err) == _SETUP_HINT else f"{_SETUP_HINT} ({err})"

requires_ha = unittest.skipIf(bool(HA_REASON), HA_REASON)


def _core_converter() -> str:
    """The schema library core's own ``llm`` helper imported, by module name.

    Core has changed this once and may again, so nothing here names a library:
    the helper's converter says which one this release speaks, and the tests
    assert against that. ``""`` when the helper exposes neither, which is only
    a reason to skip.
    """
    for attribute in ("to_openapi", "convert"):
        if (converter := getattr(llm, attribute, None)) is not None:
            return getattr(converter, "__module__", "").split(".")[0]
    return ""


#: Core's ``llm.Tool`` grew an ``integration`` attribute naming the integration
#: that provides the tool, and reports every tool that leaves it unset. Older
#: cores have no such notion, so the cases about it skip there instead of
#: asserting against a core that never asks.
def _tool_has_integration() -> bool:
    if HA_REASON:
        return False
    return hasattr(llm.Tool, "integration")


requires_tool_integration = unittest.skipUnless(
    _tool_has_integration(), "this core's llm.Tool has no 'integration' attribute"
)


#: Core 2026.10 wants a tool to answer with ``llm.ToolResult`` and deprecated
#: the bare JSON object it took before. Both shapes are built and read through
#: these, so the suite says what this core says rather than picking a side.
def _tool_result(data):
    """A tool's answer in the shape this core expects."""
    if (result_type := getattr(llm, "ToolResult", None)) is None:
        return data
    return result_type(data=data)


def _result_data(result):
    """The payload inside a tool's answer, whichever shape it came in."""
    return getattr(result, "data", result)


requires_tool_result = unittest.skipUnless(
    not HA_REASON and hasattr(llm, "ToolResult"),
    "this core has no llm.ToolResult",
)


def _delta_data(delta):
    """The payload of a tool result delta, under whichever key this core reads."""
    if "result" in delta:
        return delta["result"].data
    return delta["tool_result"]


#: The same release moved a chat log's own tool results onto ``ToolResult``:
#: ``ToolResultContent.result`` holds one, and ``tool_result`` became a property
#: that reports every access. Asked of the class, so the cases follow core.
def _chat_log_has_typed_results() -> bool:
    if HA_REASON:
        return False
    from homeassistant.components import conversation

    return "result" in getattr(
        conversation.ToolResultContent, "__dataclass_fields__", {}
    )


requires_typed_chat_log = unittest.skipUnless(
    _chat_log_has_typed_results(),
    "this core's chat log holds tool results as bare objects",
)


class _FakeHass:
    """Just enough Home Assistant for the LLM API registry, which is a dict."""

    def __init__(self) -> None:
        self.data: dict = {}


def _minimal_hass():
    """A real core object, for paths that render templates or use the bus."""
    import tempfile

    from homeassistant.core import HomeAssistant

    return HomeAssistant(tempfile.mkdtemp())


#: The id the stand-in API below is registered under. Deliberately not
#: "assist": older cores - 2026.1 for one - keep their own Assist API under
#: that id, so registering a second one there fails before a single case has run.
_SELECTED_API = "recorded_api"

if not HA_REASON:

    class _RecordedAPI(llm.API):
        """A registered API standing in for Assist, with one tool of its own."""

        async def async_get_api_instance(self, llm_context):
            class _Tool(llm.Tool):
                name = "recorded"
                description = "a tool the selected API brought along"
                parameters = vol.Schema({vol.Optional("name"): cv.string})

                async def async_call(self, hass, tool_input, llm_context):
                    return _tool_result({"called": "recorded"})

            return llm.APIInstance(
                api=self,
                api_prompt="Recorded API prompt",
                llm_context=llm_context,
                tools=[_Tool()],
                custom_serializer=llm.selector_serializer,
            )


def _schemas() -> dict[str, object]:
    """Parameter schemas in the shapes Home Assistant's Assist tools use."""
    return {
        "empty": vol.Schema({}),
        "plain_strings": vol.Schema(
            {
                vol.Optional("name"): cv.string,
                vol.Optional("area"): cv.string,
                vol.Optional("domain"): vol.All(cv.ensure_list, [cv.string]),
            }
        ),
        "selectors": vol.Schema(
            {
                vol.Optional("name"): selector.TextSelector(),
                vol.Optional("area"): selector.AreaSelector(),
                vol.Optional("brightness"): selector.NumberSelector(
                    selector.NumberSelectorConfig(min=0, max=100)
                ),
                vol.Optional("color"): selector.ColorRGBSelector(),
            }
        ),
        "nested": vol.Schema(
            {
                vol.Required("query"): cv.string,
                vol.Optional("live"): cv.boolean,
                vol.Optional("nested"): vol.Schema({vol.Optional("depth"): int}),
            }
        ),
    }


@requires_ha
class TestSchemaRendering(unittest.TestCase):
    """Every tool schema must come out as something the SDK can serialise."""

    def setUp(self) -> None:
        self.openapi_schema = load("openapi_schema")

    def test_the_library_core_uses_is_one_we_can_speak(self) -> None:
        """Whichever core ships, a converter for it has to be installed."""
        libraries = {c.name.split(".")[0] for c in self.openapi_schema.CONVERTERS}
        self.assertTrue(libraries, f"no schema converter on Home Assistant {HA_VERSION}")
        if not (core_library := _core_converter()):
            self.skipTest("core's llm helper exposes no converter to compare against")
        self.assertIn(core_library, libraries, f"on Home Assistant {HA_VERSION}")

    def test_every_schema_shape_renders(self) -> None:
        for label, schema in _schemas().items():
            with self.subTest(schema=label):
                rendered = self.openapi_schema.render_openapi_schema(
                    schema, custom_serializer=llm.selector_serializer
                )
                self.assertIsInstance(rendered, dict)
                json.dumps(rendered)  # must not raise

    def test_selectors_become_real_types(self) -> None:
        """The serializer is what turns a selector into a JSON type at all."""
        rendered = self.openapi_schema.render_openapi_schema(
            _schemas()["selectors"], custom_serializer=llm.selector_serializer
        )
        self.assertEqual(rendered["properties"]["name"], {"type": "string"})

    def test_a_schema_it_cannot_render_raises_rather_than_leaking(self) -> None:
        """Without the serializer that knows selectors, refusing is the safe answer."""
        with self.assertRaises(self.openapi_schema.SchemaConversionError):
            self.openapi_schema.render_openapi_schema(_schemas()["selectors"])

    def test_core_s_serializer_is_what_types_a_bare_validator(self) -> None:
        """The everyday path has to put a real type on every leaf.

        A converter may leave a validator it does not recognise untyped, and
        the installed ones disagree about that: ``cv.string`` is a plain
        function, which probatio renders as ``{}`` while voluptuous-openapi
        guesses from the return annotation - and guesses ``cv.boolean`` wrong.
        Core's serializer knows Home Assistant's own validators and is the one
        actually in use for Assist tools and AI Task structures, so what the
        model is told about a parameter does not ride on which library core
        ships. That is the property worth holding, rather than any converter's
        guess, and it is why the untyped rendering is left alone elsewhere.
        """
        rendered = self.openapi_schema.render_openapi_schema(
            _schemas()["plain_strings"], custom_serializer=llm.selector_serializer
        )
        self.assertEqual(rendered["properties"]["name"], {"type": "string"})
        self.assertEqual(
            rendered["properties"]["domain"]["items"], {"type": "string"}
        )


@requires_ha
class TestMixedSchemaLibraries(unittest.TestCase):
    """Two converters installed, each deaf to the other's "I cannot" marker.

    This is what broke every request once core changed library: neither
    converter raises on a foreign marker, it hands the marker back as if it
    were the schema, and the request dies in the SDK. Nothing below names a
    library - the pairs come from whatever is installed, so the same cases
    cover the next swap as well as the last one.
    """

    def setUp(self) -> None:
        self.openapi_schema = load("openapi_schema")
        self.converters = list(self.openapi_schema.CONVERTERS)
        if len(self.converters) < 2:
            self.skipTest(
                "only one schema library installed, so there is no mismatch to make"
            )

    def _pairs(self):
        """Every (converter, someone else's marker) combination."""
        for converter in self.converters:
            for other in self.converters:
                if other is not converter:
                    yield converter, other

    def test_the_libraries_disagree_on_their_marker(self) -> None:
        """The premise: identity comparison is why a foreign one gets through."""
        for converter, other in self._pairs():
            with self.subTest(converter=converter.name, other=other.name):
                self.assertIsNot(converter.unsupported, other.unsupported)

    def test_a_converter_alone_hands_a_foreign_marker_straight_back(self) -> None:
        """Not an error: a plausible-looking return value the SDK cannot send."""
        for converter, other in self._pairs():
            with self.subTest(converter=converter.name, marker=other.name):
                try:
                    leaked = converter.render(
                        _schemas()["plain_strings"],
                        custom_serializer=lambda _node: other.unsupported,
                    )
                except Exception:
                    continue  # refusing outright is the safe failure, not the bug
                with self.assertRaises(TypeError):
                    json.dumps({"tools": [{"function": {"parameters": leaked}}]})

    def test_the_shim_renders_it_whichever_dialect_it_arrives_in(self) -> None:
        """A third-party ``llm.API`` can carry a serializer from either library."""
        for converter in self.converters:
            with self.subTest(marker=converter.name):
                rendered = self.openapi_schema.render_openapi_schema(
                    _schemas()["plain_strings"],
                    custom_serializer=lambda _node: converter.unsupported,
                )
                self.assertEqual(
                    sorted(rendered["properties"]), ["area", "domain", "name"]
                )
                json.dumps(rendered)

    def test_a_foreign_marker_on_one_leaf_is_handled_too(self) -> None:
        """A merged serializer answers per node, so dialects can be mixed.

        Core's ``MergedAPI`` asks each API's serializer in turn and returns the
        first answer, so one leaf can come back in the wrong dialect while
        everything around it is fine.

        What the shim promises is that the dialect stops mattering: a mixture
        renders exactly as a serializer that defers on every node in the right
        dialect would. Asserting that equality rather than a literal type is
        deliberate - what a converter makes of a node it has to render itself
        is the converter's business and differs between libraries, so a literal
        here would pin this test to whichever one core happened to ship. The
        original bug fails it either way: a leaked marker is not what deferring
        produces, and would not survive ``json.dumps``.
        """
        mine, theirs = self.converters[0], self.converters[1]

        def mixed_dialect(node):
            return theirs.unsupported if node is cv.string else mine.unsupported

        rendered = self.openapi_schema.render_openapi_schema(
            _schemas()["plain_strings"], custom_serializer=mixed_dialect
        )
        deferred = self.openapi_schema.render_openapi_schema(
            _schemas()["plain_strings"], custom_serializer=lambda _node: None
        )
        self.assertEqual(rendered, deferred)
        self.assertEqual(sorted(rendered["properties"]), ["area", "domain", "name"])
        json.dumps(rendered)

    def test_core_s_own_serializer_renders_through_the_shim(self) -> None:
        """The everyday path, with whatever dialect this core happens to speak."""
        rendered = self.openapi_schema.render_openapi_schema(
            _schemas()["plain_strings"], custom_serializer=llm.selector_serializer
        )
        self.assertIsInstance(rendered, dict)
        json.dumps(rendered)


def _load_chat_session():
    """chat_session, or a skip: it imports the whole ai_task component chain."""
    try:
        return load("chat_session", api_stubs=True)
    except ImportError as err:  # pragma: no cover
        raise unittest.SkipTest(f"chat_session needs more of core installed: {err}")


@requires_ha
class TestToolFormatting(unittest.TestCase):
    """The integration's real formatting path, on real tools."""

    def setUp(self) -> None:
        self.chat_session = _load_chat_session()

        class _Tool(llm.Tool):
            def __init__(self, name: str, parameters: object) -> None:
                self.name = name
                self.description = f"Tool {name}"
                self.parameters = parameters

            async def async_call(self, hass, tool_input, llm_context):
                return {}

        self.tools = [_Tool(name, schema) for name, schema in _schemas().items()]

    def test_no_tool_is_skipped_and_the_block_serialises(self) -> None:
        formatted, skipped = self.chat_session._format_tools_for_api(
            self.tools, llm.selector_serializer
        )
        self.assertEqual(skipped, [])
        self.assertEqual(len(formatted), len(self.tools))
        json.dumps({"tools": formatted})  # must not raise

    def test_a_tool_that_cannot_be_rendered_is_skipped_by_name(self) -> None:
        class _Broken(llm.Tool):
            name = "broken"
            description = "unrenderable"

            @property
            def parameters(self):
                raise RuntimeError("schema is broken")

            async def async_call(self, hass, tool_input, llm_context):
                return {}

        formatted, skipped = self.chat_session._format_tools_for_api(
            [*self.tools, _Broken()], llm.selector_serializer
        )
        self.assertEqual(skipped, ["broken"])
        self.assertEqual(len(formatted), len(self.tools))

    def test_core_s_own_intent_tools_render(self) -> None:
        """``IntentTool`` builds its schema from a handler's real slot schema."""
        from homeassistant.helpers import intent

        tools = [
            llm.IntentTool(
                "HassTurnOn",
                intent.ServiceIntentHandler(
                    "HassTurnOn", "homeassistant", "turn_on", description="Turns on"
                ),
            ),
            llm.IntentTool(
                "HassLightSet",
                intent.ServiceIntentHandler(
                    "HassLightSet",
                    "light",
                    "turn_on",
                    required_domains={"light"},
                    optional_slots={
                        ("brightness", "brightness_pct"): selector.NumberSelector(
                            selector.NumberSelectorConfig(min=0, max=100)
                        ),
                        ("color", "rgb_color"): selector.ColorRGBSelector(),
                    },
                    description="Sets brightness or color",
                ),
            ),
        ]
        formatted, skipped = self.chat_session._format_tools_for_api(
            tools, llm.selector_serializer
        )
        self.assertEqual(skipped, [])
        for entry in formatted:
            self.assertIsInstance(entry["function"]["parameters"], dict)
        json.dumps({"tools": formatted})  # must not raise


@requires_ha
class TestStructuredOutput(unittest.TestCase):
    """AI Task structures go through the same converter."""

    def test_a_structure_renders_to_a_schema(self) -> None:
        structured_output = load("structured_output", api_stubs=True)
        schema = structured_output.structure_to_openapi_schema(
            vol.Schema({vol.Required("title"): cv.string, vol.Optional("count"): int}),
            custom_serializer=llm.selector_serializer,
        )
        self.assertIsInstance(schema, dict)
        self.assertEqual(sorted(schema["properties"]), ["count", "title"])


@requires_ha
class TestToolCallsAgainstCoreToolInput(unittest.TestCase):
    """Tool-call assembly and repair, against core's actual ``ToolInput``."""

    def setUp(self) -> None:
        self.stream_transform = load("stream_transform", api_stubs=True)

    def _assemble(self, *chunks):
        calls: dict[int, dict] = {}
        for chunk in chunks:
            self.stream_transform._merge_tool_call_chunk(calls, chunk)
        return calls

    @staticmethod
    def _chunk(index, *, id=None, name=None, arguments=None):
        function = None
        if name is not None or arguments is not None:
            function = type("F", (), {"name": name, "arguments": arguments})()
        return type(
            "C", (), {"index": index, "id": id, "type": None, "function": function}
        )()

    def test_repairable_arguments_produce_a_callable_tool_input(self) -> None:
        inputs, failures = self.stream_transform._finalize_tool_calls(
            self._assemble(
                self._chunk(0, id="call_1", name="GetLiveContext"),
                self._chunk(0, arguments='{"domain": sensor, "name": "Balkon"}'),
            )
        )
        self.assertEqual(failures, [])
        self.assertEqual(len(inputs), 1)
        self.assertIsInstance(inputs[0], llm.ToolInput)
        self.assertEqual(inputs[0].tool_args, {"domain": "sensor", "name": "Balkon"})
        self.assertFalse(inputs[0].external)

    def test_unreadable_arguments_are_reported_and_not_executed(self) -> None:
        """``external`` is what reports the parse error back to the model."""
        inputs, failures = self.stream_transform._finalize_tool_calls(
            self._assemble(
                self._chunk(0, id="call_1", name="HassTurnOn"),
                self._chunk(0, arguments='{"name": '),
            )
        )
        self.assertTrue(inputs[0].external)
        self.assertEqual(len(failures), 1)
        self.assertEqual(_delta_data(failures[0])["error"], "InvalidToolArguments")

    def test_unreadable_arguments_are_never_executed(self) -> None:
        """True on every core: a half-readable call is not guessed at."""
        inputs, _failures = self.stream_transform._finalize_tool_calls(
            self._assemble(
                self._chunk(0, id="call_1", name="HassTurnOn"),
                self._chunk(0, arguments='{"name": '),
            )
        )
        self.assertFalse(any(getattr(call, "tool_args", None) for call in inputs))


@requires_ha
class TestAgentToolsAPI(unittest.TestCase):
    """Web search reaches this integration's agents and nothing else (#38).

    It used to be registered with ``llm.async_register_api``, and that registry
    is what every conversation integration builds its API picker from - so the
    Brave tool turned up in Anthropic's, Gemini's and OpenAI's agent settings,
    and worked there. These cases hold the tool inside an ``llm.API`` object
    that is only ever handed to our own chat log.
    """

    ENTRY_ID = "01JDEEPSEEK"

    def _brave(self) -> dict:
        """An agent's settings with Brave chosen as its web search provider."""
        return {self.const.CONF_WEB_SEARCH: self.const.WEB_SEARCH_PROVIDER_BRAVE}

    def setUp(self) -> None:
        self.agent_tools = load("agent_tools")
        self.const = load("const")
        self.hass = _FakeHass()
        self.context = llm.LLMContext(
            platform="deepseek_conversation",
            context=None,
            language="en",
            assistant="conversation",
            device_id=None,
        )
        self.registered = _RecordedAPI(
            hass=self.hass, id=_SELECTED_API, name="Recorded"
        )
        llm.async_register_api(self.hass, self.registered)

    def _entry(self, *, brave: str | None = "brave-key"):
        data = {self.const.CONF_BRAVE_API_KEY: brave} if brave else {}
        return SimpleNamespace(
            entry_id=self.ENTRY_ID,
            title="DeepSeek",
            data=data,
            runtime_data=_runtime_data(),
        )

    async def _tools(self, options, selected):
        api = self.agent_tools.agent_llm_api(
            self.hass, self._entry(), options, selected
        )
        instance = await api.async_get_api_instance(self.context)
        return instance, [tool.name for tool in instance.tools]

    def test_the_tool_is_never_put_in_the_shared_registry(self) -> None:
        """The bug itself: what is registered is what other integrations offer."""
        registered = [api.id for api in llm.async_get_apis(self.hass)]
        self.assertIn(_SELECTED_API, registered)
        self.assertEqual(
            [api_id for api_id in registered if api_id.startswith(self.const.DOMAIN)],
            [],
        )

    def test_an_agent_without_web_search_uses_home_assistant_s_own_path(self) -> None:
        """Passing the selection through unchanged keeps the common case core's."""
        selection = self.agent_tools.agent_llm_api(
            self.hass, self._entry(), {}, [_SELECTED_API]
        )
        self.assertEqual(selection, [_SELECTED_API])

    def test_web_search_is_added_beside_the_selected_tools(self) -> None:
        instance, names = asyncio.run(
            self._tools(self._brave(), [_SELECTED_API])
        )
        self.assertEqual(names, ["recorded", "web_search"])
        self.assertIn("Recorded API prompt", instance.api_prompt)
        self.assertIn("web_search", instance.api_prompt)
        self.assertIs(instance.custom_serializer, llm.selector_serializer)

    def test_an_agent_with_no_home_assistant_api_gets_web_search_alone(self) -> None:
        _instance, names = asyncio.run(
            self._tools(self._brave(), None)
        )
        self.assertEqual(names, ["web_search"])

    @requires_tool_integration
    def test_every_tool_this_integration_adds_names_it(self) -> None:
        """Whatever we put in the instance is ours and has to say so (#44).

        Asked of the instance rather than of the class so that a second tool
        added later is covered without anyone remembering to extend this; no
        Home Assistant API is selected, so every tool here is one of ours.
        """
        instance, _names = asyncio.run(
            self._tools(self._brave(), None)
        )
        self.assertEqual(
            {tool.name: tool.integration for tool in instance.tools},
            {"web_search": self.const.DOMAIN},
        )

    def test_a_selection_that_no_longer_exists_fails_as_core_would(self) -> None:
        """Switching web search on must not change how a broken selection behaves.

        An integration removed since the agent was configured leaves its id in
        the subentry; Home Assistant refuses that, and so do we, rather than
        quietly answering without control over the home.
        """
        with self.assertRaises(HomeAssistantError):
            asyncio.run(
                self._tools(self._brave(), [_SELECTED_API, "gone"])
            )

    def test_deepseek_search_needs_no_key_of_its_own(self) -> None:
        """On the official API the entry's own key is all there is to it."""
        entry = SimpleNamespace(
            entry_id=self.ENTRY_ID,
            title="DeepSeek",
            data={"api_key": "sk-deepseek"},
            runtime_data=_runtime_data(),
        )
        api = self.agent_tools.agent_llm_api(
            self.hass,
            entry,
            {self.const.CONF_WEB_SEARCH: self.const.WEB_SEARCH_PROVIDER_DEEPSEEK},
            [_SELECTED_API],
        )
        instance = asyncio.run(api.async_get_api_instance(self.context))
        self.assertEqual(
            [tool.name for tool in instance.tools], ["recorded", "web_search"]
        )
        # The provider's own query rules reach the model through the tool.
        self.assertIn("128", instance.tools[-1].description)

    def test_a_search_is_counted_on_the_entry_s_usage_tracker(self) -> None:
        """Usage is per API key, so every agent of the entry counts in one place."""
        entry = self._entry()
        api = self.agent_tools.agent_llm_api(
            self.hass, entry, self._brave(), [_SELECTED_API]
        )
        instance = asyncio.run(api.async_get_api_instance(self.context))
        web_search = instance.tools[-1]
        # Called directly: what a provider answers is covered where the tool
        # is, and this case is only about where the count ends up.
        web_search._on_results("brave")
        self.assertEqual(entry.runtime_data.usage.searches, ["brave"])

    def test_deepseek_search_is_not_offered_behind_a_custom_base_url(self) -> None:
        entry = SimpleNamespace(
            entry_id=self.ENTRY_ID,
            title="Gateway",
            data={"api_key": "sk-other", "base_url": "https://gateway.example/v1"},
        )
        api = self.agent_tools.agent_llm_api(
            self.hass,
            entry,
            {self.const.CONF_WEB_SEARCH: self.const.WEB_SEARCH_PROVIDER_DEEPSEEK},
            [_SELECTED_API],
        )
        self.assertEqual(api, [_SELECTED_API])

    def test_the_option_without_a_key_answers_without_the_tool(self) -> None:
        """The key can be removed through Reconfigure while the choice stays."""
        api = self.agent_tools.agent_llm_api(
            self.hass,
            self._entry(brave=None),
            self._brave(),
            [_SELECTED_API],
        )
        # No key means no web search at all, so the selection is passed through.
        self.assertEqual(api, [_SELECTED_API])

    def test_a_key_removed_after_the_agent_was_built_is_survivable(self) -> None:
        """``agent_llm_api`` looks once; the instance is built per turn."""
        api = self.agent_tools.AgentToolsAPI(
            hass=self.hass,
            entry=self._entry(brave=None),
            options=self._brave(),
            selected=[_SELECTED_API],
        )
        instance = asyncio.run(api.async_get_api_instance(self.context))
        self.assertEqual([tool.name for tool in instance.tools], ["recorded"])

    def test_the_tools_render_for_the_api_like_any_other(self) -> None:
        chat_session = _load_chat_session()
        instance, _names = asyncio.run(
            self._tools(self._brave(), [_SELECTED_API])
        )
        formatted, skipped = chat_session._format_tools_for_api(
            list(instance.tools), instance.custom_serializer
        )
        self.assertEqual(skipped, [])
        json.dumps({"tools": formatted})
        web_search = next(
            entry for entry in formatted if entry["function"]["name"] == "web_search"
        )
        self.assertEqual(
            sorted(web_search["function"]["parameters"]["properties"]),
            ["count", "query"],
        )


@requires_ha
class TestToolsNameTheirIntegration(unittest.TestCase):
    """Every tool of ours says which integration provides it (#44, #45).

    Home Assistant 2026.10 added ``llm.Tool.integration`` and logs a warning
    for each tool that leaves it at the class default; from the release in
    ``llm.TOOL_INTEGRATION_BREAKS_IN_HA_VERSION`` such a tool is refused.
    Nothing was broken when it was reported - the warning is about the release
    after next - which is exactly why it is worth a test: core tags the tool
    itself as a stopgap, so once that stopgap goes, only what the tool says
    about itself is left.

    The default is read off core rather than written down here, so a release
    that changes it is a failure and not a quietly passing test.
    """

    def setUp(self) -> None:
        self.search_tool = load("web.search_tool")
        self.const = load("const")

    @requires_tool_integration
    def test_web_search_does_not_leave_the_default_in_place(self) -> None:
        """Core's own test is ``tool.integration is None``; ours must pass it."""
        self.assertIsNone(llm.Tool.integration)
        self.assertEqual(
            self.search_tool.WebSearchTool(
                _StubProvider(), lambda _provider: None
            ).integration,
            self.const.DOMAIN,
        )

    @requires_tool_integration
    def test_the_domain_is_the_one_core_derives_from_the_module_path(self) -> None:
        """Core reads it from ``custom_components/<domain>``; so must we.

        A tag that disagreed with the directory would silence the warning and
        still attribute the tool to an integration that does not exist. The
        tool lives in a subpackage, and core only ever looks at the segment
        right after ``custom_components`` - so that is the one compared.
        """
        parts = pathlib.Path(self.search_tool.__file__).parts
        self.assertEqual(
            self.const.DOMAIN, parts[parts.index("custom_components") + 1]
        )


@requires_ha
class TestToolResultsTravelInTheCoreShape(unittest.TestCase):
    """A chat log's tool results are read and written the way this core asks.

    Core 2026.10 moved them onto ``llm.ToolResult`` and reports the two old
    spellings this integration used: reading ``ToolResultContent.tool_result``
    on the way back to the API - every tool result goes through that - and
    setting ``tool_result`` on the delta that answers a call with unreadable
    arguments. Both still work until 2027.11, which is why nothing failed and
    why it is pinned here: the report is the only symptom there will be.
    """

    def setUp(self) -> None:
        from homeassistant.components import conversation

        self.conversation = conversation
        self.llm_compat = load("llm_compat")

    def _content(self, data: dict):
        """A tool result as this core's chat log would hold it."""
        fields = {
            "agent_id": "agent",
            "tool_call_id": "call_1",
            "tool_name": "GetLiveContext",
        }
        if _chat_log_has_typed_results():
            return self.conversation.ToolResultContent(
                **fields, result=llm.ToolResult(data=data)
            )
        return self.conversation.ToolResultContent(**fields, tool_result=data)

    def test_the_payload_is_read_on_every_core(self) -> None:
        self.assertEqual(
            self.llm_compat.tool_result_data(self._content({"ok": 1})), {"ok": 1}
        )

    @requires_typed_chat_log
    def test_reading_does_not_use_the_accessor_core_reports(self) -> None:
        with mock.patch.object(frame, "report_usage") as reported:
            self.llm_compat.tool_result_data(self._content({"ok": 1}))
        reported.assert_not_called()

    def test_a_result_we_write_lands_in_the_real_chat_log(self) -> None:
        """The delta is only right if core's own stream reader takes it."""

        async def check() -> None:
            hass = _minimal_hass()
            try:
                chat_log = self.conversation.ChatLog(hass, "conversation-1")
                call = llm.ToolInput(
                    tool_name="HassTurnOn", tool_args={}, id="call_1", external=True
                )
                delta = self.llm_compat.tool_result_delta(
                    tool_call_id="call_1",
                    tool_name="HassTurnOn",
                    data={"error": "InvalidToolArguments"},
                    error=True,
                )

                async def stream():
                    yield {"role": "assistant", "tool_calls": [call]}
                    yield delta

                with mock.patch.object(frame, "report_usage") as reported:
                    added = [
                        content
                        async for content in chat_log.async_add_delta_content_stream(
                            "agent", stream()
                        )
                    ]
                reported.assert_not_called()
                self.assertEqual(added[-1].role, "tool_result")
                self.assertEqual(added[-1].tool_call_id, "call_1")
                self.assertEqual(
                    self.llm_compat.tool_result_data(added[-1]),
                    {"error": "InvalidToolArguments"},
                )
            finally:
                await hass.async_stop()

        asyncio.run(check())

    @requires_tool_result
    def test_an_error_we_write_is_marked_as_one(self) -> None:
        delta = self.llm_compat.tool_result_delta(
            tool_call_id="call_1", tool_name="HassTurnOn", data={}, error=True
        )
        self.assertTrue(delta["result"].error)
        self.assertNotIn("tool_result", delta)


@requires_ha
class TestReasoningIsReplayed(unittest.TestCase):
    """What a chat log's reasoning turns into on the way back to the API.

    With ``tools`` in the request, DeepSeek wants the reasoning of *every*
    earlier assistant turn back - not only of those that called a tool - and
    answers 400 without it. Without tools it ignores it, so sending it always
    while thinking is on is the one rule that is right for both.
    """

    def setUp(self) -> None:
        from homeassistant.components import conversation

        self.conversation = conversation
        self.chat_messages = load("chat_messages")

    def _log(self) -> list:
        conversation = self.conversation
        return [
            conversation.UserContent(content="Is the hall light on?"),
            conversation.AssistantContent(
                agent_id="agent",
                content="Let me check.",
                thinking_content="I should look at the light.",
                tool_calls=[
                    llm.ToolInput(
                        tool_name="GetLiveContext", tool_args={}, id="call_1"
                    )
                ],
            ),
            conversation.ToolResultContent(
                agent_id="agent",
                tool_call_id="call_1",
                tool_name="GetLiveContext",
                **(
                    {"result": llm.ToolResult(data={"hall": "on"})}
                    if _chat_log_has_typed_results()
                    else {"tool_result": {"hall": "on"}}
                ),
            ),
            conversation.AssistantContent(
                agent_id="agent",
                content="Yes, it is on.",
                thinking_content="The context says on.",
            ),
            conversation.UserContent(content="Thanks"),
        ]

    def _assistant_reasoning(self, *, model: str, thinking_enabled: bool) -> list:
        messages = self.chat_messages.convert_content_to_messages(
            self._log(), model=model, thinking_enabled=thinking_enabled
        )
        return [
            message.get("reasoning_content")
            for message in messages
            if message["role"] == "assistant"
        ]

    def test_every_assistant_turn_carries_its_reasoning(self) -> None:
        self.assertEqual(
            self._assistant_reasoning(model="deepseek-flash", thinking_enabled=True),
            ["I should look at the light.", "The context says on."],
        )

    def test_none_is_sent_while_thinking_is_off(self) -> None:
        self.assertEqual(
            self._assistant_reasoning(model="deepseek-flash", thinking_enabled=False),
            [None, None],
        )

    def test_a_gateways_reasoner_never_gets_it(self) -> None:
        """deepseek-reasoner rejected reasoning in the messages array."""
        self.assertEqual(
            self._assistant_reasoning(model="deepseek-reasoner", thinking_enabled=True),
            [None, None],
        )


def _runtime_data() -> SimpleNamespace:
    """An entry's runtime data, as far as the web tools read it.

    ``usage.searches`` collects the provider id of every search that was
    counted, in place of the sensor the real tracker would increment.
    """
    searches: list[str] = []
    return SimpleNamespace(
        usage=SimpleNamespace(searches=searches, record_web_search=searches.append)
    )


class _StubProvider:
    """A search provider that needs no network, for cases about the tool itself.

    Answers with whatever it was given - a list of results, or an exception to
    raise - and records what it was asked, so a case can also prove that the
    tool never asked at all.
    """

    id = "stub"
    label = "Stub Search"

    def __init__(self, answer=(), *, max_query_chars=None, query_help="") -> None:
        self._answer = answer
        self.max_query_chars = max_query_chars
        self.query_help = query_help
        self.asked: list[tuple[str, int]] = []

    async def search(self, query: str, count: int) -> list:
        self.asked.append((query, count))
        if isinstance(self._answer, Exception):
            raise self._answer
        return list(self._answer)


class _FakeResponse:
    """Just enough of an httpx response for the tool to read a payload."""

    status_code = 200
    text = ""

    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def json(self) -> dict:
        return self._payload


class _FakeClient:
    """An httpx client that answers from memory and records the request."""

    def __init__(self, payload: dict) -> None:
        self._payload = payload
        self.requests: list[tuple[str, dict]] = []

    async def get(self, url: str, **kwargs) -> _FakeResponse:
        self.requests.append((url, kwargs))
        return _FakeResponse(self._payload)


@requires_ha
class TestWebSearchAnswersInTheCoreShape(unittest.TestCase):
    """A tool has to answer in the shape the running core asks for.

    Core 2026.10 expects ``llm.ToolResult`` and reports a bare JSON object,
    refusing it from 2027.11; the cores before it have no such class and take
    the object itself. The payload inside is the same either way, so that is
    asserted on every core and the wrapper only where there is one to assert.

    The network is faked rather than skipped, because the payload is built
    while reading Brave's answer and a wrapper around the wrong thing would
    otherwise still look right.
    """

    BRAVE_PAYLOAD = {
        "web": {
            "results": [
                {
                    "title": "Result one",
                    "url": "https://example.test/one",
                    "description": "First",
                },
                # No title and no url: dropped, so the test also proves the
                # payload is the tool's own and not Brave's handed through.
                {"title": None, "url": None, "description": "dropped"},
            ]
        }
    }

    def setUp(self) -> None:
        self.search_tool = load("web.search_tool")
        self.brave = load("web.brave")
        self.client = _FakeClient(self.BRAVE_PAYLOAD)

    def _answer(self):
        # The provider is handed its HTTP client, so the network is replaced
        # by passing a different one - nothing has to be patched.
        tool = self.search_tool.WebSearchTool(
            self.brave.BraveSearchProvider(
                lambda: self.client, api_key="brave-key", timeout=10.0
            ),
            lambda _provider: None,
        )
        return asyncio.run(
            tool.async_call(
                _FakeHass(),
                llm.ToolInput(
                    tool_name="web_search",
                    tool_args={"query": "pytest", "count": 2},
                    id="1",
                ),
                llm.LLMContext(
                    platform="deepseek_conversation",
                    context=None,
                    language="en",
                    assistant="conversation",
                    device_id=None,
                ),
            )
        )

    def test_the_payload_is_the_same_on_every_core(self) -> None:
        self.assertEqual(
            _result_data(self._answer()),
            {
                "query": "pytest",
                "results": [
                    {
                        "title": "Result one",
                        "url": "https://example.test/one",
                        "summary": "First",
                    }
                ],
            },
        )
        self.assertEqual(len(self.client.requests), 1)

    @requires_tool_result
    def test_this_core_gets_a_tool_result_rather_than_a_bare_object(self) -> None:
        """The deprecation itself: a bare object is what core reports on."""
        answer = self._answer()
        self.assertIsInstance(answer, llm.ToolResult)
        self.assertFalse(answer.error)


def _core_has_tool_annotations() -> bool:
    return not HA_REASON and hasattr(llm, "ToolAnnotations")


requires_tool_annotations = unittest.skipUnless(
    _core_has_tool_annotations(), "this core's llm.Tool has no annotations"
)


@requires_ha
class TestWebSearchToolContract(unittest.TestCase):
    """What the tool promises the model, whichever provider is behind it."""

    def setUp(self) -> None:
        self.search_tool = load("web.search_tool")
        self.web_types = load("web.types")
        #: Provider ids the tool reported as searches that found something.
        self.counted: list[str] = []

    def _call(self, provider, max_result_chars=0, **tool_args):
        tool = self.search_tool.WebSearchTool(
            provider, self.counted.append, max_result_chars=max_result_chars
        )
        return asyncio.run(
            tool.async_call(
                _FakeHass(),
                llm.ToolInput(tool_name="web_search", tool_args=tool_args, id="1"),
                llm.LLMContext(
                    platform="deepseek_conversation",
                    context=None,
                    language="en",
                    assistant="conversation",
                    device_id=None,
                ),
            )
        )

    def _result(self, **fields):
        return self.web_types.SearchResult(
            title="T", url="https://example.test", summary="S", **fields
        )

    def test_an_answer_over_the_agents_limit_survives_the_general_cut(self) -> None:
        """Fitted by the tool, so context_trim never has to cut results off."""
        context_trim = load("context_trim")
        const = load("const")
        long = " ".join(f"Line {i} of a long summary." for i in range(200))
        provider = _StubProvider(
            [
                self.web_types.SearchResult(title="T", url="https://e.test", summary=long)
                for _ in range(8)
            ]
        )
        data = _result_data(
            self._call(provider, max_result_chars=12_000, query="news", count=8)
        )
        sent = context_trim.format_tool_result_content(
            data,
            json_encoder=json.JSONEncoder,
            options={const.CONF_MAX_TOOL_RESULT_CHARS: 12_000},
            tool_name="web_search",
        )
        self.assertNotIn("truncated by DeepSeek integration", sent)
        self.assertEqual(len(json.loads(sent)["results"]), 8)

    def test_results_reach_the_model_without_the_ranking_signals(self) -> None:
        provider = _StubProvider(
            [self._result(source="Example", date="2026-06-01", score=2.4, authority=3)]
        )
        self.assertEqual(
            _result_data(self._call(provider, query="  ha release  ", count=3)),
            {
                "query": "ha release",
                "results": [
                    {
                        "title": "T",
                        "url": "https://example.test",
                        "source": "Example",
                        "date": "2026-06-01",
                        "summary": "S",
                    }
                ],
            },
        )
        self.assertEqual(provider.asked, [("ha release", 3)])

    def test_the_count_is_kept_within_what_the_tool_offers(self) -> None:
        for asked, sent in ((99, self.search_tool.MAX_RESULT_COUNT), (0, 1)):
            with self.subTest(asked=asked):
                provider = _StubProvider()
                self._call(provider, query="q", count=asked)
                self.assertEqual(provider.asked, [("q", sent)])

    def test_an_empty_search_says_so(self) -> None:
        """Otherwise the model answers from memory as if it had found something."""
        data = _result_data(self._call(_StubProvider(), query="q"))
        self.assertEqual(data["results"], [])
        self.assertIn("No results", data["note"])

    def test_a_search_with_results_carries_no_note(self) -> None:
        data = _result_data(self._call(_StubProvider([self._result()]), query="q"))
        self.assertNotIn("note", data)

    def test_a_query_over_the_provider_s_limit_is_refused_before_it_is_sent(self) -> None:
        provider = _StubProvider(max_query_chars=128)
        with self.assertRaises(HomeAssistantError) as caught:
            self._call(provider, query="x" * 129)
        message = str(caught.exception)
        self.assertIn("129", message)
        self.assertIn("128", message)
        self.assertIn("call web_search again", message)
        self.assertEqual(provider.asked, [])

    def test_a_query_at_the_limit_goes_through(self) -> None:
        provider = _StubProvider(max_query_chars=128)
        self._call(provider, query="x" * 128)
        self.assertEqual(len(provider.asked), 1)

    def test_a_provider_without_a_limit_is_not_checked(self) -> None:
        provider = _StubProvider(max_query_chars=None)
        self._call(provider, query="x" * 2000)
        self.assertEqual(len(provider.asked), 1)

    def test_a_provider_s_failure_reaches_the_model_in_its_own_words(self) -> None:
        """Core hands the text of a ``HomeAssistantError`` back as the result."""
        failure = self.web_types.WebToolError(
            self.web_types.WebErrorKind.BALANCE, "no balance left. Tell the user."
        )
        with self.assertRaises(HomeAssistantError) as caught:
            self._call(_StubProvider(failure), query="q")
        self.assertEqual(str(caught.exception), "no balance left. Tell the user.")

    def test_a_search_that_found_something_is_counted_once(self) -> None:
        """That is the unit DeepSeek bills, so it is the unit the sensor counts."""
        self._call(_StubProvider([self._result(), self._result()]), query="q")
        self.assertEqual(self.counted, ["stub"])

    def test_a_search_that_found_nothing_is_not_counted(self) -> None:
        self._call(_StubProvider(), query="q")
        self.assertEqual(self.counted, [])

    def test_a_search_that_failed_or_was_refused_is_not_counted(self) -> None:
        failure = self.web_types.WebToolError(
            self.web_types.WebErrorKind.UNAVAILABLE, "down"
        )
        for provider, query in (
            (_StubProvider(failure), "q"),
            (_StubProvider(max_query_chars=3), "too long"),
        ):
            with self.subTest(query=query), self.assertRaises(HomeAssistantError):
                self._call(provider, query=query)
        self.assertEqual(self.counted, [])

    def test_the_description_carries_the_provider_s_query_rules(self) -> None:
        with_rules = self.search_tool.WebSearchTool(
            _StubProvider(query_help="Query: at most 128 characters."),
            self.counted.append,
        )
        self.assertTrue(
            with_rules.description.endswith("Query: at most 128 characters.")
        )
        without = self.search_tool.WebSearchTool(_StubProvider(), self.counted.append)
        self.assertEqual(without.description, without.description.strip())

    def test_the_prompt_tells_the_model_not_to_obey_what_it_reads(self) -> None:
        """The one guard between a web page and an agent that controls a home."""
        prompt = self.search_tool.WEB_SEARCH_API_PROMPT
        self.assertIn("untrusted", prompt)
        self.assertIn("never operate the home", prompt)

    @requires_tool_annotations
    def test_the_tool_says_it_only_reads(self) -> None:
        """Core's defaults describe a tool that writes and destroys."""
        self.assertTrue(llm.Tool.annotations.destructive)
        annotations = self.search_tool.WebSearchTool(
            _StubProvider(), self.counted.append
        ).annotations
        self.assertTrue(annotations.read_only)
        self.assertFalse(annotations.destructive)
        self.assertTrue(annotations.open_world)


@requires_ha
class TestChatLogTakesThePrivateAPI(unittest.TestCase):
    """The handover itself: Home Assistant has to accept an API object.

    Everything else about #38 rests on ``async_provide_llm_data`` treating an
    ``llm.API`` the same as a registered id - if that ever stopped, the tools
    would quietly disappear rather than fail loudly, so it is pinned here
    against the real ``ChatLog``.
    """

    def _brave(self) -> dict:
        """An agent's settings with Brave chosen as its web search provider."""
        return {self.const.CONF_WEB_SEARCH: self.const.WEB_SEARCH_PROVIDER_BRAVE}

    def setUp(self) -> None:
        self.agent_tools = load("agent_tools")
        self.const = load("const")

    async def _chat_log(self, hass):
        from homeassistant.components.conversation import ChatLog

        llm.async_register_api(hass, _RecordedAPI(hass=hass, id=_SELECTED_API, name="Recorded"))
        entry = SimpleNamespace(
            entry_id="01J",
            title="DeepSeek",
            data={self.const.CONF_BRAVE_API_KEY: "brave-key"},
            runtime_data=_runtime_data(),
        )
        chat_log = ChatLog(hass, "conversation-1")
        await chat_log.async_provide_llm_data(
            llm_context=llm.LLMContext(
                platform="deepseek_conversation",
                context=None,
                language="en",
                assistant="conversation",
                device_id=None,
            ),
            user_llm_hass_api=self.agent_tools.agent_llm_api(
                hass, entry, self._brave(), [_SELECTED_API]
            ),
            user_llm_prompt="You are a test.",
        )
        return chat_log

    def test_the_chat_log_ends_up_with_both_sets_of_tools(self) -> None:
        async def check() -> None:
            hass = _minimal_hass()
            try:
                chat_log = await self._chat_log(hass)
                self.assertIsNotNone(chat_log.llm_api)
                self.assertEqual(
                    [tool.name for tool in chat_log.llm_api.tools],
                    ["recorded", "web_search"],
                )
                prompt = chat_log.content[0].content
                self.assertIn("Recorded API prompt", prompt)
                self.assertIn("web_search", prompt)
            finally:
                await hass.async_stop()

        asyncio.run(check())

    def test_core_dispatches_a_tool_call_to_each_of_them(self) -> None:
        """``async_call_tool`` resolves by name against the composed list."""

        async def check() -> None:
            hass = _minimal_hass()
            try:
                chat_log = await self._chat_log(hass)
                selected = await chat_log.llm_api.async_call_tool(
                    llm.ToolInput(tool_name="recorded", tool_args={}, id="1")
                )
                self.assertEqual(_result_data(selected), {"called": "recorded"})

                # An empty query is refused before any request is made, so this
                # proves the dispatch without going near the network.
                with self.assertRaises(HomeAssistantError) as caught:
                    await chat_log.llm_api.async_call_tool(
                        llm.ToolInput(
                            tool_name="web_search", tool_args={"query": "  "}, id="2"
                        )
                    )
                self.assertIn("non-empty query", str(caught.exception))

                with self.assertRaises(HomeAssistantError):
                    await chat_log.llm_api.async_call_tool(
                        llm.ToolInput(tool_name="nope", tool_args={}, id="3")
                    )
            finally:
                await hass.async_stop()

        asyncio.run(check())


@requires_ha
class TestUsageSensors(unittest.TestCase):
    """The sensor platform and the tracker have to agree on every counter.

    The tracker names its sensors as keyword arguments and the platform builds
    them, so a counter added to one and not the other is a ``TypeError`` at
    setup - which would take every usage sensor with it, not just the new one.
    """

    def setUp(self) -> None:
        try:
            self.sensor = load("sensor", api_stubs=True)
        except ImportError as err:  # pragma: no cover
            self.skipTest(f"the sensor platform needs more of core installed: {err}")
        self.tracker = load("usage_metrics").UsageTracker()
        entry = SimpleNamespace(
            entry_id="01J",
            title="DeepSeek",
            runtime_data=SimpleNamespace(usage=self.tracker),
        )
        added: list = []
        asyncio.run(self.sensor.async_setup_entry(None, entry, added.extend))
        self.by_key = {entity.translation_key: entity for entity in added}

    def test_a_counted_web_search_shows_on_its_sensor(self) -> None:
        self.tracker.record_web_search("deepseek")
        self.tracker.record_web_search("deepseek")
        self.assertEqual(self.by_key["web_search_requests"].native_value, 2)
        self.assertEqual(self.by_key["api_requests"].native_value or 0, 0)

    def test_reset_brings_it_back_to_zero(self) -> None:
        self.tracker.record_web_search("deepseek")
        self.tracker.reset_all()
        self.assertEqual(self.by_key["web_search_requests"].native_value, 0)

    def test_every_sensor_has_a_name_in_every_language(self) -> None:
        """A sensor without one shows up under its device's name alone."""
        translations = pathlib.Path(self.sensor.__file__).parent / "translations"
        for path in sorted(translations.glob("*.json")):
            with self.subTest(language=path.stem):
                names = json.loads(path.read_text(encoding="utf-8"))["entity"]["sensor"]
                self.assertEqual(sorted(names), sorted(self.by_key))


@requires_ha
class TestAgentFormWebSearchField(unittest.TestCase):
    """The agent form offers exactly what the entry can provide, and stores an id.

    Driven through the real subentry flow step rather than the helpers behind
    it: whether the field is on the form at all, which options it lists and
    what ends up in the subentry are all decided across ``config_flow.py``,
    ``flow_schemas.py`` and core's selector, and only the three together say
    what a user sees.
    """

    def setUp(self) -> None:
        try:
            self.config_flow = load("config_flow", api_stubs=True)
        except ImportError as err:  # pragma: no cover
            self.skipTest(f"config_flow needs more of core installed: {err}")
        self.const = load("const")

    def _flow(self, hass, *, entry_data: dict, options: dict):
        """A flow adding a conversation agent to an entry holding ``entry_data``."""
        from homeassistant.config_entries import ConfigEntryState

        flow = self.config_flow.DeepSeekSubentryFlowHandler()
        flow.hass = hass
        flow.flow_id = "flow-1"
        flow.handler = ("01JENTRY", self.const.SUBENTRY_TYPE_CONVERSATION)
        flow.context = {"source": "user"}
        flow.options = dict(options)
        entry = SimpleNamespace(
            entry_id="01JENTRY", data=entry_data, state=ConfigEntryState.LOADED
        )
        flow._get_entry = lambda: entry
        return flow

    def _run(self, *, entry_data: dict, options: dict, user_input: dict | None = None):
        async def step():
            hass = _minimal_hass()
            try:
                flow = self._flow(hass, entry_data=entry_data, options=options)
                return flow, await flow.async_step_init(user_input)
            finally:
                await hass.async_stop()

        return asyncio.run(step())

    def _field(self, result):
        """The web search field's selector on a shown form, or None."""
        for marker, selector_ in result["data_schema"].schema.items():
            if str(marker) == self.const.CONF_WEB_SEARCH:
                return selector_
        return None

    def _answers(self, **extra) -> dict:
        return {
            self.const.CONF_CHAT_MODEL: self.const.RECOMMENDED_CHAT_MODEL,
            self.const.CONF_RECOMMENDED: True,
            **extra,
        }

    KEY = {"brave_api_key": "brave-key"}
    #: A custom gateway: DeepSeek Search is offered on the official API only.
    GATEWAY = {"base_url": "https://gateway.example/v1"}

    def test_an_entry_with_nothing_to_offer_shows_no_field(self) -> None:
        _flow, result = self._run(entry_data=self.GATEWAY, options={})
        self.assertEqual(result["type"], "form")
        self.assertIsNone(self._field(result))

    def test_the_field_lists_off_and_what_the_entry_offers(self) -> None:
        deepseek = self.const.WEB_SEARCH_PROVIDER_DEEPSEEK
        brave = self.const.WEB_SEARCH_PROVIDER_BRAVE
        cases = (
            ({}, [deepseek]),  # the official API, and no Brave key
            (self.KEY, [deepseek, brave]),
            ({**self.GATEWAY, **self.KEY}, [brave]),
        )
        for entry_data, offered in cases:
            with self.subTest(entry_data=entry_data):
                _flow, result = self._run(entry_data=entry_data, options={})
                field = self._field(result)
                self.assertIsNotNone(field)
                self.assertEqual(
                    list(field.config["options"]),
                    [self.const.WEB_SEARCH_OFF, *offered],
                )

    def test_every_option_has_a_label_in_every_language(self) -> None:
        """An option without one is shown to the user as its raw id."""
        translations = pathlib.Path(self.config_flow.__file__).parent / "translations"
        for path in sorted(translations.glob("*.json")):
            with self.subTest(language=path.stem):
                labels = json.loads(path.read_text(encoding="utf-8"))["selector"][
                    self.const.CONF_WEB_SEARCH
                ]["options"]
                self.assertEqual(
                    sorted(labels),
                    sorted(
                        [
                            self.const.WEB_SEARCH_OFF,
                            self.const.WEB_SEARCH_PROVIDER_DEEPSEEK,
                            self.const.WEB_SEARCH_PROVIDER_BRAVE,
                        ]
                    ),
                )

    def test_a_chosen_provider_is_stored_by_its_id(self) -> None:
        _flow, result = self._run(
            entry_data=self.KEY,
            options={},
            user_input=self._answers(web_search=self.const.WEB_SEARCH_PROVIDER_BRAVE),
        )
        self.assertEqual(result["type"], "create_entry")
        self.assertEqual(
            result["data"][self.const.CONF_WEB_SEARCH],
            self.const.WEB_SEARCH_PROVIDER_BRAVE,
        )

    def test_off_stores_nothing(self) -> None:
        _flow, result = self._run(
            entry_data=self.KEY,
            options={self.const.CONF_WEB_SEARCH: self.const.WEB_SEARCH_PROVIDER_BRAVE},
            user_input=self._answers(web_search=self.const.WEB_SEARCH_OFF),
        )
        self.assertNotIn(self.const.CONF_WEB_SEARCH, result["data"])

    def test_a_choice_whose_key_is_gone_is_cleared_on_opening_the_form(self) -> None:
        """Cleared, not swapped for the provider that is still there."""
        flow, result = self._run(
            entry_data={},
            options={self.const.CONF_WEB_SEARCH: self.const.WEB_SEARCH_PROVIDER_BRAVE},
        )
        self.assertIsNotNone(self._field(result))
        self.assertNotIn(self.const.CONF_WEB_SEARCH, flow.options)


@requires_ha
class TestWebSearchMigration(unittest.TestCase):
    """The stored selection has to become the stored setting, exactly once."""

    def setUp(self) -> None:
        self.migration = load("migration", api_stubs=True)
        self.const = load("const")
        self.legacy = f"{self.const.LEGACY_WEB_SEARCH_API_ID_PREFIX}01JOLD"

    def _entry(self, *, version: int, subentries: list[dict]):
        from homeassistant.config_entries import ConfigEntry

        return ConfigEntry(
            version=version,
            minor_version=1,
            domain="deepseek_conversation",
            title="DeepSeek",
            data={},
            options={},
            source="user",
            unique_id=None,
            discovery_keys=MappingProxyType({}),
            subentries_data=subentries,
        )

    def _migrate(self, entry):
        """Run the migration with a config_entries stand-in that records writes."""
        written: list[tuple[str, dict]] = []

        class _ConfigEntries:
            @staticmethod
            def async_update_subentry(_entry, subentry, *, data):
                written.append((subentry.subentry_id, dict(data)))
                object.__setattr__(subentry, "data", MappingProxyType(dict(data)))

            @staticmethod
            def async_update_entry(target, **changes):
                for key, value in changes.items():
                    object.__setattr__(target, key, value)

        hass = SimpleNamespace(config_entries=_ConfigEntries())
        asyncio.run(self.migration.async_migrate_entry(hass, entry))
        return written

    def test_a_selected_web_search_api_becomes_the_setting(self) -> None:
        entry = self._entry(
            version=3,
            subentries=[
                {
                    "subentry_type": "conversation",
                    "title": "Voice",
                    "unique_id": None,
                    "data": {"llm_hass_api": ["assist", self.legacy]},
                }
            ],
        )
        written = self._migrate(entry)
        # Two writes: the selection becomes the setting, and the setting -
        # a switch at the time - then names its provider.
        self.assertEqual(len(written), 2)
        _subentry_id, data = written[-1]
        self.assertEqual(data, {"llm_hass_api": ["assist"], "web_search": "brave"})
        self.assertEqual(entry.version, 5)

    def test_an_agent_that_never_had_it_is_not_rewritten(self) -> None:
        entry = self._entry(
            version=3,
            subentries=[
                {
                    "subentry_type": "conversation",
                    "title": "Voice",
                    "unique_id": None,
                    "data": {"llm_hass_api": ["assist"]},
                }
            ],
        )
        self.assertEqual(self._migrate(entry), [])
        self.assertEqual(entry.version, 5)

    def test_running_it_again_changes_nothing(self) -> None:
        """A restored backup must not re-enter a migration that already ran."""
        entry = self._entry(
            version=3,
            subentries=[
                {
                    "subentry_type": "conversation",
                    "title": "Voice",
                    "unique_id": None,
                    "data": {"llm_hass_api": [self.legacy]},
                }
            ],
        )
        self.assertEqual(len(self._migrate(entry)), 2)
        self.assertEqual(self._migrate(entry), [])

    def test_every_agent_on_the_entry_is_migrated(self) -> None:
        entry = self._entry(
            version=3,
            subentries=[
                {
                    "subentry_type": "conversation",
                    "title": "Voice",
                    "unique_id": None,
                    "data": {"llm_hass_api": [self.legacy]},
                },
                {
                    "subentry_type": "ai_task_data",
                    "title": "Task",
                    "unique_id": None,
                    "data": {"llm_hass_api": ["assist", self.legacy]},
                },
            ],
        )
        self.assertEqual(len(self._migrate(entry)), 4)

    def test_a_web_search_switch_names_its_provider(self) -> None:
        """Version 4 stored a switch; it only ever meant Brave."""
        entry = self._entry(
            version=4,
            subentries=[
                {
                    "subentry_type": "conversation",
                    "title": "Voice",
                    "unique_id": None,
                    "data": {"llm_hass_api": ["assist"], "web_search": True},
                },
                {
                    "subentry_type": "ai_task_data",
                    "title": "Task",
                    "unique_id": None,
                    "data": {"web_search": False, "chat_model": "m"},
                },
                {
                    "subentry_type": "conversation",
                    "title": "Plain",
                    "unique_id": None,
                    "data": {"llm_hass_api": ["assist"]},
                },
            ],
        )
        written = [data for _subentry_id, data in self._migrate(entry)]
        self.assertEqual(
            written,
            [
                {"llm_hass_api": ["assist"], "web_search": "brave"},
                {"chat_model": "m"},
            ],
        )
        self.assertEqual(entry.version, 5)
        self.assertEqual(self._migrate(entry), [])


@requires_ha
class TestUpgradeFromV18(unittest.TestCase):
    """What an entry saved by 1.8.x looks like after the first start on 1.9.

    1.8.x stored ``deepseek-v4-flash`` in every new agent, so this is the path
    nearly every user takes.
    """

    def setUp(self) -> None:
        self.migration = load("migration", api_stubs=True)

    def _entry(self, models: dict[str, str], *, base_url: str | None = None):
        from homeassistant.config_entries import ConfigEntry

        return ConfigEntry(
            version=4,
            minor_version=1,
            domain="deepseek_conversation",
            title="DeepSeek",
            data={"base_url": base_url} if base_url else {},
            options={},
            source="user",
            unique_id=None,
            discovery_keys=MappingProxyType({}),
            subentries_data=[
                {
                    "subentry_type": "conversation",
                    "title": title,
                    "unique_id": None,
                    "data": {"chat_model": model, "web_search": True},
                }
                for title, model in models.items()
            ],
        )

    def _start(self, entry):
        """Run both migrations; return each agent's model and the issues raised."""

        class _ConfigEntries:
            @staticmethod
            def async_update_subentry(_entry, subentry, *, data):
                object.__setattr__(subentry, "data", MappingProxyType(dict(data)))

            @staticmethod
            def async_update_entry(target, **changes):
                for key, value in changes.items():
                    object.__setattr__(target, key, value)

        hass = SimpleNamespace(config_entries=_ConfigEntries())
        asyncio.run(self.migration.async_migrate_entry(hass, entry))
        with mock.patch.object(self.migration.ir, "async_create_issue") as issue:
            self.migration.async_migrate_legacy_model_option(hass, entry)
        agents = {
            sub.title: (sub.data.get("chat_model"), sub.data.get("web_search"))
            for sub in entry.subentries.values()
        }
        return agents, [call.kwargs["translation_placeholders"] for call in issue.call_args_list]

    def test_the_v4_flash_ids_move_quietly(self) -> None:
        """DeepSeek already answered them with V4.1 Flash: nothing to dismiss."""
        agents, issues = self._start(
            self._entry(
                {
                    "Voice": "deepseek-v4-flash",
                    "Vision": "deepseek-v4-flash-vision-exp",
                    "Pro": "deepseek-v4-pro",
                }
            )
        )
        self.assertEqual(
            agents,
            {
                "Voice": ("deepseek-flash", "brave"),
                "Vision": ("deepseek-flash", "brave"),
                "Pro": ("deepseek-v4-pro", "brave"),
            },
        )
        self.assertEqual(issues, [])

    def test_a_dead_id_is_reported_with_every_id_it_replaced(self) -> None:
        agents, issues = self._start(
            self._entry(
                {
                    "Chat": "deepseek-chat",
                    "Reasoner": "deepseek-reasoner",
                    "Voice": "deepseek-v4-flash",
                }
            )
        )
        self.assertEqual({model for model, _ in agents.values()}, {"deepseek-flash"})
        self.assertEqual(
            issues,
            [
                {
                    "entry_title": "Chat, Reasoner",
                    "old_model": "deepseek-chat, deepseek-reasoner",
                    "new_model": "deepseek-flash",
                    "retired_on": "2026-07-24",
                }
            ],
        )

    def test_a_gateway_keeps_its_ids(self) -> None:
        agents, issues = self._start(
            self._entry(
                {"Voice": "deepseek-v4-flash"}, base_url="https://openrouter.ai/api/v1"
            )
        )
        self.assertEqual(agents, {"Voice": ("deepseek-v4-flash", "brave")})
        self.assertEqual(issues, [])

    def test_a_second_start_changes_nothing(self) -> None:
        entry = self._entry({"Chat": "deepseek-chat", "Voice": "deepseek-v4-flash"})
        self._start(entry)
        agents, issues = self._start(entry)
        self.assertEqual(issues, [])
        self.assertEqual({model for model, _ in agents.values()}, {"deepseek-flash"})


if __name__ == "__main__":
    unittest.main()
