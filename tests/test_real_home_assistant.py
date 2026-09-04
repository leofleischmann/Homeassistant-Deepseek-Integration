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
from types import MappingProxyType, SimpleNamespace
import unittest

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
    from homeassistant.helpers import config_validation as cv, llm, selector

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


#: Core's ``ToolInput`` grew ``external`` so a call with unreadable arguments
#: can be reported back. Older cores have no such field and the integration
#: falls back to dropping the call, so the tests for it skip rather than fail.
def _tool_input_has_external() -> bool:
    if HA_REASON:
        return False
    import dataclasses

    return "external" in {field.name for field in dataclasses.fields(llm.ToolInput)}


requires_external = unittest.skipUnless(
    _tool_input_has_external(), "this core's ToolInput has no 'external' field"
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


if not HA_REASON:

    class _RecordedAPI(llm.API):
        """A registered API standing in for Assist, with one tool of its own."""

        async def async_get_api_instance(self, llm_context):
            class _Tool(llm.Tool):
                name = "recorded"
                description = "a tool the selected API brought along"
                parameters = vol.Schema({vol.Optional("name"): cv.string})

                async def async_call(self, hass, tool_input, llm_context):
                    return {"called": "recorded"}

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
        """
        mine, theirs = self.converters[0], self.converters[1]

        def mixed_dialect(node):
            return theirs.unsupported if node is cv.string else mine.unsupported

        rendered = self.openapi_schema.render_openapi_schema(
            _schemas()["plain_strings"], custom_serializer=mixed_dialect
        )
        self.assertEqual(rendered["properties"]["name"], {"type": "string"})
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
        if _tool_input_has_external():
            self.assertFalse(inputs[0].external)

    @requires_external
    def test_unreadable_arguments_are_reported_and_not_executed(self) -> None:
        """Reporting the parse error back needs core's ``external`` field."""
        inputs, failures = self.stream_transform._finalize_tool_calls(
            self._assemble(
                self._chunk(0, id="call_1", name="HassTurnOn"),
                self._chunk(0, arguments='{"name": '),
            )
        )
        self.assertTrue(inputs[0].external)
        self.assertEqual(len(failures), 1)
        self.assertEqual(failures[0]["tool_result"]["error"], "InvalidToolArguments")

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
            hass=self.hass, id="assist", name="Assist"
        )
        llm.async_register_api(self.hass, self.registered)

    def _entry(self, *, brave: str | None = "brave-key"):
        data = {self.const.CONF_BRAVE_API_KEY: brave} if brave else {}
        return SimpleNamespace(entry_id=self.ENTRY_ID, title="DeepSeek", data=data)

    async def _tools(self, options, selected):
        api = self.agent_tools.agent_llm_api(
            self.hass, self._entry(), options, selected
        )
        instance = await api.async_get_api_instance(self.context)
        return instance, [tool.name for tool in instance.tools]

    def test_the_tool_is_never_put_in_the_shared_registry(self) -> None:
        """The bug itself: what is registered is what other integrations offer."""
        self.assertEqual(
            [api.id for api in llm.async_get_apis(self.hass)], ["assist"]
        )

    def test_an_agent_without_web_search_uses_home_assistant_s_own_path(self) -> None:
        """Passing the selection through unchanged keeps the common case core's."""
        selection = self.agent_tools.agent_llm_api(
            self.hass, self._entry(), {}, ["assist"]
        )
        self.assertEqual(selection, ["assist"])

    def test_web_search_is_added_beside_the_selected_tools(self) -> None:
        instance, names = asyncio.run(
            self._tools({self.const.CONF_WEB_SEARCH: True}, ["assist"])
        )
        self.assertEqual(names, ["recorded", "web_search"])
        self.assertIn("Recorded API prompt", instance.api_prompt)
        self.assertIn("web_search", instance.api_prompt)
        self.assertIs(instance.custom_serializer, llm.selector_serializer)

    def test_an_agent_with_no_home_assistant_api_gets_web_search_alone(self) -> None:
        _instance, names = asyncio.run(
            self._tools({self.const.CONF_WEB_SEARCH: True}, None)
        )
        self.assertEqual(names, ["web_search"])

    def test_a_selection_that_no_longer_exists_fails_as_core_would(self) -> None:
        """Switching web search on must not change how a broken selection behaves.

        An integration removed since the agent was configured leaves its id in
        the subentry; Home Assistant refuses that, and so do we, rather than
        quietly answering without control over the home.
        """
        with self.assertRaises(HomeAssistantError):
            asyncio.run(
                self._tools({self.const.CONF_WEB_SEARCH: True}, ["assist", "gone"])
            )

    def test_the_option_without_a_key_answers_without_the_tool(self) -> None:
        """The key can be removed through Reconfigure while the switch stays on."""
        api = self.agent_tools.agent_llm_api(
            self.hass,
            self._entry(brave=None),
            {self.const.CONF_WEB_SEARCH: True},
            ["assist"],
        )
        # No key means no web search at all, so the selection is passed through.
        self.assertEqual(api, ["assist"])

    def test_a_key_removed_after_the_agent_was_built_is_survivable(self) -> None:
        """``agent_llm_api`` looks once; the instance is built per turn."""
        api = self.agent_tools.AgentToolsAPI(
            hass=self.hass, entry=self._entry(brave=None), selected=["assist"]
        )
        instance = asyncio.run(api.async_get_api_instance(self.context))
        self.assertEqual([tool.name for tool in instance.tools], ["recorded"])

    def test_the_tools_render_for_the_api_like_any_other(self) -> None:
        chat_session = _load_chat_session()
        instance, _names = asyncio.run(
            self._tools({self.const.CONF_WEB_SEARCH: True}, ["assist"])
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
class TestChatLogTakesThePrivateAPI(unittest.TestCase):
    """The handover itself: Home Assistant has to accept an API object.

    Everything else about #38 rests on ``async_provide_llm_data`` treating an
    ``llm.API`` the same as a registered id - if that ever stopped, the tools
    would quietly disappear rather than fail loudly, so it is pinned here
    against the real ``ChatLog``.
    """

    def setUp(self) -> None:
        self.agent_tools = load("agent_tools")
        self.const = load("const")

    async def _chat_log(self, hass):
        from homeassistant.components.conversation import ChatLog

        llm.async_register_api(hass, _RecordedAPI(hass=hass, id="assist", name="Assist"))
        entry = SimpleNamespace(
            entry_id="01J",
            title="DeepSeek",
            data={self.const.CONF_BRAVE_API_KEY: "brave-key"},
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
                hass, entry, {self.const.CONF_WEB_SEARCH: True}, ["assist"]
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
                self.assertEqual(selected, {"called": "recorded"})

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
        self.assertEqual(len(written), 1)
        _subentry_id, data = written[0]
        self.assertEqual(data, {"llm_hass_api": ["assist"], "web_search": True})
        self.assertEqual(entry.version, 4)

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
        self.assertEqual(entry.version, 4)

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
        self.assertEqual(len(self._migrate(entry)), 1)
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
        self.assertEqual(len(self._migrate(entry)), 2)


if __name__ == "__main__":
    unittest.main()
