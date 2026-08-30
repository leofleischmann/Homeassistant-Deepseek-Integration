"""Tests for request_builder: what actually goes on the wire.

The Assist loop, the ``generate_content`` action and the debug suite all build
their requests here, so these tests are what keeps them from drifting apart on
thinking flags, sampling parameters and the token limit.

Pure standard library, so they run without Home Assistant installed::

    python -m unittest discover -s tests
"""

from __future__ import annotations

import unittest

from integration_modules import load

const = load("const")
request_builder = load("request_builder")

build_args = request_builder.build_chat_completion_args
thinking_params = request_builder.deepseek_chat_thinking_params

_DEEPSEEK_MODEL = const.RECOMMENDED_CHAT_MODEL
_GATEWAY_MODEL = "gpt-4o"
_MESSAGES = [{"role": "user", "content": "hi"}]


class TestThinkingParams(unittest.TestCase):
    def test_off_is_sent_explicitly(self) -> None:
        """V4 defaults to reasoning on, so "off" has to be stated, not omitted."""
        self.assertEqual(
            thinking_params(thinking_enabled=False, model=_DEEPSEEK_MODEL),
            {"extra_body": {"thinking": {"type": "disabled"}}},
        )

    def test_on_carries_the_effort(self) -> None:
        self.assertEqual(
            thinking_params(
                thinking_enabled=True, reasoning_effort="low", model=_DEEPSEEK_MODEL
            ),
            {
                "extra_body": {"thinking": {"type": "enabled"}},
                "reasoning_effort": "low",
            },
        )

    def test_an_unknown_effort_falls_back(self) -> None:
        self.assertEqual(
            thinking_params(
                thinking_enabled=True, reasoning_effort="turbo", model=_DEEPSEEK_MODEL
            )["reasoning_effort"],
            const.RECOMMENDED_REASONING_EFFORT,
        )

    def test_a_foreign_model_gets_no_deepseek_fields(self) -> None:
        """An OpenAI-compatible proxy rejects or ignores extra_body."""
        self.assertEqual(thinking_params(thinking_enabled=True, model=_GATEWAY_MODEL), {})
        self.assertEqual(thinking_params(thinking_enabled=False, model=_GATEWAY_MODEL), {})

    def test_extra_body_helper(self) -> None:
        self.assertEqual(
            request_builder.deepseek_chat_extra_body(thinking_enabled=True),
            {"thinking": {"type": "enabled"}},
        )


class TestBuildChatCompletionArgs(unittest.TestCase):
    def test_defaults_of_an_agent_on_the_recommended_settings(self) -> None:
        args = build_args(
            model=_DEEPSEEK_MODEL, messages=_MESSAGES, options={}, stream=False
        )
        self.assertEqual(args["model"], _DEEPSEEK_MODEL)
        self.assertEqual(args["messages"], _MESSAGES)
        self.assertEqual(args["max_tokens"], const.RECOMMENDED_MAX_TOKENS)
        self.assertIs(args["stream"], False)
        self.assertEqual(args["temperature"], const.RECOMMENDED_TEMPERATURE)
        self.assertEqual(args["top_p"], const.RECOMMENDED_TOP_P)
        self.assertEqual(args["extra_body"], {"thinking": {"type": "disabled"}})
        for absent in ("tools", "tool_choice", "response_format", "stream_options"):
            self.assertNotIn(absent, args)

    def test_thinking_replaces_the_sampling_parameters(self) -> None:
        """DeepSeek ignores temperature/top_p in thinking mode; sending them is noise."""
        args = build_args(
            model=_DEEPSEEK_MODEL,
            messages=_MESSAGES,
            options={
                const.CONF_THINKING_ENABLED: True,
                const.CONF_TEMPERATURE: 0.3,
                const.CONF_TOP_P: 0.5,
                const.CONF_REASONING_EFFORT: "max",
            },
            stream=False,
        )
        self.assertNotIn("temperature", args)
        self.assertNotIn("top_p", args)
        self.assertEqual(args["reasoning_effort"], "max")

    def test_streaming_asks_for_usage(self) -> None:
        """Without this the token sensors never see a streamed turn."""
        args = build_args(
            model=_DEEPSEEK_MODEL, messages=_MESSAGES, options={}, stream=True
        )
        self.assertEqual(args["stream_options"], {"include_usage": True})

    def test_max_tokens_is_coerced_not_trusted(self) -> None:
        args = build_args(
            model=_DEEPSEEK_MODEL,
            messages=_MESSAGES,
            options={const.CONF_MAX_TOKENS: "8000"},
            stream=False,
        )
        self.assertEqual(args["max_tokens"], 8000)

    def test_an_empty_tool_list_is_omitted(self) -> None:
        """Sending ``tools: []`` makes some gateways reject the request."""
        args = build_args(
            model=_DEEPSEEK_MODEL,
            messages=_MESSAGES,
            options={},
            stream=False,
            tools=[],
            tool_choice=None,
        )
        self.assertNotIn("tools", args)
        self.assertNotIn("tool_choice", args)

    def test_tools_and_response_format_pass_through(self) -> None:
        tools = [{"type": "function", "function": {"name": "x"}}]
        args = build_args(
            model=_DEEPSEEK_MODEL,
            messages=_MESSAGES,
            options={},
            stream=False,
            tools=tools,
            tool_choice="auto",
            response_format={"type": const.RESPONSE_FORMAT_JSON_OBJECT},
        )
        self.assertEqual(args["tools"], tools)
        self.assertEqual(args["tool_choice"], "auto")
        self.assertEqual(
            args["response_format"], {"type": const.RESPONSE_FORMAT_JSON_OBJECT}
        )

    def test_a_gateway_model_gets_sampling_but_no_extra_body(self) -> None:
        args = build_args(
            model=_GATEWAY_MODEL, messages=_MESSAGES, options={}, stream=False
        )
        self.assertNotIn("extra_body", args)
        self.assertEqual(args["temperature"], const.RECOMMENDED_TEMPERATURE)


class TestResolveGenerateContentModel(unittest.TestCase):
    resolve = staticmethod(request_builder.resolve_generate_content_model)

    def test_falls_back_to_the_agent_then_the_recommended_model(self) -> None:
        self.assertEqual(self.resolve({}, {}), const.RECOMMENDED_CHAT_MODEL)
        self.assertEqual(
            self.resolve({const.CONF_CHAT_MODEL: "deepseek-v4-pro"}, {}),
            "deepseek-v4-pro",
        )

    def test_a_per_call_override_wins(self) -> None:
        self.assertEqual(
            self.resolve(
                {const.CONF_CHAT_MODEL: "deepseek-v4-pro"},
                {const.CONF_CHAT_MODEL: "  deepseek-v4-flash  "},
            ),
            "deepseek-v4-flash",
        )

    def test_a_blank_override_is_not_an_override(self) -> None:
        for blank in ("", "   "):
            with self.subTest(override=blank):
                self.assertEqual(
                    self.resolve(
                        {const.CONF_CHAT_MODEL: "deepseek-v4-pro"},
                        {const.CONF_CHAT_MODEL: blank},
                    ),
                    "deepseek-v4-pro",
                )


class TestGenerateContentArgs(unittest.TestCase):
    def test_agent_settings_are_the_baseline(self) -> None:
        model, args = request_builder.build_generate_content_completion_args(
            agent_options={
                const.CONF_CHAT_MODEL: "deepseek-v4-pro",
                const.CONF_MAX_TOKENS: 700,
            },
            messages=_MESSAGES,
            service_data={},
        )
        self.assertEqual(model, "deepseek-v4-pro")
        self.assertEqual(args["max_tokens"], 700)
        self.assertIs(args["stream"], False)
        self.assertNotIn("stream_options", args)

    def test_per_call_overrides_are_applied(self) -> None:
        _model, args = request_builder.build_generate_content_completion_args(
            agent_options={const.CONF_MAX_TOKENS: 700, const.CONF_TEMPERATURE: 1.0},
            messages=_MESSAGES,
            service_data={
                const.CONF_MAX_TOKENS: 32,
                const.CONF_TEMPERATURE: 0.1,
                const.CONF_RESPONSE_FORMAT: const.RESPONSE_FORMAT_JSON_OBJECT,
            },
        )
        self.assertEqual(args["max_tokens"], 32)
        self.assertEqual(args["temperature"], 0.1)
        self.assertEqual(
            args["response_format"], {"type": const.RESPONSE_FORMAT_JSON_OBJECT}
        )

    def test_switching_thinking_on_per_call_drops_the_sampling_parameters(self) -> None:
        _model, args = request_builder.build_generate_content_completion_args(
            agent_options={},
            messages=_MESSAGES,
            service_data={
                const.CONF_THINKING_ENABLED: True,
                const.CONF_TEMPERATURE: 0.1,
            },
        )
        self.assertNotIn("temperature", args)
        self.assertEqual(args["extra_body"], {"thinking": {"type": "enabled"}})

    def test_an_already_migrated_model_is_not_resolved_again(self) -> None:
        """The caller migrates a retired id before the request is built."""
        model, args = request_builder.build_generate_content_completion_args(
            agent_options={const.CONF_CHAT_MODEL: "deepseek-chat"},
            messages=_MESSAGES,
            service_data={},
            model=const.RECOMMENDED_CHAT_MODEL,
        )
        self.assertEqual(model, const.RECOMMENDED_CHAT_MODEL)
        self.assertEqual(args["model"], const.RECOMMENDED_CHAT_MODEL)


class TestEffectiveThinkingForGenerateContent(unittest.TestCase):
    effective = staticmethod(
        request_builder.effective_thinking_enabled_for_generate_content
    )

    def test_falls_back_to_the_agent_setting(self) -> None:
        self.assertIs(self.effective({}, {}), const.DEFAULT_THINKING_ENABLED)
        self.assertIs(self.effective({const.CONF_THINKING_ENABLED: True}, {}), True)

    def test_a_per_call_false_beats_an_agent_true(self) -> None:
        self.assertIs(
            self.effective(
                {const.CONF_THINKING_ENABLED: True},
                {const.CONF_THINKING_ENABLED: False},
            ),
            False,
        )


class TestReasoningTextFromMessage(unittest.TestCase):
    class _Message:
        def __init__(self, **attrs: object) -> None:
            for key, value in attrs.items():
                setattr(self, key, value)

    def test_reads_either_field_name(self) -> None:
        self.assertEqual(
            request_builder.reasoning_text_from_chat_message(
                self._Message(reasoning_content="because")
            ),
            "because",
        )
        self.assertEqual(
            request_builder.reasoning_text_from_chat_message(
                self._Message(reasoning="because")
            ),
            "because",
        )

    def test_a_null_field_falls_through_to_the_next(self) -> None:
        self.assertEqual(
            request_builder.reasoning_text_from_chat_message(
                self._Message(reasoning_content=None, reasoning="because")
            ),
            "because",
        )

    def test_no_reasoning_is_an_empty_string(self) -> None:
        self.assertEqual(
            request_builder.reasoning_text_from_chat_message(self._Message()), ""
        )


if __name__ == "__main__":
    unittest.main()
