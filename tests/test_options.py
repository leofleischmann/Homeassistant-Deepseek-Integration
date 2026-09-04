"""Tests for options: reading stored agent settings.

Every value here comes out of a config subentry, which may predate the current
form, so each read has to survive a missing key, a string where a number
belongs, or a value outside the bounds the form now enforces.

Pure standard library, so they run without Home Assistant installed::

    python -m unittest discover -s tests
"""

from __future__ import annotations

import unittest

from integration_modules import load

const = load("const")
options = load("options")

#: Read from const rather than spelled out: it is Home Assistant's key, and
#: this is the module under test's own view of it.
LLM_API = const.CONF_LLM_HASS_API


class TestCoerceMaxTokens(unittest.TestCase):
    def test_usable_values_pass_through(self) -> None:
        self.assertEqual(options.coerce_max_tokens(2000), 2000)
        # A NumberSelector hands back a float, and older entries stored strings.
        self.assertEqual(options.coerce_max_tokens(2000.0), 2000)
        self.assertEqual(options.coerce_max_tokens("2000"), 2000)

    def test_unusable_values_fall_back(self) -> None:
        for value in (None, "", "abc", [], {}):
            with self.subTest(value=value):
                self.assertEqual(
                    options.coerce_max_tokens(value), const.RECOMMENDED_MAX_TOKENS
                )
        self.assertEqual(options.coerce_max_tokens(None, fallback=99), 99)

    def test_clamped_to_the_bounds(self) -> None:
        self.assertEqual(options.coerce_max_tokens(0), 1)
        self.assertEqual(options.coerce_max_tokens(-10), 1)
        self.assertEqual(
            options.coerce_max_tokens(const.MAX_TOKENS_UPPER_BOUND * 2),
            const.MAX_TOKENS_UPPER_BOUND,
        )


class TestCoerceMaxToolIterations(unittest.TestCase):
    def test_clamped_to_the_bounds(self) -> None:
        self.assertEqual(options.coerce_max_tool_iterations(5), 5)
        self.assertEqual(options.coerce_max_tool_iterations(0), 1)
        self.assertEqual(
            options.coerce_max_tool_iterations(999),
            const.MAX_TOOL_ITERATIONS_UPPER_BOUND,
        )

    def test_unusable_values_fall_back(self) -> None:
        self.assertEqual(
            options.coerce_max_tool_iterations("nope"),
            const.RECOMMENDED_MAX_TOOL_ITERATIONS,
        )


class TestRequestTimeout(unittest.TestCase):
    def test_zero_and_negative_mean_unset(self) -> None:
        # Zero is not "no timeout" here: the SDK default of 600 s would let one
        # unresponsive endpoint block a voice pipeline.
        self.assertEqual(
            options.coerce_request_timeout(0), float(const.RECOMMENDED_REQUEST_TIMEOUT)
        )
        self.assertEqual(
            options.coerce_request_timeout(-1), float(const.RECOMMENDED_REQUEST_TIMEOUT)
        )

    def test_clamped_to_the_bounds(self) -> None:
        self.assertEqual(
            options.coerce_request_timeout(1), float(const.REQUEST_TIMEOUT_LOWER_BOUND)
        )
        self.assertEqual(
            options.coerce_request_timeout(10_000),
            float(const.REQUEST_TIMEOUT_UPPER_BOUND),
        )
        self.assertEqual(options.coerce_request_timeout(45), 45.0)

    def test_streamed_calls_read_the_option(self) -> None:
        self.assertEqual(
            options.request_timeout_from_options({}),
            float(const.RECOMMENDED_REQUEST_TIMEOUT),
        )
        self.assertEqual(
            options.request_timeout_from_options({const.CONF_REQUEST_TIMEOUT: 90}), 90.0
        )

    def test_blocking_calls_never_go_below_the_floor(self) -> None:
        """The option is a stall detector; a blocking call needs the whole run."""
        self.assertEqual(
            options.blocking_request_timeout_from_options({}),
            float(const.MIN_BLOCKING_REQUEST_TIMEOUT),
        )
        self.assertEqual(
            options.blocking_request_timeout_from_options(
                {const.CONF_REQUEST_TIMEOUT: const.REQUEST_TIMEOUT_UPPER_BOUND}
            ),
            float(const.REQUEST_TIMEOUT_UPPER_BOUND),
        )


class TestToolResultCharLimit(unittest.TestCase):
    def test_zero_disables_truncation(self) -> None:
        self.assertEqual(options.coerce_max_tool_result_chars(0), 0)
        self.assertEqual(options.coerce_max_tool_result_chars(-1), 0)

    def test_a_non_zero_value_leaves_room_for_the_notice(self) -> None:
        self.assertEqual(
            options.coerce_max_tool_result_chars(100), const.MIN_TOOL_RESULT_CHARS
        )
        self.assertEqual(
            options.coerce_max_tool_result_chars(10**9),
            const.MAX_TOOL_RESULT_CHARS_UPPER_BOUND,
        )
        self.assertEqual(options.coerce_max_tool_result_chars(8000), 8000)

    def test_read_from_options(self) -> None:
        self.assertEqual(
            options.max_tool_result_chars_from_options({}),
            const.RECOMMENDED_MAX_TOOL_RESULT_CHARS,
        )
        self.assertEqual(
            options.max_tool_result_chars_from_options(
                {const.CONF_MAX_TOOL_RESULT_CHARS: 0}
            ),
            0,
        )


class TestHistoryRoundLimit(unittest.TestCase):
    def test_zero_keeps_the_whole_history(self) -> None:
        self.assertEqual(options.coerce_max_history_rounds(0), 0)
        self.assertEqual(options.coerce_max_history_rounds(-4), 0)

    def test_clamped_to_the_upper_bound(self) -> None:
        self.assertEqual(options.coerce_max_history_rounds(6), 6)
        self.assertEqual(
            options.coerce_max_history_rounds(10_000),
            const.MAX_HISTORY_ROUNDS_UPPER_BOUND,
        )

    def test_unusable_values_fall_back(self) -> None:
        self.assertEqual(options.coerce_max_history_rounds("x", fallback=5), 5)

    def test_read_from_options(self) -> None:
        self.assertEqual(
            options.max_history_rounds_from_options({}),
            const.RECOMMENDED_MAX_HISTORY_ROUNDS,
        )
        self.assertEqual(
            options.max_history_rounds_from_options({const.CONF_MAX_HISTORY_ROUNDS: 3}),
            3,
        )


class TestReasoningEffort(unittest.TestCase):
    def test_known_values_pass_through(self) -> None:
        for value in const.REASONING_EFFORT_VALUES:
            with self.subTest(value=value):
                self.assertEqual(options.normalized_reasoning_effort(value), value)

    def test_anything_else_becomes_the_recommended_effort(self) -> None:
        for value in (None, "", "LOW", "extreme", 3, ["high"]):
            with self.subTest(value=value):
                self.assertEqual(
                    options.normalized_reasoning_effort(value),
                    const.RECOMMENDED_REASONING_EFFORT,
                )


class TestAgentOptionReshaping(unittest.TestCase):
    def test_going_back_to_recommended_forgets_the_overrides(self) -> None:
        """Otherwise the agent runs on values its own form no longer shows."""
        stored = {
            const.CONF_RECOMMENDED: True,
            const.CONF_PROMPT: "hi",
            const.CONF_CHAT_MODEL: "deepseek-v4-pro",
            const.CONF_MAX_TOKENS: 42,
            const.CONF_REASONING_EFFORT: "low",
            const.CONF_TEMPERATURE: 0.3,
        }
        self.assertEqual(
            options.recommended_agent_options(stored),
            {
                const.CONF_RECOMMENDED: True,
                const.CONF_PROMPT: "hi",
                const.CONF_CHAT_MODEL: "deepseek-v4-pro",
            },
        )

    def test_an_ai_task_drops_the_assist_only_settings(self) -> None:
        stored = {
            const.CONF_CHAT_MODEL: "deepseek-v4-flash",
            const.CONF_MAX_TOKENS: 900,
            const.CONF_STRIP_MARKDOWN: True,
            const.CONF_INCLUDE_USER_CONTEXT: True,
            const.CONF_MAX_HISTORY_ROUNDS: 4,
        }
        self.assertEqual(
            options.ai_task_options_from(stored),
            {
                const.CONF_CHAT_MODEL: "deepseek-v4-flash",
                const.CONF_MAX_TOKENS: 900,
            },
        )
        self.assertIn(const.CONF_STRIP_MARKDOWN, stored, "input must not be mutated")


class TestStrippedMarkdownDefaultAdoption(unittest.TestCase):
    def test_the_old_default_gives_way(self) -> None:
        """1.7.0 wrote the default into every entry, so a stored False says nothing."""
        self.assertEqual(options.adopt_strip_markdown_default({
            const.CONF_STRIP_MARKDOWN: const.PREVIOUS_STRIP_MARKDOWN_DEFAULT,
            const.CONF_MAX_TOKENS: 10,
        }), {const.CONF_MAX_TOKENS: 10})

    def test_a_deliberate_choice_is_kept(self) -> None:
        deliberate = not const.PREVIOUS_STRIP_MARKDOWN_DEFAULT
        self.assertEqual(
            options.adopt_strip_markdown_default(
                {const.CONF_STRIP_MARKDOWN: deliberate}
            ),
            {const.CONF_STRIP_MARKDOWN: deliberate},
        )

    def test_an_absent_setting_is_untouched(self) -> None:
        self.assertEqual(options.adopt_strip_markdown_default({}), {})


class TestFoldContextSwitch(unittest.TestCase):
    def test_entries_without_the_removed_switch_are_untouched(self) -> None:
        self.assertEqual(options.fold_context_switch({const.CONF_MAX_TOKENS: 5}),
                         {const.CONF_MAX_TOKENS: 5})

    def test_switch_on_only_drops_the_key(self) -> None:
        self.assertEqual(
            options.fold_context_switch(
                {const.CONF_CONTEXT_MANAGEMENT_ENABLED: True}
            ),
            {},
        )

    def test_switch_off_becomes_two_explicit_zeros(self) -> None:
        self.assertEqual(
            options.fold_context_switch(
                {const.CONF_CONTEXT_MANAGEMENT_ENABLED: False}
            ),
            {
                const.CONF_MAX_TOOL_RESULT_CHARS: 0,
                const.CONF_MAX_HISTORY_ROUNDS: 0,
            },
        )


class TestMoveWebSearchSelection(unittest.TestCase):
    """Web search stops being a globally registered LLM API (#38).

    An agent that had selected it has to keep searching the web without the
    user having to notice that the setting moved.
    """

    LEGACY = f"{const.LEGACY_WEB_SEARCH_API_ID_PREFIX}01JABCDEF"

    def test_an_agent_that_never_selected_it_is_left_alone(self) -> None:
        for stored in ({}, {LLM_API: ["assist"]}, {LLM_API: "assist"}):
            with self.subTest(stored=stored):
                self.assertIsNone(options.move_web_search_selection(stored))

    def test_the_selection_becomes_the_setting(self) -> None:
        moved = options.move_web_search_selection(
            {LLM_API: ["assist", self.LEGACY], const.CONF_CHAT_MODEL: "m"}
        )
        self.assertEqual(
            moved,
            {
                LLM_API: ["assist"],
                const.CONF_WEB_SEARCH: True,
                const.CONF_CHAT_MODEL: "m",
            },
        )

    def test_web_search_alone_leaves_no_empty_selection_behind(self) -> None:
        """An empty list is not what "nothing selected" looks like in a subentry."""
        moved = options.move_web_search_selection({LLM_API: [self.LEGACY]})
        self.assertEqual(moved, {const.CONF_WEB_SEARCH: True})

    def test_a_single_stored_string_is_understood_too(self) -> None:
        """Entries older than version 2 stored one id rather than a list."""
        self.assertEqual(
            options.move_web_search_selection({LLM_API: self.LEGACY}),
            {const.CONF_WEB_SEARCH: True},
        )

    def test_another_entry_s_web_search_api_moves_as_well(self) -> None:
        """The id carried the entry id, so two keys meant two different ids."""
        other = f"{const.LEGACY_WEB_SEARCH_API_ID_PREFIX}01JZZZZZZ"
        self.assertEqual(
            options.move_web_search_selection({LLM_API: [other]}),
            {const.CONF_WEB_SEARCH: True},
        )

    def test_the_original_settings_are_not_written_through(self) -> None:
        stored = {LLM_API: [self.LEGACY]}
        options.move_web_search_selection(stored)
        self.assertEqual(stored, {LLM_API: [self.LEGACY]})


if __name__ == "__main__":
    unittest.main()
