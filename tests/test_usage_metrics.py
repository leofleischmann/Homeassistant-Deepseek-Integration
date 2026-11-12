"""Tests for the usage tracker's own bookkeeping.

The tracker drives sensor entities but only ever calls three methods on them,
so the stand-ins below are all it needs - no Home Assistant. What a completion
adds to the token counters is covered where the stream is (``transform_stream``
collects the usage); this file is about the counters that have no other home.
"""

from __future__ import annotations

import unittest

from integration_modules import load

usage_metrics = load("usage_metrics")


class _Counter:
    """A cumulative sensor, as far as the tracker is concerned."""

    def __init__(self) -> None:
        self.value = 0

    def increment(self, amount: int) -> None:
        self.value += amount

    def reset_to_zero(self) -> None:
        self.value = 0


class _Snapshot(_Counter):
    """A last-request sensor: also takes a value outright."""

    def set_value(self, value: int) -> None:
        self.value = value

    def set_usage(self, usage, *, source: str, request_count: int) -> None:
        self.value = request_count


def _bound_tracker() -> tuple[object, dict[str, _Counter]]:
    counters = {
        name: _Counter()
        for name in (
            "prompt",
            "completion",
            "total",
            "reasoning",
            "cache_hit",
            "api_requests",
            "web_searches",
        )
    }
    tracker = usage_metrics.UsageTracker()
    tracker.bind_sensors(
        **counters,
        last_request=_Snapshot(),
        last_request_prompt=_Snapshot(),
        last_request_completion=_Snapshot(),
    )
    return tracker, counters


class TestWebSearchCounter(unittest.TestCase):
    """Web searches are billed apart from tokens, so they are counted apart."""

    def test_each_recorded_search_adds_one(self) -> None:
        tracker, counters = _bound_tracker()
        tracker.record_web_search("deepseek")
        tracker.record_web_search("brave")
        self.assertEqual(counters["web_searches"].value, 2)

    def test_a_search_is_not_an_api_request(self) -> None:
        """``api_requests`` counts chat completions, which is what tokens bill by."""
        tracker, counters = _bound_tracker()
        tracker.record_web_search("deepseek")
        self.assertEqual(counters["api_requests"].value, 0)
        self.assertEqual(tracker.request_count, 0)

    def test_reset_clears_it_with_everything_else(self) -> None:
        tracker, counters = _bound_tracker()
        tracker.record_web_search("deepseek")
        tracker.reset_all()
        self.assertEqual(counters["web_searches"].value, 0)

    def test_a_search_before_the_sensors_exist_does_not_fail_the_turn(self) -> None:
        """An automation can run a task while the entry is still starting."""
        usage_metrics.UsageTracker().record_web_search("deepseek")


if __name__ == "__main__":
    unittest.main()
