"""Tests for the web package's core: providers and the rules around them.

Everything here runs without Home Assistant. That is the point of how the
package is cut: a provider is handed its HTTP client, so a test hands it one
that answers from memory, and which tools an agent gets is worked out from two
plain mappings. The tool the model actually calls needs core's ``llm`` helper
and is covered in ``test_real_home_assistant.py``.
"""

from __future__ import annotations

import asyncio
import json
import unittest

from integration_modules import load

const = load("const")
brave = load("web.brave")
deepseek = load("web.deepseek")
selection = load("web.selection")
web_types = load("web.types")


class _Response:
    """Just enough of an httpx response for a provider to read."""

    def __init__(self, status_code: int = 200, payload=None, text: str = "") -> None:
        self.status_code = status_code
        self.text = text
        self._payload = payload

    def json(self):
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


class _Client:
    """An HTTP client that answers from memory and records every request."""

    def __init__(self, response) -> None:
        self._response = response
        self.requests: list[tuple[str, str, dict]] = []

    async def _answer(self, method: str, url: str, **kwargs):
        self.requests.append((method, url, kwargs))
        if isinstance(self._response, Exception):
            raise self._response
        return self._response

    async def get(self, url: str, **kwargs):
        return await self._answer("GET", url, **kwargs)

    async def post(self, url: str, **kwargs):
        return await self._answer("POST", url, **kwargs)


def _failure(coroutine) -> web_types.WebToolError:
    """Run a call that must fail, and hand back how it failed."""
    try:
        asyncio.run(coroutine)
    except web_types.WebToolError as err:
        return err
    raise AssertionError("expected a WebToolError")


class TestBraveResults(unittest.TestCase):
    """What survives of Brave's answer is decided in one function."""

    def test_title_url_and_description_are_kept(self) -> None:
        results = brave.parse_brave_results(
            {"web": {"results": [{"title": "T", "url": "https://e.test", "description": "D"}]}},
            5,
        )
        self.assertEqual(
            results,
            [web_types.SearchResult(title="T", url="https://e.test", summary="D")],
        )

    def test_a_result_with_nothing_to_cite_is_dropped(self) -> None:
        results = brave.parse_brave_results(
            {"web": {"results": [{"title": None, "url": None, "description": "D"}, "junk"]}},
            5,
        )
        self.assertEqual(results, [])

    def test_no_more_than_asked_for(self) -> None:
        items = [{"title": str(n), "url": f"https://e.test/{n}"} for n in range(6)]
        results = brave.parse_brave_results({"web": {"results": items}}, 2)
        self.assertEqual([result.title for result in results], ["0", "1"])

    def test_an_answer_of_the_wrong_shape_is_no_results(self) -> None:
        for payload in (None, [], {}, {"web": None}, {"web": {"results": "x"}}):
            with self.subTest(payload=payload):
                self.assertEqual(brave.parse_brave_results(payload, 5), [])


class TestBraveProvider(unittest.TestCase):
    """The request Brave gets, and what each way of failing turns into."""

    def _provider(self, response) -> tuple[object, _Client]:
        client = _Client(response)
        provider = brave.BraveSearchProvider(
            lambda: client, api_key="brave-key", timeout=10.0
        )
        return provider, client

    def test_the_request_carries_the_key_and_the_timeout(self) -> None:
        provider, client = self._provider(_Response(payload={"web": {"results": []}}))
        self.assertEqual(asyncio.run(provider.search("ha release", 3)), [])
        method, url, kwargs = client.requests[0]
        self.assertEqual((method, url), ("GET", brave.BRAVE_WEB_SEARCH_URL))
        self.assertEqual(kwargs["params"], {"q": "ha release", "count": 3})
        self.assertEqual(kwargs["headers"]["X-Subscription-Token"], "brave-key")
        self.assertEqual(kwargs["timeout"], 10.0)

    def test_the_client_is_only_asked_for_when_a_search_runs(self) -> None:
        """Building a provider per turn must cost nothing."""
        asked: list[int] = []
        brave.BraveSearchProvider(
            lambda: asked.append(1), api_key="brave-key", timeout=10.0
        )
        self.assertEqual(asked, [])

    def test_each_failure_says_what_kind_it_is(self) -> None:
        kinds = web_types.WebErrorKind
        cases = (
            (_Response(401), kinds.AUTH),
            (_Response(403), kinds.AUTH),
            (_Response(422, text="bad q"), kinds.INVALID_REQUEST),
            (_Response(429), kinds.RATE_LIMITED),
            (_Response(503), kinds.UNAVAILABLE),
            (_Response(payload=ValueError("not json")), kinds.BAD_RESPONSE),
            (OSError("connection reset"), kinds.UNAVAILABLE),
        )
        for response, kind in cases:
            with self.subTest(kind=kind):
                provider, _client = self._provider(response)
                self.assertEqual(_failure(provider.search("q", 1)).kind, kind)

    def test_an_error_body_is_quoted_but_only_its_start(self) -> None:
        provider, _client = self._provider(_Response(500, text="x" * 500))
        message = str(_failure(provider.search("q", 1)))
        self.assertIn("HTTP 500", message)
        self.assertLess(len(message), 260)


class TestSearchResultAsTheModelSeesIt(unittest.TestCase):
    """Every key is repeated per result, so only what is known is sent."""

    def test_a_full_result_keeps_its_source_and_date(self) -> None:
        result = web_types.SearchResult(
            title="T", url="https://e.test", summary="S", source="Example", date="2026-06-01"
        )
        self.assertEqual(
            result.as_tool_dict(),
            {
                "title": "T",
                "url": "https://e.test",
                "source": "Example",
                "date": "2026-06-01",
                "summary": "S",
            },
        )

    def test_unknown_fields_are_left_out_not_sent_as_null(self) -> None:
        result = web_types.SearchResult(title="T", url="https://e.test", summary="S")
        self.assertEqual(
            result.as_tool_dict(), {"title": "T", "url": "https://e.test", "summary": "S"}
        )

    def test_ranking_signals_never_reach_the_model(self) -> None:
        result = web_types.SearchResult(
            title="T", url="https://e.test", summary="S", score=2.4, authority=3
        )
        self.assertNotIn("score", result.as_tool_dict())
        self.assertNotIn("authority", result.as_tool_dict())


def _result(summary: str, n: int = 0):
    return web_types.SearchResult(
        title=f"Title {n}", url=f"https://e.test/{n}", summary=summary, source="Example"
    )


def _size(payload: dict) -> int:
    """What the tool-result limit counts: json.dumps with its defaults."""
    return len(json.dumps(payload))


class TestToolPayloadFitsTheLimit(unittest.TestCase):
    """A search answer shares the agent's limit out instead of losing its end.

    The live case behind it: DeepSeek sent summaries of up to 3,700 characters
    where its docs say 600 to 800, so eight results came to 28,000 characters
    and the general 12,000 cut dropped three of them.
    """

    def _long(self, n: int) -> str:
        return " ".join(f"Sentence {n}.{i} says something." for i in range(150))

    def test_no_limit_changes_nothing(self) -> None:
        results = [_result(self._long(n), n) for n in range(8)]
        payload = web_types.tool_payload("q", results, max_chars=0)
        self.assertEqual(
            [item["summary"] for item in payload["results"]],
            [result.summary for result in results],
        )

    def test_an_answer_within_the_limit_stays_whole(self) -> None:
        results = [_result("short", n) for n in range(3)]
        payload = web_types.tool_payload("q", results, max_chars=12_000)
        self.assertEqual([item["summary"] for item in payload["results"]], ["short"] * 3)

    def test_every_result_arrives_and_the_whole_fits(self) -> None:
        results = [_result(self._long(n), n) for n in range(8)]
        payload = web_types.tool_payload("q", results, max_chars=12_000)
        self.assertEqual(len(payload["results"]), 8)
        self.assertLessEqual(_size(payload), 12_000)
        for item in payload["results"]:
            self.assertTrue(item["summary"].endswith("…"))
            self.assertGreater(len(item["summary"]), 1_000)

    def test_short_summaries_keep_everything_and_leave_room_for_the_rest(self) -> None:
        results = [_result("brief", 0), _result(self._long(1), 1), _result(self._long(2), 2)]
        payload = web_types.tool_payload("q", results, max_chars=4_000)
        self.assertEqual(payload["results"][0]["summary"], "brief")
        self.assertLessEqual(_size(payload), 4_000)
        # The two long ones split what the short one left, near evenly.
        long_ones = [len(item["summary"]) for item in payload["results"][1:]]
        self.assertLess(abs(long_ones[0] - long_ones[1]), 60)

    def test_cuts_land_on_a_sentence_end(self) -> None:
        payload = web_types.tool_payload("q", [_result(self._long(0))], max_chars=1_000)
        self.assertTrue(payload["results"][0]["summary"].endswith("something.…"))

    def test_escaped_characters_are_counted_as_sent(self) -> None:
        """An umlaut is six characters once serialized; the limit counts those."""
        results = [_result("Größenänderung über Nacht. " * 200, n) for n in range(4)]
        payload = web_types.tool_payload("q", results, max_chars=5_000)
        self.assertLessEqual(_size(payload), 5_000)
        self.assertEqual(len(payload["results"]), 4)

    def test_a_limit_too_small_for_the_titles_is_left_to_the_general_cut(self) -> None:
        results = [_result(self._long(n), n) for n in range(10)]
        payload = web_types.tool_payload("q", results, max_chars=500)
        self.assertEqual(
            [item["summary"] for item in payload["results"]],
            [result.summary for result in results],
        )


class TestDeepSeekEndpoint(unittest.TestCase):
    """The tools endpoints hang off the host the entry's base URL names."""

    def test_with_or_without_v1_the_url_is_the_documented_one(self) -> None:
        for base_url in (
            "https://api.deepseek.com/v1",
            "https://api.deepseek.com/v1/",
            "https://api.deepseek.com",
            "https://api.deepseek.com/",
            " https://api.deepseek.com/V1 ",
        ):
            with self.subTest(base_url=base_url):
                self.assertEqual(
                    deepseek.tools_endpoint(base_url, deepseek.SEARCH_PATH),
                    "https://api.deepseek.com/v1/tools/search",
                )

    def test_the_host_is_never_replaced_by_a_fixed_one(self) -> None:
        """The entry's key must only ever go where its owner pointed it."""
        self.assertEqual(
            deepseek.tools_endpoint("https://gateway.example/ds/v1", deepseek.SEARCH_PATH),
            "https://gateway.example/ds/v1/tools/search",
        )


#: One result exactly as the Search API documents it.
_DOCUMENTED_RESULT = {
    "rerank_score": 2.4,
    "title": "DeepSeek V4 model released",
    "url": "https://example.com/deepseek-v4",
    "snippet": "Short.",
    "summary": "A long summary written for a model.",
    "content": "",
    "site_name": "Example News",
    "site_icon": "https://example.com/favicon.ico",
    "date": "2026-06-01T00:00:00+00:00",
    "authority": 3,
}


class TestDeepSeekResults(unittest.TestCase):
    """What survives of a Search API answer is decided in one function."""

    def test_a_documented_result_maps_field_by_field(self) -> None:
        (result,) = deepseek.parse_search_results({"results": [_DOCUMENTED_RESULT]}, 5)
        self.assertEqual(
            result,
            web_types.SearchResult(
                title="DeepSeek V4 model released",
                url="https://example.com/deepseek-v4",
                summary="A long summary written for a model.",
                source="Example News",
                date="2026-06-01",
                score=2.4,
                authority=3,
            ),
        )

    def test_the_short_snippet_stands_in_when_there_is_no_summary(self) -> None:
        (result,) = deepseek.parse_search_results(
            {"results": [{**_DOCUMENTED_RESULT, "summary": ""}]}, 5
        )
        self.assertEqual(result.summary, "Short.")

    def test_what_the_api_does_not_know_stays_unknown(self) -> None:
        (result,) = deepseek.parse_search_results(
            {
                "results": [
                    {**_DOCUMENTED_RESULT, "site_name": None, "date": None, "authority": -1}
                ]
            },
            5,
        )
        self.assertIsNone(result.source)
        self.assertIsNone(result.date)
        self.assertIsNone(result.authority)

    def test_a_date_that_is_not_a_date_is_dropped(self) -> None:
        (result,) = deepseek.parse_search_results(
            {"results": [{**_DOCUMENTED_RESULT, "date": "yesterday"}]}, 5
        )
        self.assertIsNone(result.date)

    def test_a_result_with_nothing_to_cite_is_dropped(self) -> None:
        results = deepseek.parse_search_results(
            {"results": [{"title": "", "url": None, "summary": "S"}, "junk"]}, 5
        )
        self.assertEqual(results, [])

    def test_no_more_than_asked_for(self) -> None:
        items = [{**_DOCUMENTED_RESULT, "title": str(n)} for n in range(6)]
        results = deepseek.parse_search_results({"results": items}, 2)
        self.assertEqual([result.title for result in results], ["0", "1"])

    def test_no_matches_is_an_empty_list(self) -> None:
        self.assertEqual(deepseek.parse_search_results({"results": []}, 5), [])

    def test_an_answer_without_a_results_list_is_an_error_not_no_results(self) -> None:
        """An empty search and a broken answer must not look the same."""
        for payload in (None, [], {}, {"results": None}, {"results": "x"}):
            with self.subTest(payload=payload):
                with self.assertRaises(web_types.WebToolError) as caught:
                    deepseek.parse_search_results(payload, 5)
                self.assertEqual(
                    caught.exception.kind, web_types.WebErrorKind.BAD_RESPONSE
                )


class TestDeepSeekProvider(unittest.TestCase):
    """The request DeepSeek gets, and what each documented failure turns into."""

    def _provider(self, response, *, base_url="https://api.deepseek.com/v1"):
        client = _Client(response)
        provider = deepseek.DeepSeekSearchProvider(
            deepseek.DeepSeekWebClient(
                lambda: client, api_key="sk-deepseek", base_url=base_url, timeout=10.0
            )
        )
        return provider, client

    def test_the_request_is_the_documented_one(self) -> None:
        provider, client = self._provider(
            _Response(payload={"results": [_DOCUMENTED_RESULT]})
        )
        results = asyncio.run(provider.search("ha release", 3))
        self.assertEqual(len(results), 1)
        method, url, kwargs = client.requests[0]
        self.assertEqual(
            (method, url), ("POST", "https://api.deepseek.com/v1/tools/search")
        )
        # Nothing but the query and the count leaves the house.
        self.assertEqual(kwargs["json"], {"query": "ha release", "count": 3})
        self.assertEqual(kwargs["headers"]["Authorization"], "Bearer sk-deepseek")
        self.assertEqual(kwargs["timeout"], 10.0)

    def test_the_client_is_only_asked_for_when_a_search_runs(self) -> None:
        asked: list[int] = []
        deepseek.DeepSeekSearchProvider(
            deepseek.DeepSeekWebClient(
                lambda: asked.append(1),
                api_key="sk",
                base_url="https://api.deepseek.com/v1",
                timeout=10.0,
            )
        )
        self.assertEqual(asked, [])

    def test_the_provider_states_its_query_limit_and_syntax(self) -> None:
        provider, _client = self._provider(_Response(payload={"results": []}))
        self.assertEqual(provider.max_query_chars, 128)
        self.assertIn("128", provider.query_help)
        self.assertIn("site:", provider.query_help)
        self.assertIn("after:", provider.query_help)

    def test_each_documented_failure_says_what_kind_it_is(self) -> None:
        kinds = web_types.WebErrorKind
        error = {"error": {"message": "'count' must be between 1 and 50"}}
        cases = (
            (_Response(400, payload=error), kinds.INVALID_REQUEST),
            (_Response(401, payload=error), kinds.AUTH),
            (_Response(402, payload=error), kinds.BALANCE),
            (_Response(408, payload=error), kinds.UNAVAILABLE),
            (_Response(413, payload=error), kinds.INVALID_REQUEST),
            (_Response(429, payload=error), kinds.RATE_LIMITED),
            (_Response(500, payload=error), kinds.UNAVAILABLE),
            (_Response(504, payload=ValueError("html"), text="<html>"), kinds.UNAVAILABLE),
            (_Response(200, payload=ValueError("not json")), kinds.BAD_RESPONSE),
            (_Response(200, payload={"data": []}), kinds.BAD_RESPONSE),
            (TimeoutError(), kinds.UNAVAILABLE),
        )
        for response, kind in cases:
            with self.subTest(kind=kind, response=getattr(response, "status_code", response)):
                provider, _client = self._provider(response)
                self.assertEqual(_failure(provider.search("q", 1)).kind, kind)

    def test_a_rejected_request_repeats_what_the_api_said(self) -> None:
        """The model can only fix the call if it is told what was wrong with it."""
        provider, _client = self._provider(
            _Response(400, payload={"error": {"message": "'count' must be between 1 and 50"}})
        )
        self.assertIn("'count' must be between 1 and 50", str(_failure(provider.search("q", 1))))

    def test_an_empty_balance_is_something_to_tell_the_user(self) -> None:
        provider, _client = self._provider(_Response(402, payload={}))
        message = str(_failure(provider.search("q", 1)))
        self.assertIn("balance", message)
        self.assertIn("Tell the user", message)

    def test_an_error_page_is_quoted_but_only_its_start(self) -> None:
        provider, _client = self._provider(
            _Response(418, payload=ValueError("html"), text="x" * 500)
        )
        self.assertLess(len(str(_failure(provider.search("q", 1)))), 300)

    def test_a_timeout_without_a_message_still_says_what_happened(self) -> None:
        provider, _client = self._provider(TimeoutError())
        self.assertIn("TimeoutError", str(_failure(provider.search("q", 1))))


class TestWhichWebToolsAnAgentGets(unittest.TestCase):
    """An agent gets the provider it named, if its entry can offer that one."""

    DEEPSEEK = const.WEB_SEARCH_PROVIDER_DEEPSEEK
    BRAVE = const.WEB_SEARCH_PROVIDER_BRAVE
    OFFICIAL = {const.CONF_BASE_URL: "https://api.deepseek.com/v1"}
    GATEWAY = {const.CONF_BASE_URL: "https://gateway.example/v1"}
    KEY = {const.CONF_BRAVE_API_KEY: "brave-key"}

    def test_what_an_entry_offers_follows_its_base_url_and_its_brave_key(self) -> None:
        cases = (
            (self.OFFICIAL, (self.DEEPSEEK,)),
            ({**self.OFFICIAL, **self.KEY}, (self.DEEPSEEK, self.BRAVE)),
            ({**self.GATEWAY, **self.KEY}, (self.BRAVE,)),
            (self.GATEWAY, ()),
            # An entry that never stored a base URL is on the official API.
            ({}, (self.DEEPSEEK,)),
            ({**self.GATEWAY, const.CONF_BRAVE_API_KEY: "  "}, ()),
        )
        for entry_data, offered in cases:
            with self.subTest(entry_data=entry_data):
                self.assertEqual(
                    selection.available_search_providers(entry_data), offered
                )

    def test_named_and_available_is_a_plan(self) -> None:
        both = {**self.OFFICIAL, **self.KEY}
        for provider in (self.DEEPSEEK, self.BRAVE):
            with self.subTest(provider=provider):
                plan = selection.web_tools_plan(both, {const.CONF_WEB_SEARCH: provider})
                self.assertTrue(plan)
                self.assertEqual(plan.search_provider, provider)

    def test_no_choice_is_no_plan(self) -> None:
        self.assertFalse(selection.web_tools_plan({**self.OFFICIAL, **self.KEY}, {}))

    def test_a_choice_the_entry_cannot_offer_is_not_honoured(self) -> None:
        """Nor replaced by the other provider: gone means no web search."""
        cases = (
            (self.OFFICIAL, self.BRAVE),  # the Brave key was removed
            ({**self.GATEWAY, **self.KEY}, self.DEEPSEEK),  # moved to a gateway
        )
        for entry_data, chosen in cases:
            with self.subTest(chosen=chosen):
                plan = selection.web_tools_plan(
                    entry_data, {const.CONF_WEB_SEARCH: chosen}
                )
                self.assertFalse(plan)
                self.assertIsNone(plan.search_provider)

    def test_only_a_provider_id_counts_as_a_choice(self) -> None:
        """No guessing: an unknown id, the form's "off" and the old switch are all nothing."""
        for stored in ("searxng", const.WEB_SEARCH_OFF, True, "", None):
            with self.subTest(stored=stored):
                self.assertFalse(
                    selection.web_tools_plan(
                        {**self.OFFICIAL, **self.KEY}, {const.CONF_WEB_SEARCH: stored}
                    )
                )


if __name__ == "__main__":
    unittest.main()
