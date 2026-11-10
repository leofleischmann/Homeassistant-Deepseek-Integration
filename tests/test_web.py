"""Tests for the web package's core: providers and the rules around them.

Everything here runs without Home Assistant. That is the point of how the
package is cut: a provider is handed its HTTP client, so a test hands it one
that answers from memory, and which tools an agent gets is worked out from two
plain mappings. The tool the model actually calls needs core's ``llm`` helper
and is covered in ``test_real_home_assistant.py``.
"""

from __future__ import annotations

import asyncio
import unittest

from integration_modules import load

const = load("const")
brave = load("web.brave")
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


class TestWhichWebToolsAnAgentGets(unittest.TestCase):
    """An agent gets what it asked for, limited to what its entry can provide."""

    KEY = {const.CONF_BRAVE_API_KEY: "brave-key"}
    ON = {const.CONF_WEB_SEARCH: True}

    def test_an_entry_offers_brave_once_it_holds_a_key(self) -> None:
        self.assertEqual(selection.available_search_providers({}), ())
        self.assertEqual(
            selection.available_search_providers({const.CONF_BRAVE_API_KEY: "  "}), ()
        )
        self.assertEqual(
            selection.available_search_providers(self.KEY),
            (const.WEB_SEARCH_PROVIDER_BRAVE,),
        )

    def test_asked_for_and_available_is_a_plan(self) -> None:
        plan = selection.web_tools_plan(self.KEY, self.ON)
        self.assertTrue(plan)
        self.assertEqual(plan.search_provider, const.WEB_SEARCH_PROVIDER_BRAVE)

    def test_not_asked_for_is_no_plan(self) -> None:
        self.assertFalse(selection.web_tools_plan(self.KEY, {}))

    def test_a_setting_that_outlived_its_key_is_not_honoured(self) -> None:
        """The key can be removed through Reconfigure while the setting stays."""
        plan = selection.web_tools_plan({}, self.ON)
        self.assertFalse(plan)
        self.assertIsNone(plan.search_provider)


if __name__ == "__main__":
    unittest.main()
