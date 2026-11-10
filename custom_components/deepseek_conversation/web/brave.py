"""Brave Search as a web search provider.

Needs its own key, kept in ``entry.data`` (setup / reconfigure in
``config_flow.py``). The only provider that works behind a custom base URL,
which is why it stays even where another one needs no extra account.

Affects / influenced by:
- ``factory.py`` builds this from a config entry and hands it the HTTP client;
  nothing here imports Home Assistant.
- ``selection.py`` decides whether an entry can offer it (a key is set).
- ``parse_brave_results`` fixes what of Brave's answer survives; the model sees
  it through ``SearchResult.as_tool_dict``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ..const import LOGGER, WEB_SEARCH_PROVIDER_BRAVE
from .types import SearchResult, WebErrorKind, WebToolError

BRAVE_WEB_SEARCH_URL = "https://api.search.brave.com/res/v1/web/search"
#: How much of an error body is worth repeating in a message.
_BODY_PREVIEW_CHARS = 200


def parse_brave_results(payload: Any, count: int) -> list[SearchResult]:
    """Pick the web results out of Brave's answer.

    Defensive about every level, because this runs on whatever came back: a
    result that carries neither a title nor a URL gives the model nothing to
    cite and is dropped.
    """
    web = payload.get("web") if isinstance(payload, dict) else None
    raw_results = web.get("results") if isinstance(web, dict) else None
    if not isinstance(raw_results, list):
        return []
    results: list[SearchResult] = []
    for item in raw_results[:count]:
        if not isinstance(item, dict):
            continue
        title = item.get("title")
        url = item.get("url")
        if not title and not url:
            continue
        results.append(
            SearchResult(
                title=str(title or ""),
                url=str(url or ""),
                summary=str(item.get("description") or ""),
            )
        )
    return results


class BraveSearchProvider:
    """Search the web through the Brave Search API."""

    id = WEB_SEARCH_PROVIDER_BRAVE
    label = "Brave Search"

    def __init__(
        self, http: Callable[[], Any], *, api_key: str, timeout: float
    ) -> None:
        """Bind the subscription token.

        ``http`` returns the httpx client to use and is only called when a
        search actually runs, so building the provider costs nothing.
        """
        self._http = http
        self._api_key = api_key
        self._timeout = timeout

    async def search(self, query: str, count: int) -> list[SearchResult]:
        """Run one search and return title, URL and description per hit."""
        try:
            response = await self._http().get(
                BRAVE_WEB_SEARCH_URL,
                params={"q": query, "count": count},
                headers={
                    "Accept": "application/json",
                    "Accept-Encoding": "gzip",
                    "X-Subscription-Token": self._api_key,
                },
                timeout=self._timeout,
            )
        except Exception as err:
            LOGGER.debug("[Debug web_brave]: request failed: %s", err)
            raise WebToolError(
                WebErrorKind.UNAVAILABLE, f"Brave Search request failed: {err}"
            ) from err

        status = response.status_code
        if status in (401, 403):
            LOGGER.debug("[Debug web_brave]: auth rejected status=%s", status)
            raise WebToolError(
                WebErrorKind.AUTH, "Brave Search rejected the API key (unauthorized)"
            )
        if status >= 400:
            body_preview = (response.text or "")[:_BODY_PREVIEW_CHARS]
            LOGGER.debug("[Debug web_brave]: HTTP %s body=%r", status, body_preview)
            if status == 429:
                kind = WebErrorKind.RATE_LIMITED
            elif status >= 500:
                kind = WebErrorKind.UNAVAILABLE
            else:
                kind = WebErrorKind.INVALID_REQUEST
            raise WebToolError(
                kind, f"Brave Search returned HTTP {status}: {body_preview}"
            )

        try:
            payload = response.json()
        except Exception as err:
            LOGGER.debug("[Debug web_brave]: invalid JSON: %s", err)
            raise WebToolError(
                WebErrorKind.BAD_RESPONSE, f"Brave Search returned invalid JSON: {err}"
            ) from err

        return parse_brave_results(payload, count)
