"""DeepSeek's own Search API as a web search provider.

A standalone JSON endpoint, ``POST /v1/tools/search``, on the same host and
with the same bearer key as the chat API - so it needs no second account, and
it works with every model because the model never sees more than an ordinary
function tool. (DeepSeek also runs search server-side, but only through its
Responses and Anthropic-format APIs and not for every model; this integration
speaks chat completions.) Billed per successful search on the account's
balance, separately from tokens.

Two pieces: ``DeepSeekWebClient`` is the transport for ``/v1/tools/*`` and owns
every status code; ``DeepSeekSearchProvider`` is the search on top of it.

Affects / influenced by:
- ``factory.py`` builds both from a config entry and hands over the HTTP
  client; nothing here imports Home Assistant.
- ``selection.py`` offers this provider only on DeepSeek's official base URL.
  The request goes to the host the entry's base URL names, never to a fixed
  one, so the entry's key cannot reach a server its owner did not configure.
- ``parse_search_results`` fixes what of the answer survives; the model sees
  it through ``SearchResult.as_tool_dict``.
- The error messages are read by the model (see ``WebToolError``), so each
  says what to do next.
"""

from __future__ import annotations

from collections.abc import Callable
import re
import time
from typing import Any

from ..const import LOGGER, WEB_SEARCH_PROVIDER_DEEPSEEK
from .types import SearchResult, WebErrorKind, WebToolError

SEARCH_PATH = "/v1/tools/search"
#: The API refuses a longer query with HTTP 400.
MAX_QUERY_CHARS = 128
#: How much of an API error message is worth repeating to the model.
_ERROR_PREVIEW_CHARS = 200
_DAY_RE = re.compile(r"\d{4}-\d{2}-\d{2}")


def tools_endpoint(base_url: str, path: str) -> str:
    """The URL of a ``/v1/tools`` endpoint on the host ``base_url`` names.

    The chat API is configured with or without a trailing ``/v1``; the tools
    endpoints always carry it, so it is cut off before ``path`` is appended.
    """
    root = base_url.strip().rstrip("/")
    if root.lower().endswith("/v1"):
        root = root[:-3].rstrip("/")
    return f"{root}{path}"


def _api_error_message(payload: Any, text: str) -> str:
    """What the API said went wrong, shortened.

    Errors come in the OpenAI shape, ``{"error": {"message": ...}}``. Anything
    else - a proxy's HTML page, say - is quoted from its raw start instead.
    """
    error = payload.get("error") if isinstance(payload, dict) else None
    message = error.get("message") if isinstance(error, dict) else None
    return str(message or text or "no details")[:_ERROR_PREVIEW_CHARS]


class DeepSeekWebClient:
    """POST to DeepSeek's ``/v1/tools`` endpoints with the entry's own key."""

    def __init__(
        self,
        http: Callable[[], Any],
        *,
        api_key: str,
        base_url: str,
        timeout: float,
    ) -> None:
        """Bind the credentials.

        ``http`` returns the httpx client to use and is only called when a
        request actually runs, so building the client costs nothing.
        """
        self._http = http
        self._api_key = api_key
        self._base_url = base_url
        self._timeout = timeout

    async def post(self, path: str, body: dict[str, Any], *, what: str) -> Any:
        """Send one request and return the decoded answer.

        ``what`` names the service in messages ("DeepSeek Search"). There are
        deliberately no retries: this runs inside a voice turn, and the model
        can call again if the message says that is worth doing.
        """
        url = tools_endpoint(self._base_url, path)
        started = time.monotonic()
        try:
            response = await self._http().post(
                url,
                json=body,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Accept": "application/json",
                },
                timeout=self._timeout,
            )
        except Exception as err:
            reason = str(err) or type(err).__name__
            LOGGER.debug("[Debug web_deepseek]: POST %s failed: %s", url, reason)
            raise WebToolError(
                WebErrorKind.UNAVAILABLE,
                f"{what} could not be reached: {reason}. Try again in a moment.",
            ) from err

        status = response.status_code
        LOGGER.debug(
            "[Debug web_deepseek]: POST %s status=%s in %.2fs",
            url,
            status,
            time.monotonic() - started,
        )

        try:
            payload = response.json()
        except Exception as err:
            if status < 400:
                raise WebToolError(
                    WebErrorKind.BAD_RESPONSE, f"{what} returned invalid JSON: {err}"
                ) from err
            payload = None

        if status >= 400:
            detail = _api_error_message(payload, getattr(response, "text", "") or "")
            LOGGER.debug("[Debug web_deepseek]: HTTP %s detail=%r", status, detail)
            raise self._status_error(status, detail, what)
        return payload

    @staticmethod
    def _status_error(status: int, detail: str, what: str) -> WebToolError:
        """Turn a documented error status into a message the model can act on."""
        if status == 401:
            return WebToolError(
                WebErrorKind.AUTH,
                f"{what} rejected the DeepSeek API key. Tell the user to check it.",
            )
        if status == 402:
            return WebToolError(
                WebErrorKind.BALANCE,
                f"The DeepSeek account has no balance left, so {what} is "
                "unavailable until it is topped up. Tell the user.",
            )
        if status == 429:
            return WebToolError(
                WebErrorKind.RATE_LIMITED,
                f"{what} is rate limiting this account. Try again in a moment.",
            )
        if status == 408 or status >= 500:
            return WebToolError(
                WebErrorKind.UNAVAILABLE,
                f"{what} is temporarily unavailable (HTTP {status}). "
                "Try again in a moment.",
            )
        return WebToolError(
            WebErrorKind.INVALID_REQUEST,
            f"{what} rejected the request (HTTP {status}): {detail}",
        )


def _day(value: Any) -> str | None:
    """The calendar day of an RFC 3339 timestamp, or None.

    The model judges freshness by the day; the time and offset would only cost
    tokens on every result.
    """
    if isinstance(value, str) and (match := _DAY_RE.match(value)):
        return match.group()
    return None


def parse_search_results(payload: Any, count: int) -> list[SearchResult]:
    """Pick the results out of a Search API answer.

    ``summary`` is the field DeepSeek writes for a model to read (600 to 800
    characters); ``snippet`` is its short UI counterpart and only stands in
    when a page has no summary. Ranking score and authority are kept for the
    debug log and never sent to the model.
    """
    raw_results = payload.get("results") if isinstance(payload, dict) else None
    if not isinstance(raw_results, list):
        raise WebToolError(
            WebErrorKind.BAD_RESPONSE,
            "DeepSeek Search answered without a list of results",
        )

    results: list[SearchResult] = []
    for item in raw_results[:count]:
        if not isinstance(item, dict):
            continue
        title = item.get("title")
        url = item.get("url")
        if not title and not url:
            continue
        score = item.get("rerank_score")
        authority = item.get("authority")
        results.append(
            SearchResult(
                title=str(title or ""),
                url=str(url or ""),
                summary=str(item.get("summary") or item.get("snippet") or ""),
                source=str(item["site_name"]) if item.get("site_name") else None,
                date=_day(item.get("date")),
                score=float(score)
                if isinstance(score, (int, float)) and not isinstance(score, bool)
                else None,
                # -1 is the API's own "unknown".
                authority=authority
                if isinstance(authority, int)
                and not isinstance(authority, bool)
                and authority >= 0
                else None,
            )
        )
    return results


class DeepSeekSearchProvider:
    """Search the web through DeepSeek's Search API."""

    id = WEB_SEARCH_PROVIDER_DEEPSEEK
    label = "DeepSeek Search"
    max_query_chars = MAX_QUERY_CHARS
    #: Static on purpose: it is part of the tool definition sent with every
    #: request, and a text that changed would defeat DeepSeek's prefix cache.
    query_help = (
        f"Query: at most {MAX_QUERY_CHARS} characters, key terms rather than a "
        "sentence. Operators: site:example.com, -site:example.com, "
        '"exact phrase", -word, lang:de, after:2026-01-31, before:2026-12-31.'
    )

    def __init__(self, client: DeepSeekWebClient) -> None:
        """Bind the transport."""
        self._client = client

    async def search(self, query: str, count: int) -> list[SearchResult]:
        """Run one search.

        Only ``query`` and ``count`` are sent. ``need_content`` stays off: full
        pages would multiply the size of every result.
        """
        payload = await self._client.post(
            SEARCH_PATH, {"query": query, "count": count}, what=self.label
        )
        return parse_search_results(payload, count)
