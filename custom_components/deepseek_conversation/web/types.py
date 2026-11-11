"""What a web search is, whoever answers it and whoever asks.

The vocabulary the providers and the tool share: a result, a failure, and the
protocol a provider has to speak. Keeping it free of Home Assistant and of any
one search API is what lets a provider be added without the tool changing, and
lets the tool be tested against a provider that is five lines long.

Affects / influenced by:
- ``brave.py`` and ``deepseek.py`` produce these; ``search_tool.py`` consumes
  them and turns a ``WebToolError`` into the ``HomeAssistantError`` core
  reports to the model.
- A field added to ``SearchResult`` reaches the model only through
  ``SearchResult.as_tool_dict``, so that method is the model-facing contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol


class WebErrorKind(StrEnum):
    """Why a web request produced nothing usable.

    For the log and for tests, which should not have to match on message text.
    The model only ever sees the message.
    """

    #: The provider refused the request as sent; repeating it will not help.
    INVALID_REQUEST = "invalid_request"
    #: The provider rejected the credential.
    AUTH = "auth"
    #: The account behind the credential has run out of balance.
    BALANCE = "balance"
    #: The provider is throttling this account.
    RATE_LIMITED = "rate_limited"
    #: The provider could not be reached or failed on its side.
    UNAVAILABLE = "unavailable"
    #: The provider answered with something that is not the documented shape.
    BAD_RESPONSE = "bad_response"


class WebToolError(Exception):
    """A web request that failed, with a message written for the model.

    Home Assistant hands the text of a failed tool call back to the model as
    the tool's result, so the message is what the model reads to decide whether
    to retry, rephrase or tell the user.
    """

    def __init__(self, kind: WebErrorKind, message: str) -> None:
        """Keep the kind for the log beside the message for the model."""
        super().__init__(message)
        self.kind = kind


@dataclass(frozen=True, slots=True)
class SearchResult:
    """One search hit, reduced to what a model needs to answer from it."""

    title: str
    url: str
    summary: str
    #: Name of the site, so a spoken answer can name its source without
    #: reading out a URL.
    source: str | None = None
    #: Day of publication as ``YYYY-MM-DD``, where the provider knows it.
    date: str | None = None
    #: The provider's own ranking signals. They go to the debug log, where
    #: they help judge result quality, and never to the model.
    score: float | None = None
    authority: int | None = None

    def as_tool_dict(self) -> dict[str, Any]:
        """The result as the model sees it.

        Unknown fields are left out rather than sent as null: every key is
        repeated for every result, and an empty one only costs tokens.
        """
        data: dict[str, Any] = {"title": self.title, "url": self.url}
        if self.source:
            data["source"] = self.source
        if self.date:
            data["date"] = self.date
        data["summary"] = self.summary
        return data


class SearchProvider(Protocol):
    """A search backend, as far as the tool is concerned."""

    #: Stored in an agent's settings, so it must never change once released.
    id: str
    #: Named in messages to the model.
    label: str
    #: Longest query the backend accepts, or ``None`` when it has no limit the
    #: tool should enforce before sending.
    max_query_chars: int | None
    #: How to write a query for this backend, appended to the tool
    #: description. May be empty, and must be static (it is part of every
    #: request's cached prefix).
    query_help: str

    async def search(self, query: str, count: int) -> list[SearchResult]:
        """Return up to ``count`` results, or raise ``WebToolError``."""
        ...
