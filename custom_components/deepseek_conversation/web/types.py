"""What a web search is, whoever answers it and whoever asks.

The vocabulary the providers and the tool share: a result, a failure, and the
protocol a provider has to speak. Keeping it free of Home Assistant and of any
one search API is what lets a provider be added without the tool changing, and
lets the tool be tested against a provider that is five lines long.

Affects / influenced by:
- ``brave.py`` produces these; ``search_tool.py`` consumes them and turns a
  ``WebToolError`` into the ``HomeAssistantError`` core reports to the model.
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

    def as_tool_dict(self) -> dict[str, Any]:
        """The result as the model sees it."""
        return {"title": self.title, "url": self.url, "description": self.summary}


class SearchProvider(Protocol):
    """A search backend, as far as the tool is concerned."""

    #: Stored in an agent's settings, so it must never change once released.
    id: str
    #: Named in the tool description and in messages.
    label: str

    async def search(self, query: str, count: int) -> list[SearchResult]:
        """Return up to ``count`` results, or raise ``WebToolError``."""
        ...
