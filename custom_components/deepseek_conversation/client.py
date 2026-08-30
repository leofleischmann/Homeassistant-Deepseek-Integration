"""Building and checking the OpenAI-compatible client for one config entry.

Everything about *how* this integration reaches the API: which shared httpx
client to borrow from Home Assistant, what timeout and retry budget a voice
assistant can live with, and whether a set of credentials works at all.

Kept apart from the config flow so entry setup does not have to import it -
the probe is needed by both, and the flow is the wrong place for setup to
reach into.
"""

from __future__ import annotations

from typing import Any

import openai

from homeassistant.config_entries import ConfigEntry  # pyright: ignore[reportMissingImports]
from homeassistant.const import CONF_API_KEY  # pyright: ignore[reportMissingImports]
from homeassistant.core import HomeAssistant  # pyright: ignore[reportMissingImports]
from homeassistant.exceptions import (  # pyright: ignore[reportMissingImports]
    ConfigEntryAuthFailed,
    ConfigEntryNotReady,
)
from homeassistant.helpers.httpx_client import get_async_client  # pyright: ignore[reportMissingImports]

from .const import (
    CONF_BASE_URL,
    DEEPSEEK_API_BASE_URL,
    DEEPSEEK_MAX_RETRIES,
    LOGGER,
)
from .options import request_timeout_from_options
from .types import default_agent_options

#: Short on purpose: this only has to tell a reachable endpoint from an
#: unreachable one, and it runs before Home Assistant considers the entry set up.
_PROBE_TIMEOUT = 10.0


def async_http_client(hass: HomeAssistant) -> Any:
    """Return HA's shared httpx client, preferring the HTTP/2-capable one.

    The OpenAI SDK ends a streamed completion by breaking out of the SSE
    iterator at ``[DONE]`` and closing the response without draining it. Over
    HTTP/1.1 httpx cannot return such a connection to the pool, so **every** API
    round opens a new TCP+TLS connection — a full handshake per tool-calling
    round trip, which is expensive on long-haul or proxied routes. Over HTTP/2
    only the stream is closed and the connection stays pooled.

    Requires ``h2`` (declared in the manifest) and a core new enough for
    ``alpn_protocols``. ALPN negotiates per endpoint, so an API that offers no
    HTTP/2 transparently falls back to HTTP/1.1; if anything here is missing the
    plain shared HTTP/1.1 client is used, which is the previous behaviour.
    """
    try:
        import h2  # noqa: F401  # pyright: ignore[reportMissingImports]

        from homeassistant.util.ssl import SSL_ALPN_HTTP11_HTTP2  # pyright: ignore[reportMissingImports]
    except ImportError as err:
        LOGGER.debug(
            "[Debug setup]: HTTP/2 support unavailable (%s); using HTTP/1.1. "
            "Each API round will open a new TLS connection.",
            err,
        )
        return get_async_client(hass)

    try:
        client = get_async_client(hass, alpn_protocols=SSL_ALPN_HTTP11_HTTP2)
    except TypeError as err:
        # Core predates the alpn_protocols keyword.
        LOGGER.debug(
            "[Debug setup]: shared client does not accept alpn_protocols (%s); "
            "using HTTP/1.1",
            err,
        )
        return get_async_client(hass)

    LOGGER.debug(
        "[Debug setup]: using HTTP/2-capable shared client (falls back to "
        "HTTP/1.1 per endpoint via ALPN)"
    )
    return client


async def async_probe_deepseek_client(client: openai.AsyncOpenAI) -> None:
    """Validate credentials via ``models.list()`` without using completion quota.

    OpenAI-compatible gateways without ``/models`` (404/405/501) are skipped so the
    first real chat call can surface auth issues. Used by the config flow to check
    what the user typed, and by ``async_create_client`` on every entry setup.
    """
    try:
        await client.with_options(timeout=_PROBE_TIMEOUT).models.list()
    except openai.APIStatusError as err:
        if err.status_code in (404, 405, 501):
            LOGGER.debug(
                "DeepSeek base URL does not implement /models (%s); skipping probe",
                err.status_code,
            )
            return
        raise


async def async_validate_credentials(hass: HomeAssistant, data: dict[str, Any]) -> None:
    """Check that an API key and base URL the user typed actually work.

    Used by the config flow, which has no config entry yet - hence the raw
    mapping rather than one. The plain shared httpx client is enough here: this
    is one short request, not a stream, so HTTP/2 buys nothing.
    """
    base_url = data.get(CONF_BASE_URL, DEEPSEEK_API_BASE_URL)
    if base_url:
        base_url = base_url.strip()
    if not base_url:
        base_url = DEEPSEEK_API_BASE_URL

    # The OpenAI client wraps Home Assistant's shared httpx client, which HA owns
    # and closes on shutdown; closing it here would only trigger a framework
    # warning without releasing anything, so the client is left for GC.
    client = openai.AsyncOpenAI(
        api_key=data[CONF_API_KEY],
        base_url=base_url,
        http_client=get_async_client(hass),
    )
    await async_probe_deepseek_client(client)


def async_build_client(hass: HomeAssistant, entry: ConfigEntry) -> openai.AsyncOpenAI:
    """Build the API client one config entry answers through."""
    return openai.AsyncOpenAI(
        api_key=entry.data[CONF_API_KEY],
        base_url=entry.data.get(CONF_BASE_URL, DEEPSEEK_API_BASE_URL),
        http_client=async_http_client(hass),
        # The SDK would default to 600 s and two retries, which lets one
        # unresponsive endpoint block a voice pipeline for ten minutes. Call
        # sites narrow this further per request (see request_timeout_from_options).
        timeout=request_timeout_from_options(default_agent_options(entry)),
        max_retries=DEEPSEEK_MAX_RETRIES,
    )


async def async_create_client(
    hass: HomeAssistant, entry: ConfigEntry
) -> openai.AsyncOpenAI:
    """Build the client and prove the credentials work, or refuse the setup.

    Rejected credentials become ``ConfigEntryAuthFailed`` so Home Assistant
    starts a reauth flow; anything else is ``ConfigEntryNotReady`` so it retries
    on its own.
    """
    client = async_build_client(hass, entry)
    try:
        await async_probe_deepseek_client(client)
    except openai.AuthenticationError as err:
        LOGGER.error("Invalid DeepSeek API key: %s", err)
        raise ConfigEntryAuthFailed("Invalid DeepSeek API key") from err
    except openai.APIStatusError as err:
        if err.status_code in (401, 403):
            LOGGER.error("DeepSeek rejected credentials (%s): %s", err.status_code, err)
            raise ConfigEntryAuthFailed("Invalid DeepSeek credentials") from err
        LOGGER.warning(
            "Unexpected DeepSeek status during setup probe (%s): %s",
            err.status_code,
            err,
        )
        raise ConfigEntryNotReady(
            f"DeepSeek API returned {err.status_code}: {err}"
        ) from err
    except openai.APIConnectionError as err:
        LOGGER.error("Failed to connect to DeepSeek API: %s", err)
        raise ConfigEntryNotReady(
            f"Failed to connect to DeepSeek API: {err}"
        ) from err
    except openai.OpenAIError as err:
        LOGGER.error("DeepSeek SDK error during setup: %s", err)
        raise ConfigEntryNotReady(f"DeepSeek API error: {err}") from err
    return client
