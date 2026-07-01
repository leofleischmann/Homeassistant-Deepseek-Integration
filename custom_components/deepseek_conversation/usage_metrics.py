"""Token usage tracking for DeepSeek API completions.

Updated by conversation.py (Assist stream), __init__.py (generate_content), and
sensor.py (RestoreSensor entities). Stream usage requires stream_options in
build_chat_completion_args().

Connection fingerprint (base URL + API key) is compared on sensor setup; when it
changes after reconfigure/reauth, cumulative counters reset (see sensor.py).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, TYPE_CHECKING

from .const import CONF_API_KEY, CONF_BASE_URL, DEEPSEEK_API_BASE_URL, DOMAIN, LOGGER

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant  # pyright: ignore[reportMissingImports]

    from .sensor import (
        DeepSeekLastRequestSensor,
        DeepSeekSnapshotSensor,
        DeepSeekUsageCounterSensor,
    )

_CONNECTION_FP_KEY = "connection_fingerprint"


def connection_fingerprint(data: Mapping[str, Any]) -> str:
    """Stable id for the API endpoint credentials (reconfigure/reauth detection)."""
    base = data.get(CONF_BASE_URL) or DEEPSEEK_API_BASE_URL
    key = data.get(CONF_API_KEY, "")
    return f"{base}:{key}"


def connection_changed_since_last_setup(
    hass: HomeAssistant, entry_id: str, data: Mapping[str, Any]
) -> bool:
    """Return True when base URL or API key changed since the previous platform setup."""
    fp = connection_fingerprint(data)
    store = hass.data.setdefault(DOMAIN, {})
    key = f"{entry_id}_{_CONNECTION_FP_KEY}"
    previous = store.get(key)
    store[key] = fp
    changed = previous is not None and previous != fp
    if changed:
        LOGGER.info(
            "[Debug usage_metrics]: connection changed for %s, usage counters will reset",
            entry_id,
        )
    return changed


@dataclass(frozen=True, slots=True)
class CompletionUsage:
    """Normalized token usage from an OpenAI-compatible completion response."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    reasoning_tokens: int = 0

    def __post_init__(self) -> None:
        if self.total_tokens <= 0 and (self.prompt_tokens or self.completion_tokens):
            object.__setattr__(
                self,
                "total_tokens",
                self.prompt_tokens + self.completion_tokens,
            )


def completion_usage_from_api(usage: Any) -> CompletionUsage | None:
    """Parse ``usage`` from a chat completion or stream chunk."""
    if usage is None:
        return None

    if hasattr(usage, "model_dump"):
        data = usage.model_dump(exclude_none=True)
    elif isinstance(usage, dict):
        data = usage
    else:
        data = {
            "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
            "completion_tokens": getattr(usage, "completion_tokens", 0) or 0,
            "total_tokens": getattr(usage, "total_tokens", 0) or 0,
        }
        details = getattr(usage, "completion_tokens_details", None)
        if details is not None:
            data["completion_tokens_details"] = (
                details.model_dump(exclude_none=True)
                if hasattr(details, "model_dump")
                else details
            )

    prompt = int(data.get("prompt_tokens") or 0)
    completion = int(data.get("completion_tokens") or 0)
    total = int(data.get("total_tokens") or 0)
    reasoning = 0
    details = data.get("completion_tokens_details")
    if isinstance(details, dict):
        reasoning = int(details.get("reasoning_tokens") or 0)

    if not any((prompt, completion, total, reasoning)):
        return None

    return CompletionUsage(
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=total,
        reasoning_tokens=reasoning,
    )


class UsageTracker:
    """Accumulates API token usage and drives sensor entities."""

    def __init__(self) -> None:
        self.request_count = 0
        self.last_usage: CompletionUsage | None = None
        self.last_source: str | None = None
        self._prompt: DeepSeekUsageCounterSensor | None = None
        self._completion: DeepSeekUsageCounterSensor | None = None
        self._total: DeepSeekUsageCounterSensor | None = None
        self._reasoning: DeepSeekUsageCounterSensor | None = None
        self._api_requests: DeepSeekUsageCounterSensor | None = None
        self._last: DeepSeekLastRequestSensor | None = None
        self._last_prompt: DeepSeekSnapshotSensor | None = None
        self._last_completion: DeepSeekSnapshotSensor | None = None

    def bind_sensors(
        self,
        *,
        prompt: DeepSeekUsageCounterSensor,
        completion: DeepSeekUsageCounterSensor,
        total: DeepSeekUsageCounterSensor,
        reasoning: DeepSeekUsageCounterSensor,
        api_requests: DeepSeekUsageCounterSensor,
        last_request: DeepSeekLastRequestSensor,
        last_request_prompt: DeepSeekSnapshotSensor,
        last_request_completion: DeepSeekSnapshotSensor,
    ) -> None:
        """Register sensor entities (called from sensor platform setup)."""
        self._prompt = prompt
        self._completion = completion
        self._total = total
        self._reasoning = reasoning
        self._api_requests = api_requests
        self._last = last_request
        self._last_prompt = last_request_prompt
        self._last_completion = last_request_completion

    def reset_all(self) -> None:
        """Zero cumulative and snapshot sensors after API connection change."""
        self.request_count = 0
        self.last_usage = None
        self.last_source = None
        for sensor in (
            self._prompt,
            self._completion,
            self._total,
            self._reasoning,
            self._api_requests,
        ):
            if sensor is not None:
                sensor.reset_to_zero()
        if self._last is not None:
            self._last.reset_to_zero()
        if self._last_prompt is not None:
            self._last_prompt.reset_to_zero()
        if self._last_completion is not None:
            self._last_completion.reset_to_zero()
        LOGGER.info("[Debug usage_metrics]: usage counters reset after connection change")

    def record(self, usage: CompletionUsage, *, source: str) -> None:
        """Add one API completion's usage to cumulative sensors."""
        if self._prompt is None:
            LOGGER.debug(
                "[Debug usage_metrics]: usage received before sensors bound: %s",
                usage,
            )
            return

        self.request_count += 1
        self.last_usage = usage
        self.last_source = source

        self._prompt.increment(usage.prompt_tokens)
        self._completion.increment(usage.completion_tokens)
        total_delta = usage.total_tokens or (
            usage.prompt_tokens + usage.completion_tokens
        )
        self._total.increment(total_delta)
        self._reasoning.increment(usage.reasoning_tokens)
        self._api_requests.increment(1)
        self._last.set_usage(usage, source=source, request_count=self.request_count)
        self._last_prompt.set_value(usage.prompt_tokens)
        self._last_completion.set_value(usage.completion_tokens)

        LOGGER.info(
            "[Debug usage_metrics]: +%d prompt / +%d completion tokens "
            "(total +%d, reasoning=%d, source=%s, requests=%d)",
            usage.prompt_tokens,
            usage.completion_tokens,
            total_delta,
            usage.reasoning_tokens,
            source,
            self.request_count,
        )

    def usage_as_dict(self, usage: CompletionUsage) -> dict[str, int]:
        """Serialize usage for service responses."""
        return {
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "total_tokens": usage.total_tokens
            or usage.prompt_tokens + usage.completion_tokens,
            "reasoning_tokens": usage.reasoning_tokens,
        }
