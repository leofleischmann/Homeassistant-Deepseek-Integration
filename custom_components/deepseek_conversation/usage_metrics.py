"""Token usage tracking for DeepSeek API completions.

Updated by chat_session.py (Assist stream), services.py (generate_content), and
sensor.py (RestoreSensor entities). Manual reset via button.py -> reset_all().
Stream usage requires stream_options in build_chat_completion_args().
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

from .const import LOGGER

if TYPE_CHECKING:
    from .sensor import (
        DeepSeekLastRequestSensor,
        DeepSeekSnapshotSensor,
        DeepSeekUsageCounterSensor,
    )


@dataclass(frozen=True, slots=True)
class CompletionUsage:
    """Normalized token usage from an OpenAI-compatible completion response."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    reasoning_tokens: int = 0
    cache_hit_tokens: int = 0
    cache_miss_tokens: int = 0

    def __post_init__(self) -> None:
        if self.total_tokens <= 0 and (self.prompt_tokens or self.completion_tokens):
            object.__setattr__(
                self,
                "total_tokens",
                self.prompt_tokens + self.completion_tokens,
            )
        if (
            self.cache_miss_tokens <= 0
            and self.cache_hit_tokens > 0
            and self.prompt_tokens > self.cache_hit_tokens
        ):
            # OpenAI-style gateways report only the cached count; derive the rest.
            object.__setattr__(
                self,
                "cache_miss_tokens",
                self.prompt_tokens - self.cache_hit_tokens,
            )

    @property
    def cache_hit_rate(self) -> float:
        """Share of prompt tokens served from the provider's prefix cache (0-1)."""
        cached_total = self.cache_hit_tokens + self.cache_miss_tokens
        if cached_total <= 0:
            return 0.0
        return round(self.cache_hit_tokens / cached_total, 4)


def _details_as_dict(details: Any) -> dict[str, Any]:
    """Normalize a ``*_tokens_details`` value to a plain dict.

    Gateways return these as pydantic models, plain dicts, or bare objects; the
    last kind used to fall through as a non-dict and silently drop the nested
    counters.
    """
    if details is None:
        return {}
    if isinstance(details, dict):
        return details
    if hasattr(details, "model_dump"):
        return details.model_dump(exclude_none=True)
    return {
        key: getattr(details, key)
        for key in ("reasoning_tokens", "cached_tokens")
        if getattr(details, key, None) is not None
    }


def _cache_tokens_from_usage(data: dict[str, Any]) -> tuple[int, int]:
    """Read prefix-cache counters from an OpenAI-compatible ``usage`` payload.

    DeepSeek reports ``prompt_cache_hit_tokens`` / ``prompt_cache_miss_tokens``
    at the top level; OpenAI-compatible gateways instead nest a single
    ``prompt_tokens_details.cached_tokens``. Support both so the numbers are
    available regardless of which endpoint the entry points at.
    """
    hit = int(data.get("prompt_cache_hit_tokens") or 0)
    miss = int(data.get("prompt_cache_miss_tokens") or 0)
    if hit or miss:
        return hit, miss

    details = _details_as_dict(data.get("prompt_tokens_details"))
    hit = int(details.get("cached_tokens") or 0)
    return hit, miss


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
            "prompt_cache_hit_tokens": getattr(usage, "prompt_cache_hit_tokens", 0) or 0,
            "prompt_cache_miss_tokens": (
                getattr(usage, "prompt_cache_miss_tokens", 0) or 0
            ),
        }
        for attr in ("completion_tokens_details", "prompt_tokens_details"):
            details = getattr(usage, attr, None)
            if details is not None:
                data[attr] = _details_as_dict(details)

    prompt = int(data.get("prompt_tokens") or 0)
    completion = int(data.get("completion_tokens") or 0)
    total = int(data.get("total_tokens") or 0)
    reasoning = int(
        _details_as_dict(data.get("completion_tokens_details")).get("reasoning_tokens")
        or 0
    )
    cache_hit, cache_miss = _cache_tokens_from_usage(data)

    if not any((prompt, completion, total, reasoning, cache_hit, cache_miss)):
        return None

    return CompletionUsage(
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=total,
        reasoning_tokens=reasoning,
        cache_hit_tokens=cache_hit,
        cache_miss_tokens=cache_miss,
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
        self._cache_hit: DeepSeekUsageCounterSensor | None = None
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
        cache_hit: DeepSeekUsageCounterSensor,
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
        self._cache_hit = cache_hit
        self._api_requests = api_requests
        self._last = last_request
        self._last_prompt = last_request_prompt
        self._last_completion = last_request_completion

    def reset_all(self) -> None:
        """Zero all usage sensors (button entity or service)."""
        self.request_count = 0
        self.last_usage = None
        self.last_source = None
        for sensor in (
            self._prompt,
            self._completion,
            self._total,
            self._reasoning,
            self._cache_hit,
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
        LOGGER.info("[Debug usage_metrics]: usage counters reset manually")

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
        if self._cache_hit is not None:
            self._cache_hit.increment(usage.cache_hit_tokens)
        self._api_requests.increment(1)
        self._last.set_usage(usage, source=source, request_count=self.request_count)
        self._last_prompt.set_value(usage.prompt_tokens)
        self._last_completion.set_value(usage.completion_tokens)

        LOGGER.info(
            "[Debug usage_metrics]: +%d prompt / +%d completion tokens "
            "(total +%d, reasoning=%d, cache_hit=%d/%d, source=%s, requests=%d)",
            usage.prompt_tokens,
            usage.completion_tokens,
            total_delta,
            usage.reasoning_tokens,
            usage.cache_hit_tokens,
            usage.cache_hit_tokens + usage.cache_miss_tokens,
            source,
            self.request_count,
        )

    def usage_as_dict(self, usage: CompletionUsage) -> dict[str, int | float]:
        """Serialize usage for service responses."""
        return {
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "total_tokens": usage.total_tokens
            or usage.prompt_tokens + usage.completion_tokens,
            "reasoning_tokens": usage.reasoning_tokens,
            "cache_hit_tokens": usage.cache_hit_tokens,
            "cache_miss_tokens": usage.cache_miss_tokens,
            "cache_hit_rate": usage.cache_hit_rate,
        }
