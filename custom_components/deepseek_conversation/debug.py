"""Exhaustive debug suite for DeepSeek Conversation — service ``run_debug``."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import importlib.metadata
import json
import os
import sys
import time
from collections.abc import Mapping
from typing import Any, Callable, Coroutine

import openai

from homeassistant.config_entries import ConfigEntry  # pyright: ignore[reportMissingImports]
from homeassistant.const import CONF_API_KEY, CONF_LLM_HASS_API  # pyright: ignore[reportMissingImports]
from homeassistant.core import HomeAssistant  # pyright: ignore[reportMissingImports]
from homeassistant.helpers import entity_registry as er  # pyright: ignore[reportMissingImports]
from homeassistant.helpers import llm  # pyright: ignore[reportMissingImports]
from homeassistant.helpers.httpx_client import get_async_client  # pyright: ignore[reportMissingImports]

from .const import (
    CONF_CHAT_MODEL,
    CONF_MAX_TOKENS,
    CONF_PROMPT,
    CONF_REASONING_EFFORT,
    CONF_TEMPERATURE,
    CONF_THINKING_ENABLED,
    CONF_TOP_P,
    CONF_BASE_URL,
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_THINKING_ENABLED,
    DOMAIN,
    LOGGER,
    RECOMMENDED_CHAT_MODEL,
    RECOMMENDED_MAX_TOKENS,
    RECOMMENDED_REASONING_EFFORT,
    RECOMMENDED_TEMPERATURE,
    RECOMMENDED_TOP_P,
    DEEPSEEK_API_BASE_URL,
    MAX_TOKENS_UPPER_BOUND,
)
from .options import coerce_max_tokens, request_timeout_from_options
from .request_builder import deepseek_chat_thinking_params
from .usage_metrics import completion_usage_from_api

REPORT_FILENAME = "deepseek_conversation_debug_report.txt"
LOG_CANDIDATES = ("home-assistant.log", "home-assistant.log.1")


def _redact_entry(entry: ConfigEntry) -> dict[str, Any]:
    """Summarise the entry for the report, with the API key masked.

    Settings live on the agents, so they are listed per subentry; the entry
    itself carries only the connection.
    """
    data = {**entry.data}
    if CONF_API_KEY in data:
        data[CONF_API_KEY] = "***"
    return {
        "title": entry.title,
        "entry_id": entry.entry_id,
        "data": data,
        "agents": [
            {
                "type": subentry.subentry_type,
                "title": subentry.title,
                "settings": dict(subentry.data),
            }
            for subentry in entry.subentries.values()
        ],
    }


def _read_log_tail(config_dir: str, max_lines: int) -> str:
    needles_primary = (
        DOMAIN,
        "deepseek",
        "DeepSeek",
        "deepseek debug",
        "[deepseek debug]",
        "Error talking to DeepSeek",
        "async_provide_llm",
        "ConverseError",
    )
    needles_error = ("ERROR", "Traceback", "Exception in ")
    blocks: list[str] = []
    for name in LOG_CANDIDATES:
        path = os.path.join(config_dir, name)
        if not os.path.isfile(path):
            continue
        try:
            with open(path, encoding="utf-8", errors="replace") as f:
                file_lines = f.readlines()
        except OSError as err:
            return f"--- could not read {path}: {err} ---\n"
        window = file_lines[-max_lines * 12 :]

        def pick(nlist: tuple[str, ...]) -> list[str]:
            out: list[str] = []
            for line in window:
                if any(n in line for n in nlist):
                    out.append(line.rstrip("\n"))
            return out[-max_lines:]

        blocks.append(f"=== A) integration / deepseek (max {max_lines} lines) from {name} ===\n")
        blocks.append("\n".join(pick(needles_primary)))
        blocks.append(f"\n=== B) errors / trace (max {max_lines // 2} lines) from {name} ===\n")
        blocks.append("\n".join(pick(needles_error)[-(max_lines // 2) :]))
        blocks.append("\n")
        return "".join(blocks)
    return "No home-assistant.log found under config.\n"


def _write_report(path: str, body: str) -> None:
    """Write the debug report file (runs in the executor)."""
    with open(path, "w", encoding="utf-8") as f:
        f.write(body)


def _ha_version() -> str:
    try:
        from homeassistant.const import __version__ as ver  # type: ignore[import-not-found]

        return str(ver)
    except Exception:
        return "unknown"


def _pkg_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


async def _timed(
    name: str,
    coro_factory: Callable[[], Coroutine[Any, Any, Any]],
) -> tuple[Any, float, str | None]:
    t0 = time.perf_counter()
    err: str | None = None
    try:
        result = await coro_factory()
        return result, time.perf_counter() - t0, None
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
        return None, time.perf_counter() - t0, err


def _msg_summary(msg: Any) -> dict[str, Any]:
    if msg is None:
        return {}
    out: dict[str, Any] = {
        "content_len": len((msg.content or "")),
        "content_preview": (msg.content or "")[:240],
    }
    rc = getattr(msg, "reasoning_content", None)
    if rc is not None:
        out["reasoning_chars"] = len(rc) if isinstance(rc, str) else 0
        out["reasoning_preview"] = (rc[:200] + "…") if isinstance(rc, str) and len(rc) > 200 else rc
    return out


def _choice_meta(resp: Any) -> dict[str, Any]:
    ch = resp.choices[0] if resp.choices else None
    if not ch:
        return {}
    meta: dict[str, Any] = {"finish_reason": getattr(ch, "finish_reason", None)}
    if getattr(ch, "message", None):
        meta.update(_msg_summary(ch.message))
    return meta


async def _latency_profile(
    client: openai.AsyncOpenAI,
    model: str,
    opts: dict[str, Any],
    out: dict[str, Any],
    log: Callable[[str], None],
) -> None:
    """Time identical minimal requests to separate fixed cost from variance.

    Fires three back-to-back requests, then one after a pause longer than the
    httpx keep-alive window. Interpreting the numbers:

    * back-to-back calls all similar and the delayed one much slower — the
      connection setup (DNS/TCP/TLS, or a proxy's own upstream handshake)
      dominates, not inference;
    * all four similar — the floor is provider-side inference/queueing;
    * high spread between identical back-to-back calls — the endpoint queues or
      routes requests unevenly, which is what makes some voice commands slow.
    """
    prompt = [{"role": "user", "content": "Reply with the single word: ok"}]
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": prompt,
        "max_tokens": 4,
        "stream": False,
        **deepseek_chat_thinking_params(
            thinking_enabled=False,
            reasoning_effort=str(
                opts.get(CONF_REASONING_EFFORT, RECOMMENDED_REASONING_EFFORT)
            ),
            model=model,
        ),
    }

    async def _one() -> float:
        t0 = time.perf_counter()
        await client.with_options(timeout=60.0).chat.completions.create(**kwargs)
        return time.perf_counter() - t0

    result: dict[str, Any] = {"ok": False}
    try:
        warm = [round(await _one(), 3) for _ in range(3)]
        # Longer than homeassistant.helpers.httpx_client.KEEP_ALIVE_TIMEOUT (15 s)
        # so the pooled connection is certainly gone.
        log("latency_profile: idling 16 s to force a reconnect for the next call")
        await asyncio.sleep(16)
        cold = round(await _one(), 3)
        spread = round(max(warm) - min(warm), 3)
        result = {
            "ok": True,
            "back_to_back_seconds": warm,
            "after_18s_idle_seconds": cold,
            "back_to_back_spread_seconds": spread,
            "reconnect_penalty_seconds": round(cold - min(warm), 3),
            "note": (
                "Large reconnect_penalty => connection setup dominates. "
                "Large back_to_back_spread => the endpoint queues/routes unevenly. "
                "Both small => the floor is provider-side inference time."
            ),
        }
    except Exception as err:  # diagnostics must never abort the report
        result = {"ok": False, "error": f"{type(err).__name__}: {err}"}

    out["edge_cases"]["latency_profile"] = result
    log(f"latency_profile: {result}")


async def async_run_debug_suite(
    hass: HomeAssistant,
    entry: ConfigEntry,
    *,
    agent_options: Mapping[str, Any],
    log_tail_lines: int = 600,
    max_completion_tokens_test: int = 48,
) -> dict[str, Any]:
    """Run extended diagnostics; write ``/config/deepseek_conversation_debug_report.txt``.

    ``agent_options`` are the settings the report probes with - the agent named
    in the action, or the entry's first one.
    """
    lines: list[str] = []
    out: dict[str, Any] = {
        "environment": {},
        "http": {},
        "llm": {},
        "entities": [],
        "completions": {},
        "streams": {},
        "edge_cases": {},
        "report_path": None,
        "redacted_entry": _redact_entry(entry),
    }

    def log(msg: str) -> None:
        lines.append(msg)
        LOGGER.info("[deepseek debug] %s", msg)

    log(f"=== DeepSeek Conversation EXHAUSTIVE debug {datetime.now(timezone.utc).isoformat()} ===")
    log(f"Python {sys.version.split()[0]} | platform {sys.platform}")

    # --- Environment ---
    env = {
        "home_assistant_version": _ha_version(),
        "openai_sdk_version": getattr(openai, "__version__", "unknown"),
        "voluptuous_openapi_version": _pkg_version("voluptuous-openapi"),
        "h2_version": _pkg_version("h2"),
        "integration_domain": DOMAIN,
        "config_dir": hass.config.config_dir,
        "component_in_loaded_components": DOMAIN in hass.config.components,
        "coerce_max_tokens_upper_bound": MAX_TOKENS_UPPER_BOUND,
        "request_timeout_seconds": request_timeout_from_options(agent_options),
        # HTTP/1.1 here means every API round re-handshakes TLS, because the
        # OpenAI SDK closes each streamed response without draining it.
        "negotiated_http_version": getattr(
            entry.runtime_data, "http_version", None
        )
        if entry.runtime_data is not None
        else None,
    }
    out["environment"] = env
    for k, v in env.items():
        log(f"ENV {k}={v!r}")

    client: openai.AsyncClient | None = None
    runtime = entry.runtime_data
    if runtime is not None:
        client = runtime.client
    if not isinstance(client, openai.AsyncOpenAI):
        log("FAIL: runtime_data client is not AsyncOpenAI — aborting API tests.")
        out["completions"]["_abort"] = {
            "ok": False,
            "error": "runtime_data client is not AsyncOpenAI",
        }
        out["summary"] = {
            "error": "no_openai_client",
            "completions_ok": 0,
            "completions_fail": 1,
            "streams_ok": 0,
            "streams_fail": 0,
            "edge_ok": 0,
            "edge_fail": 0,
        }
        flat_abort: dict[str, Any] = {"client": "missing", "summary": out["summary"]}
        for k, v in out["completions"].items():
            flat_abort[f"completion.{k}"] = v
        out["tests"] = flat_abort
        report_body = "\n".join(lines) + "\n" + _read_log_tail(hass.config.config_dir, log_tail_lines)
        path = hass.config.path(REPORT_FILENAME)
        await hass.async_add_executor_job(_write_report, path, report_body)
        out["report_path"] = path
        return out

    opts = dict(agent_options)
    model = str(opts.get(CONF_CHAT_MODEL, RECOMMENDED_CHAT_MODEL))
    raw_mt = opts.get(CONF_MAX_TOKENS, RECOMMENDED_MAX_TOKENS)
    mt_coerced = coerce_max_tokens(raw_mt)
    thinking_opt = bool(opts.get(CONF_THINKING_ENABLED, DEFAULT_THINKING_ENABLED))
    base_url = str(entry.data.get(CONF_BASE_URL, DEEPSEEK_API_BASE_URL))
    log(
        f"OPTIONS model={model!r} raw_max_tokens={raw_mt!r} (type={type(raw_mt).__name__}) "
        f"coerced_max_tokens={mt_coerced} coerce_ceiling={MAX_TOKENS_UPPER_BOUND} "
        f"thinking_option={thinking_opt} "
        f"request_timeout={request_timeout_from_options(opts):.0f}s "
        f"base_url={base_url!r}"
    )

    # --- HTTP probe (no API key in URL) ---
    async def _http_probe() -> dict[str, Any]:
        http = get_async_client(hass)
        url = base_url.rstrip("/") + "/"
        r = await http.get(url, timeout=10.0)
        return {"status": r.status_code, "len": len(r.text or ""), "url": url}

    _hr, ht, herr = await _timed("http_get", _http_probe)
    reach_note: str | None = None
    if herr is None and isinstance(_hr, dict):
        st = _hr.get("status")
        if st in (401, 403):
            reach_note = (
                "401/403 on base URL without Authorization is expected for this probe; "
                "the host answers and TLS/route work. API auth is tested via chat.completions."
            )
        elif isinstance(st, int) and st >= 500:
            reach_note = "5xx from base URL — check provider status or proxy."
    out["http"] = {
        "ok": herr is None,
        "seconds": round(ht, 3),
        "error": herr,
        "result": _hr,
        "reachability_note": reach_note,
    }
    log(f"HTTP GET {(_hr or {})} err={herr} t={ht:.3f}s note={reach_note!r}")

    # --- models.list (optional) ---
    async def _models_list() -> Any:
        return await client.with_options(timeout=15.0).models.list()

    ml, mlt, mlerr = await _timed("models_list", _models_list)
    out["http"]["models_list_seconds"] = round(mlt, 3)
    if mlerr:
        out["http"]["models_list_error"] = mlerr
        log(f"models.list: SKIP/FAIL {mlerr}")
    else:
        n = len(getattr(ml, "data", []) or [])
        out["http"]["models_list_count"] = n
        log(f"models.list: OK count={n} t={mlt:.3f}s")

    # --- LLM APIs registered ---
    try:
        apis = llm.async_get_apis(hass)
        api_rows = [{"id": a.id, "name": a.name} for a in apis]
        sel = opts.get(CONF_LLM_HASS_API)
        sel_list: list[str] = [sel] if isinstance(sel, str) else sel or []
        out["llm"] = {
            "selected_api_id": sel,
            "registered_apis": api_rows,
            "selected_exists": bool(sel_list and any(a.id in sel_list for a in apis)),
        }
        log(f"LLM APIs registered={len(api_rows)} selected={sel!r} exists={out['llm']['selected_exists']}")
    except Exception as e:
        out["llm"] = {"error": str(e)}
        log(f"LLM API enumeration FAIL: {e}")

    # --- Entity registry ---
    try:
        reg = er.async_get(hass)
        for eid, ent in reg.entities.items():
            if ent.config_entry_id != entry.entry_id:
                continue
            st = hass.states.get(eid)
            out["entities"].append(
                {
                    "entity_id": eid,
                    "platform": ent.platform,
                    "unique_id": ent.unique_id,
                    "disabled_by": str(ent.disabled_by) if ent.disabled_by else None,
                    "state": st.state if st else None,
                }
            )
        log(f"ENTITY registry count={len(out['entities'])}")
    except Exception as e:
        out["entities"] = [{"error": str(e)}]
        log(f"ENTITY registry FAIL: {e}")

    # --- Completion tests ---
    async def _complete(name: str, kwargs: dict[str, Any]) -> None:
        log(f"--- completion {name} kwargs_keys={sorted(kwargs.keys())} ---")
        res, dt, err = await _timed(
            name,
            lambda k=kwargs: client.with_options(timeout=120.0).chat.completions.create(**k),
        )
        block: dict[str, Any] = {"ok": err is None, "seconds": round(dt, 3), "error": err}
        if err is None and res is not None:
            block["choice"] = _choice_meta(res)
            block["model_from_response"] = getattr(res, "model", None)
            block["usage"] = getattr(res, "usage", None)
            if block["usage"] is not None and hasattr(block["usage"], "model_dump"):
                block["usage"] = block["usage"].model_dump(exclude_none=True)
        out["completions"][name] = block
        log(f"completion {name}: {'OK' if err is None else 'FAIL'} t={dt:.3f}s meta={block.get('choice')}")

    # 1) ping with enough tokens for visible reply
    await _complete(
        "ping_max_tokens_8",
        {
            "model": model,
            "messages": [{"role": "user", "content": "Reply with the single word: PONG"}],
            "max_tokens": 8,
            "stream": False,
            **deepseek_chat_thinking_params(
                thinking_enabled=False,
                reasoning_effort=str(opts.get(CONF_REASONING_EFFORT, RECOMMENDED_REASONING_EFFORT)),
                model=model,
            ),
        },
    )

    cap = max(8, min(max_completion_tokens_test, mt_coerced, 128))
    out["debug_token_policy"] = {
        "configured_coerced_max_tokens": mt_coerced,
        "suite_cap_most_tests": cap,
        "max_completion_tokens_test_param": max_completion_tokens_test,
        "note": (
            "Debug suite intentionally uses small max_tokens per call (ping=8; most tests "
            "use suite_cap <= 128) for speed and API cost. This is NOT the Assist conversation "
            "path — see conversation.py model_args['max_tokens'] = coerce_max_tokens(options[...])."
        ),
    }
    log(f"TOKEN POLICY (debug only): {out['debug_token_policy']}")

    # 2) production-like no-thinking with sampling
    args_nt: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": "Reply with exactly: OK"}],
        "max_tokens": cap,
        "stream": False,
        "temperature": float(opts.get(CONF_TEMPERATURE, RECOMMENDED_TEMPERATURE)),
        "top_p": float(opts.get(CONF_TOP_P, RECOMMENDED_TOP_P)),
        **deepseek_chat_thinking_params(
            thinking_enabled=False,
            reasoning_effort=str(opts.get(CONF_REASONING_EFFORT, RECOMMENDED_REASONING_EFFORT)),
            model=model,
        ),
    }
    await _complete("non_stream_no_thinking_user_sampling", args_nt)

    # 3) thinking on, no sampling in kwargs (matches integration)
    args_t: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": "Say hi in one short word."}],
        "max_tokens": min(48, cap),
        "stream": False,
        **deepseek_chat_thinking_params(
            thinking_enabled=True,
            reasoning_effort=str(opts.get(CONF_REASONING_EFFORT, RECOMMENDED_REASONING_EFFORT)),
            model=model,
        ),
    }
    await _complete("non_stream_thinking_only", args_t)

    # 4) system + user (like service path)
    sys_prompt = (opts.get(CONF_PROMPT) or "").strip() or DEFAULT_SYSTEM_PROMPT
    await _complete(
        "non_stream_with_system_message",
        {
            "model": model,
            "messages": [
                {"role": "system", "content": sys_prompt[:2000]},
                {"role": "user", "content": "Confirm you are ready with one word."},
            ],
            "max_tokens": min(24, cap),
            "stream": False,
            **deepseek_chat_thinking_params(
                thinking_enabled=False,
                reasoning_effort=str(opts.get(CONF_REASONING_EFFORT, RECOMMENDED_REASONING_EFFORT)),
                model=model,
            ),
        },
    )

    # 5) mini multi-turn
    await _complete(
        "non_stream_multiturn",
        {
            "model": model,
            "messages": [
                {"role": "user", "content": "Remember code X7."},
                {"role": "assistant", "content": "OK, code X7 noted."},
                {"role": "user", "content": "Which code did I give?"},
            ],
            "max_tokens": min(64, cap),
            "stream": False,
            **deepseek_chat_thinking_params(
                thinking_enabled=False,
                reasoning_effort=str(opts.get(CONF_REASONING_EFFORT, RECOMMENDED_REASONING_EFFORT)),
                model=model,
            ),
        },
    )

    # 6) high max_tokens acceptance (still small completion text)
    high_cap = min(2048, mt_coerced, 8192)
    out["debug_token_policy"]["high_max_tokens_probe_request"] = high_cap
    await _complete(
        "non_stream_high_max_tokens_probe",
        {
            "model": model,
            "messages": [{"role": "user", "content": "Answer only: 1"}],
            "max_tokens": high_cap,
            "stream": False,
            **deepseek_chat_thinking_params(
                thinking_enabled=False,
                reasoning_effort=str(opts.get(CONF_REASONING_EFFORT, RECOMMENDED_REASONING_EFFORT)),
                model=model,
            ),
        },
    )
    out["completions"]["_note_high_cap"] = f"Requested max_tokens={high_cap} (from coerced user limit)"

    # 7) observational: thinking + temperature (may be ignored or rejected by API)
    args_conflict: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": "Reply only: test"}],
        "max_tokens": 16,
        "stream": False,
        "temperature": 0.7,
        "top_p": 0.9,
        **deepseek_chat_thinking_params(
            thinking_enabled=True,
            reasoning_effort=str(opts.get(CONF_REASONING_EFFORT, RECOMMENDED_REASONING_EFFORT)),
            model=model,
        ),
    }
    await _complete("non_stream_thinking_plus_sampling_params_observational", args_conflict)

    # --- Stream tests ---
    async def _stream(name: str, kwargs: dict[str, Any], max_chunks: int = 120) -> None:
        log(f"--- stream {name} ---")

        async def _run() -> dict[str, Any]:
            t_start = time.perf_counter()
            stream = await client.with_options(timeout=90.0).chat.completions.create(**kwargs)
            chunks = 0
            content_acc = ""
            reasoning_acc = ""
            last_role = None
            ttfb: float | None = None
            gaps: list[float] = []
            t_prev = t_start
            usage_seen: dict[str, Any] | None = None
            async for ev in stream:
                now = time.perf_counter()
                chunks += 1
                if ttfb is None:
                    ttfb = now - t_start
                else:
                    gaps.append(now - t_prev)
                t_prev = now
                if chunks > max_chunks:
                    break
                if (u := getattr(ev, "usage", None)) is not None:
                    parsed = completion_usage_from_api(u)
                    if parsed is not None:
                        usage_seen = {
                            "prompt_tokens": parsed.prompt_tokens,
                            "completion_tokens": parsed.completion_tokens,
                            "reasoning_tokens": parsed.reasoning_tokens,
                            "cache_hit_tokens": parsed.cache_hit_tokens,
                            "cache_miss_tokens": parsed.cache_miss_tokens,
                            "cache_hit_rate": parsed.cache_hit_rate,
                        }
                if not ev.choices:
                    continue
                d = ev.choices[0].delta
                if d is None:
                    continue
                if d.content:
                    content_acc += d.content
                rd = getattr(d, "reasoning_content", None)
                if rd:
                    reasoning_acc += rd
                if d.role:
                    last_role = d.role
            total = time.perf_counter() - t_start
            # A proxy that buffers the whole reply re-emits it as one or two
            # chunks with all the time spent before the first one.
            buffered = chunks <= 2 or (ttfb is not None and total > 0 and ttfb / total > 0.9)
            return {
                "chunks": chunks,
                "content_len": len(content_acc),
                "content_preview": content_acc[:300],
                "reasoning_len": len(reasoning_acc),
                "last_delta_role": last_role,
                "http_version": getattr(
                    getattr(stream, "response", None), "http_version", None
                ),
                "time_to_first_chunk_s": round(ttfb, 3) if ttfb is not None else None,
                "mean_inter_chunk_ms": (
                    round(sum(gaps) / len(gaps) * 1000, 1) if gaps else None
                ),
                "looks_buffered_by_proxy": buffered,
                "usage": usage_seen,
            }

        res, dt, err = await _timed(name, _run)
        out["streams"][name] = {
            "ok": err is None,
            "seconds": round(dt, 3),
            "error": err,
            "stats": res,
        }
        log(f"stream {name}: {'OK' if err is None else 'FAIL'} t={dt:.3f}s stats={res}")

    await _stream(
        "stream_no_thinking",
        {
            "model": model,
            "messages": [{"role": "user", "content": "Count: 1,2,3 as digits only."}],
            "max_tokens": 24,
            "stream": True,
            **deepseek_chat_thinking_params(
                thinking_enabled=False,
                reasoning_effort=str(opts.get(CONF_REASONING_EFFORT, RECOMMENDED_REASONING_EFFORT)),
                model=model,
            ),
        },
    )

    await _stream(
        "stream_thinking_on",
        {
            "model": model,
            "messages": [{"role": "user", "content": "Give a one-word answer: OK"}],
            "max_tokens": 32,
            "stream": True,
            **deepseek_chat_thinking_params(
                thinking_enabled=True,
                reasoning_effort=str(opts.get(CONF_REASONING_EFFORT, RECOMMENDED_REASONING_EFFORT)),
                model=model,
            ),
        },
    )

    # --- Edge: max_tokens=1 and empty prompt edge ---
    await _complete(
        "edge_max_tokens_1",
        {
            "model": model,
            "messages": [{"role": "user", "content": "."}],
            "max_tokens": 1,
            "stream": False,
            **deepseek_chat_thinking_params(
                thinking_enabled=False,
                reasoning_effort=str(opts.get(CONF_REASONING_EFFORT, RECOMMENDED_REASONING_EFFORT)),
                model=model,
            ),
        },
    )

    # --- Optional: JSON mode (may fail on some models) ---
    await _complete(
        "edge_json_object_mode",
        {
            "model": model,
            "messages": [{"role": "user", "content": 'Return JSON with key "ok" true'}],
            "max_tokens": 64,
            "stream": False,
            "response_format": {"type": "json_object"},
            **deepseek_chat_thinking_params(
                thinking_enabled=False,
                reasoning_effort=str(opts.get(CONF_REASONING_EFFORT, RECOMMENDED_REASONING_EFFORT)),
                model=model,
            ),
        },
    )

    # --- Parallel smoke: two tiny pings (concurrency sanity) ---
    async def _dual() -> tuple[str, str]:
        async def one(msg: str) -> str:
            r = await client.with_options(timeout=20.0).chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": msg}],
                max_tokens=4,
                stream=False,
                **deepseek_chat_thinking_params(
                    thinking_enabled=False,
                    reasoning_effort=str(
                        opts.get(CONF_REASONING_EFFORT, RECOMMENDED_REASONING_EFFORT)
                    ),
                    model=model,
                ),
            )
            return (r.choices[0].message.content or "").strip()

        a, b = await asyncio.gather(one("A"), one("B"))
        return a, b

    dr, ddt, derr = await _timed("parallel_two_pings", _dual)
    out["edge_cases"]["parallel_two_pings"] = {
        "ok": derr is None,
        "seconds": round(ddt, 3),
        "error": derr,
        "results": (
            {"a": dr[0], "b": dr[1]}
            if (derr is None and dr is not None)
            else None
        ),
    }
    log(f"parallel_two_pings: {out['edge_cases']['parallel_two_pings']}")

    # --- Round-trip latency profile ---
    await _latency_profile(client, model, opts, out, log)

    # --- Summary counts ---
    def _count_ok(container: dict[str, Any]) -> tuple[int, int]:
        okc = failc = 0
        for v in container.values():
            if not isinstance(v, dict):
                continue
            if v.get("ok") is True:
                okc += 1
            elif v.get("ok") is False:
                failc += 1
        return okc, failc

    c_ok, c_fail = _count_ok(out["completions"])
    s_ok, s_fail = _count_ok(out["streams"])
    e_ok, e_fail = _count_ok(out["edge_cases"])
    out["summary"] = {
        "completions_ok": c_ok,
        "completions_fail": c_fail,
        "streams_ok": s_ok,
        "streams_fail": s_fail,
        "edge_ok": e_ok,
        "edge_fail": e_fail,
    }
    log(f"SUMMARY completions OK/FAIL={c_ok}/{c_fail} streams={s_ok}/{s_fail} edge={e_ok}/{e_fail}")

    log("--- log excerpt ---")
    log_excerpt = await hass.async_add_executor_job(
        _read_log_tail, hass.config.config_dir, log_tail_lines
    )
    lines.append(log_excerpt)
    out["log_excerpt_chars"] = len(log_excerpt)

    # JSON appendix for machine-readable copy
    lines.append("\n=== JSON summary (machine-readable) ===\n")
    try:
        json_block = json.dumps(
            {k: v for k, v in out.items() if k != "report_path"},
            default=str,
            indent=2,
        )
        lines.append(json_block[:200_000])
    except TypeError as e:
        lines.append(f"(json encode error: {e})")

    report_body = "\n".join(lines)
    path = hass.config.path(REPORT_FILENAME)
    await hass.async_add_executor_job(_write_report, path, report_body)
    out["report_path"] = path
    LOGGER.info("DeepSeek exhaustive debug report written to %s", path)

    # Flatten `tests` for backward compatibility with service response consumers
    flat: dict[str, Any] = {"client": "ok", "summary": out["summary"]}
    flat.update({f"env.{k}": v for k, v in out["environment"].items()})
    flat["http"] = out["http"]
    flat["llm"] = out["llm"]
    flat["entities"] = out["entities"]
    flat["debug_token_policy"] = out.get("debug_token_policy", {})
    for k, v in out["completions"].items():
        flat[f"completion.{k}"] = v
    for k, v in out["streams"].items():
        flat[f"stream.{k}"] = v
    for k, v in out["edge_cases"].items():
        flat[f"edge.{k}"] = v
    out["tests"] = flat

    return out
