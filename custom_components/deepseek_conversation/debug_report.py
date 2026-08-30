"""Everything the debug report is made of except the API probes themselves.

Reading the log tail, masking the API key, summarising a response, looking up
versions, writing the file. None of it talks to DeepSeek - that is ``debug.py``
- and all of it is written to survive whatever it finds, because a diagnostics
run must produce a report even when the thing being diagnosed is broken.
"""

from __future__ import annotations

import importlib.metadata
import os
from typing import Any

from homeassistant.config_entries import ConfigEntry  # pyright: ignore[reportMissingImports]
from homeassistant.const import CONF_API_KEY  # pyright: ignore[reportMissingImports]

from .const import DOMAIN

REPORT_FILENAME = "deepseek_conversation_debug_report.txt"
_LOG_CANDIDATES = ("home-assistant.log", "home-assistant.log.1")


def redact_entry(entry: ConfigEntry) -> dict[str, Any]:
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


def read_log_tail(config_dir: str, max_lines: int) -> str:
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
    for name in _LOG_CANDIDATES:
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


def write_report(path: str, body: str) -> None:
    """Write the debug report file (runs in the executor)."""
    with open(path, "w", encoding="utf-8") as f:
        f.write(body)


def ha_version() -> str:
    try:
        from homeassistant.const import __version__ as ver  # type: ignore[import-not-found]

        return str(ver)
    except Exception:
        return "unknown"


def pkg_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


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


def choice_meta(resp: Any) -> dict[str, Any]:
    ch = resp.choices[0] if resp.choices else None
    if not ch:
        return {}
    meta: dict[str, Any] = {"finish_reason": getattr(ch, "finish_reason", None)}
    if getattr(ch, "message", None):
        meta.update(_msg_summary(ch.message))
    return meta
