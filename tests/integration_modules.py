"""Import integration modules without Home Assistant installed.

CI runs ``python -m unittest discover -s tests`` on a bare Python, so a test
cannot simply import ``custom_components.deepseek_conversation.options``: that
executes the package ``__init__.py``, which pulls in ``openai`` and most of
Home Assistant.

Two steps make the pure modules reachable anyway:

* a package object for ``deepseek_conversation`` is put into ``sys.modules``
  with its ``__path__`` pointing at the real directory, so ``from .const import
  …`` resolves while the real ``__init__.py`` is never executed;
* the two Home Assistant symbols ``const.py`` needs are stubbed, but only when
  Home Assistant is genuinely absent - on a machine that has it installed the
  real modules are used.

This deliberately does not stub anything else. A module that reaches for more
of Home Assistant than that is not one of the pure modules and belongs in a
test suite that runs against a real Home Assistant.
"""

from __future__ import annotations

import importlib
import importlib.util
import pathlib
import sys
import types

PACKAGE = "deepseek_conversation"

_ROOT = pathlib.Path(__file__).resolve().parents[1] / "custom_components" / PACKAGE


def _stub(name: str, **attrs: object) -> types.ModuleType:
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    sys.modules[name] = module
    return module


def _homeassistant_available() -> bool:
    """Whether the real Home Assistant modules const.py needs are installed."""
    try:
        return (
            importlib.util.find_spec("homeassistant.const") is not None
            and importlib.util.find_spec("homeassistant.helpers.llm") is not None
        )
    except (ImportError, ValueError):
        return False


def _ensure_homeassistant() -> None:
    """Provide ``CONF_LLM_HASS_API`` and ``LLM_API_ASSIST`` if HA is missing.

    Both are keys persisted in every user's stored config entry, so their
    values are frozen by compatibility and safe to mirror here.
    """
    if _homeassistant_available():
        return

    homeassistant = _stub("homeassistant")
    homeassistant.__path__ = []  # type: ignore[attr-defined]
    helpers = _stub("homeassistant.helpers")
    helpers.__path__ = []  # type: ignore[attr-defined]
    const = _stub("homeassistant.const", CONF_LLM_HASS_API="llm_hass_api")
    llm_module = _stub("homeassistant.helpers.llm", LLM_API_ASSIST="assist")

    homeassistant.const = const  # type: ignore[attr-defined]
    homeassistant.helpers = helpers  # type: ignore[attr-defined]
    helpers.llm = llm_module  # type: ignore[attr-defined]


def _ensure_package() -> None:
    """Register the integration directory as a package without running its init."""
    if PACKAGE in sys.modules:
        return
    package = types.ModuleType(PACKAGE)
    package.__path__ = [str(_ROOT)]  # type: ignore[attr-defined]
    sys.modules[PACKAGE] = package


def load(name: str) -> types.ModuleType:
    """Import ``custom_components/deepseek_conversation/<name>.py``."""
    _ensure_homeassistant()
    _ensure_package()
    return importlib.import_module(f"{PACKAGE}.{name}")
