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

``load(name, api_stubs=True)`` goes one step further and fakes the OpenAI SDK
and the two Home Assistant modules a streaming module imports at module level.
Those stubs exist only to let the import succeed - a test using them builds its
own inputs and never asserts against the stubs - so reach for them only when
the logic under test is plain Python that happens to live in such a module.
Anything that genuinely exercises Home Assistant belongs in a test suite that
runs against a real one.
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


def _installed(module: str) -> bool:
    """Whether ``module`` can be imported for real."""
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def _homeassistant_available() -> bool:
    """Whether the real Home Assistant modules const.py needs are installed."""
    return _installed("homeassistant.const") and _installed("homeassistant.helpers.llm")


def _attach(parent: str, child: str, module: types.ModuleType) -> None:
    """Hang a stub off its parent, if the parent is one we created."""
    if (owner := sys.modules.get(parent)) is not None:
        setattr(owner, child, module)


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


def _ensure_api_stubs() -> None:
    """Satisfy the module-level imports of the streaming modules.

    Only the names those modules bind at import time, nothing more: the tests
    that ask for this build their own delta objects.
    """
    if not _installed("openai"):
        types_module = _stub("openai.types")
        types_module.__path__ = []  # type: ignore[attr-defined]
        chat = _stub("openai.types.chat", ChatCompletionChunk=type("ChatCompletionChunk", (), {}))
        openai_module = _stub("openai", AsyncStream=type("AsyncStream", (), {}))
        openai_module.__path__ = []  # type: ignore[attr-defined]
        openai_module.types = types_module  # type: ignore[attr-defined]
        types_module.chat = chat  # type: ignore[attr-defined]

    # `_installed`, not `in sys.modules`: a real Home Assistant that simply has
    # not been imported yet is still a real Home Assistant, and stubbing over it
    # would both shadow it and reach for a parent stub that was never created.
    if not _installed("homeassistant.components.conversation"):
        components = _stub("homeassistant.components")
        components.__path__ = []  # type: ignore[attr-defined]
        components.conversation = _stub("homeassistant.components.conversation")  # type: ignore[attr-defined]
        _attach("homeassistant", "components", components)

    if not _installed("homeassistant.exceptions"):
        exceptions = _stub(
            "homeassistant.exceptions",
            HomeAssistantError=type("HomeAssistantError", (Exception,), {}),
        )
        _attach("homeassistant", "exceptions", exceptions)


def _ensure_package() -> None:
    """Register the integration directory as a package without running its init."""
    if PACKAGE in sys.modules:
        return
    package = types.ModuleType(PACKAGE)
    package.__path__ = [str(_ROOT)]  # type: ignore[attr-defined]
    sys.modules[PACKAGE] = package


def load(name: str, *, api_stubs: bool = False) -> types.ModuleType:
    """Import ``custom_components/deepseek_conversation/<name>.py``.

    Set ``api_stubs`` for a module that imports the OpenAI SDK or Home
    Assistant's conversation component at module level; see the module
    docstring for what that buys and what it costs.
    """
    _ensure_homeassistant()
    if api_stubs:
        _ensure_api_stubs()
    _ensure_package()
    return importlib.import_module(f"{PACKAGE}.{name}")
