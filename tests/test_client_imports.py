"""What importing ``client.py`` must already have taken out of the OpenAI SDK.

The SDK resolves its resource namespaces lazily: ``client.chat`` is a
``cached_property`` that imports ``openai.resources.chat.chat`` on first use,
and from openai 3.26 - the build Home Assistant 2026.11 pins - that package
resolves the submodule through ``importlib.import_module``. This integration
first uses it for the streamed completion in ``chat_session.py``, which runs in
the event loop, and Home Assistant reports every ``import_module`` there as a
blocking call - once per restart, in every user's log (#47).

``client.py`` closes that by importing the resource at module level, which Home
Assistant runs in an executor thread. Nothing in the SDK says where its imports
happen, so the cases below assert the property rather than the mechanism: once
that module is imported, walking the hot path imports nothing. The last case is
the control. It shows the hot path *would* import on an SDK that defers its
resources, and skips on one that does not, so an SDK that moved the work again
cannot leave the other two passing for the wrong reason.

Each case runs in its own interpreter, because what is asserted is the state of
``sys.modules`` at one moment and the suites around this one import freely. They
need the SDK, and Home Assistant to import ``client.py`` at all, so they skip in
CI; ``hatest.bat`` builds both.
"""

from __future__ import annotations

import importlib.util
import pathlib
import subprocess
import sys
import textwrap
import unittest

_TESTS_DIR = str(pathlib.Path(__file__).resolve().parent)

_SETUP_HINT = (
    "needs the OpenAI SDK and Home Assistant; build them with scripts/ha_testenv.py"
)


def _installed(module: str) -> bool:
    """Whether ``module`` can be imported for real."""
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


requires_sdk = unittest.skipUnless(
    _installed("openai") and _installed("homeassistant.helpers.httpx_client"),
    _SETUP_HINT,
)

#: Runs in the child before the body does. ``openai`` itself is imported here,
#: ahead of any recording, so what a case measures is the SDK resolving a
#: resource and never the SDK being imported in the first place.
_RECORDER = '''
import importlib
import sys

import openai

_real_import_module = importlib.import_module
_calls = []


def _record(name, package=None):
    _calls.append(f"{package or ''}{name}")
    return _real_import_module(name, package)


def sdk_imports(action):
    """The openai modules ``action`` resolved through ``importlib``."""
    importlib.import_module = _record
    try:
        _calls.clear()
        action()
    finally:
        # Back in place before the assertion runs, so a failing case can still
        # import whatever it needs to report itself.
        importlib.import_module = _real_import_module
    return [name for name in _calls if "openai" in name]


def hot_path():
    """Reach the call ``chat_session.py`` makes, without making it."""
    client = openai.AsyncOpenAI(api_key="t", base_url="https://example.invalid/v1")
    return client.with_options(timeout=5.0).chat.completions
'''


def _run(body: str) -> subprocess.CompletedProcess[str]:
    script = (
        textwrap.dedent(
            f"""
            import sys
            sys.path.insert(0, {_TESTS_DIR!r})
            import integration_modules as im
            """
        )
        + _RECORDER
        + textwrap.dedent(body)
    )
    return subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=False
    )


@requires_sdk
class TestTheChatResourceIsImportedUpFront(unittest.TestCase):
    def test_importing_the_client_module_imports_it(self) -> None:
        result = _run(
            """
            im.load("client")
            assert "openai.resources.chat.chat" in sys.modules, sorted(
                name for name in sys.modules if name.startswith("openai.resources")
            )
            print("ok")
            """
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_the_streamed_completion_then_imports_nothing(self) -> None:
        result = _run(
            """
            im.load("client")
            left = sdk_imports(hot_path)
            assert not left, f"imported inside the event loop: {left}"
            print("ok")
            """
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_without_it_the_hot_path_is_what_would_import(self) -> None:
        """The control: an SDK that defers its resources must defer them here.

        It is the hot path alone, with ``client.py`` never imported. An SDK that
        imports its resources eagerly has nothing left to defer and nothing for
        the fix to move, so the case skips rather than asserting against it.
        """
        result = _run(
            """
            print("defers" if sdk_imports(hot_path) else "eager")
            """
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        reported = result.stdout.strip()
        if reported == "eager":
            self.skipTest("this openai build imports its resources eagerly")
        self.assertEqual(reported, "defers")


if __name__ == "__main__":
    unittest.main()
