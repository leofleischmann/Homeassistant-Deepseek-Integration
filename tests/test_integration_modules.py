"""Tests for the loader every other test suite goes through.

Its failure mode is invisible in CI: CI has neither Home Assistant nor the
OpenAI SDK installed, so the branches that matter on a contributor's machine -
the ones that must leave a real installation alone - never run there. Each case
therefore runs in its own interpreter, both to pin down what is installed and to
keep the stubs out of the other suites' ``sys.modules``.
"""

from __future__ import annotations

import pathlib
import subprocess
import sys
import textwrap
import unittest

_TESTS_DIR = str(pathlib.Path(__file__).resolve().parent)


def _run(body: str) -> subprocess.CompletedProcess[str]:
    script = textwrap.dedent(
        f"""
        import sys
        sys.path.insert(0, {_TESTS_DIR!r})
        import integration_modules as im
        """
    ) + textwrap.dedent(body)
    return subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=False
    )


class TestLoader(unittest.TestCase):
    def test_pure_modules_load_without_home_assistant(self) -> None:
        result = _run(
            """
            options = im.load("options")
            assert options.coerce_max_tokens("2000") == 2000
            print("ok")
            """
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_api_stubs_let_a_streaming_module_load(self) -> None:
        result = _run(
            """
            module = im.load("stream_transform", api_stubs=True)
            assert module._stream_delta_text is not None
            print("ok")
            """
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_an_installed_home_assistant_is_left_alone(self) -> None:
        """The stubs must never shadow a real install, nor assume their own.

        On a machine that has Home Assistant, ``_ensure_homeassistant`` returns
        without creating anything - so the API stubs must not reach for a parent
        stub that does not exist, and must not stub what is really there.
        """
        result = _run(
            """
            im._installed = lambda name: True   # as if everything were installed
            im._ensure_homeassistant()
            im._ensure_api_stubs()
            assert "homeassistant" not in sys.modules, "stubbed over a real install"
            assert "openai" not in sys.modules, "stubbed over a real openai"
            print("ok")
            """
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
