#!/usr/bin/env python3
"""Build ``.venv-ha``: a real Home Assistant to run the integration tests against.

The unit suite runs on a bare Python because CI has no Home Assistant, which is
what makes it fast - but it also means the stubs, not core, decide what the
integration sees. A change inside core is invisible to it, and that is exactly
the kind of change that has broken every request before.

So this builds the other half: an environment with core actually installed, in
which ``tests/test_real_home_assistant.py`` stops skipping and checks the
integration against the real ``llm`` helpers, the real schema library and the
real ``ToolInput``.

Nothing here is pinned to one Home Assistant release. What to install is read
from the manifests: the integration's own ``requirements``, and then the
``requirements`` of the core components it depends on, taken from the core that
was just installed. Ask for a different version and it brings its own.

Usage (or via ``hatest.bat``)::

    python scripts/ha_testenv.py              # latest Home Assistant
    python scripts/ha_testenv.py 2026.9.0     # any release

The environment lands in ``.venv-ha`` next to the repo and is git-ignored;
delete that directory to start over. It is roughly a gigabyte.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import venv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VENV = ROOT / ".venv-ha"
MANIFEST = ROOT / "custom_components" / "deepseek_conversation" / "manifest.json"

#: Packages a core component declares that cannot be built here. They are
#: skipped up front rather than failing the run: the tests that need them skip
#: too, and the rest - which is what this environment exists for - still runs.
#: Anything else that fails to install is reported and also skipped.
UNBUILDABLE = {
    # Compiled wheels with no Windows build; only assist_pipeline's audio path
    # needs them, which these tests never reach.
    "pymicro-vad",
    "pyspeex-noise",
}


def python_in_venv() -> Path:
    """The interpreter inside the environment, per platform."""
    if sys.platform == "win32":
        return VENV / "Scripts" / "python.exe"
    return VENV / "bin" / "python"


def run(*args: str) -> None:
    print(f"  $ {' '.join(args[1:])}")
    subprocess.run(args, check=True)


def capture(interpreter: Path, code: str) -> str:
    return subprocess.run(
        [str(interpreter), "-c", code],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def integration_manifest() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def component_requirements(components: Path, roots: list[str]) -> list[str]:
    """What the core components this integration depends on need, transitively.

    Home Assistant declares requirements per component rather than in its own
    dependencies, so a plain ``pip install homeassistant`` cannot even import
    ``homeassistant.components.conversation``. Reading it back out of the
    installed core is what keeps this correct across releases: the set follows
    whatever that version declares, including components that did not exist yet
    or have since been split up.

    Soft dependencies count too. They are ordinary imports once present - it is
    ``ai_task``'s ``after_dependencies`` that reaches ``camera``, and importing
    the platform pulls that in whether or not the test cares about cameras.
    """
    found: list[str] = []
    seen: set[str] = set()
    queue = list(roots)
    while queue:
        domain = queue.pop()
        if domain in seen:
            continue
        seen.add(domain)
        manifest = components / domain / "manifest.json"
        if not manifest.exists():
            continue
        data = json.loads(manifest.read_text(encoding="utf-8"))
        found.extend(data.get("requirements", []))
        queue.extend(data.get("dependencies", []))
        queue.extend(data.get("after_dependencies", []))
    return sorted(set(found))


def install(interpreter: Path, packages: list[str], *, what: str) -> list[str]:
    """Install ``packages``, and return the ones that would not install.

    One pip call for the common case; on failure each package is retried alone,
    so a single unbuildable wheel costs that package rather than the whole
    environment.
    """
    if not packages:
        return []
    print(f"Installing {what} ({len(packages)}) ...")
    attempt = subprocess.run(
        [str(interpreter), "-m", "pip", "install", "-q", *packages]
    )
    if attempt.returncode == 0:
        return []

    print("  one of them failed; retrying individually ...")
    failed: list[str] = []
    for package in packages:
        one = subprocess.run(
            [str(interpreter), "-m", "pip", "install", "-q", package],
            capture_output=True,
            text=True,
        )
        if one.returncode != 0:
            failed.append(package)
    return failed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "version",
        nargs="?",
        help="Home Assistant version, e.g. 2026.9.0 (default: latest)",
    )
    args = parser.parse_args()

    interpreter = python_in_venv()
    if not interpreter.exists():
        print(f"Creating {VENV} ...")
        venv.EnvBuilder(with_pip=True).create(VENV)
    else:
        print(f"Reusing {VENV}")

    core = f"homeassistant=={args.version}" if args.version else "homeassistant"
    run(str(interpreter), "-m", "pip", "install", "-q", "--upgrade", "pip")
    print(f"Installing {core} ...")
    run(str(interpreter), "-m", "pip", "install", "-q", core)

    manifest = integration_manifest()
    installed = capture(
        interpreter, "from homeassistant.const import __version__; print(__version__)"
    )
    components = Path(
        capture(
            interpreter,
            "import homeassistant, pathlib;"
            " print(pathlib.Path(homeassistant.__file__).parent)",
        )
    ) / "components"

    failed = install(
        interpreter, list(manifest.get("requirements", [])), what="the integration's own"
    )
    # Both the hard and the soft dependencies: the tests import the conversation
    # and ai_task chains, and after_dependencies is where those are declared.
    roots = [
        *manifest.get("dependencies", []),
        *manifest.get("after_dependencies", []),
    ]
    wanted = [
        package
        for package in component_requirements(components, roots)
        if package.split("==")[0].split(">")[0].split("[")[0].strip() not in UNBUILDABLE
    ]
    failed += install(interpreter, wanted, what=f"what {', '.join(roots)} declare")

    print(f"\nHome Assistant {installed} ready in {VENV}")
    if failed:
        print(
            "Could not install, so anything needing it skips:\n  "
            + "\n  ".join(failed)
        )
    print(f"Run the suite against it:\n  {interpreter} -m unittest discover -s tests")
    return 0


if __name__ == "__main__":
    sys.exit(main())
