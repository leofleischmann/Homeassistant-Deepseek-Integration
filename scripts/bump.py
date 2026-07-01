#!/usr/bin/env python3
"""Bump manifest version and draft CHANGELOG.md for the next release.

Used by bump.bat. Release workflow (.github/workflows/release.yml) reads the
matching ## [version] section from CHANGELOG.md when creating GitHub releases.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = REPO_ROOT / "custom_components" / "deepseek_conversation" / "manifest.json"
CHANGELOG = REPO_ROOT / "CHANGELOG.md"
VERSION_RE = re.compile(r"^\d+\.\d+\.\d+([-.][\w.]+)?$")


def _run_git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args],
        cwd=REPO_ROOT,
        text=True,
        stderr=subprocess.STDOUT,
    ).strip()


def _last_tag() -> str | None:
    try:
        return _run_git("describe", "--tags", "--abbrev=0")
    except subprocess.CalledProcessError:
        return None


def _commits_since(tag: str | None) -> list[str]:
    log_range = f"{tag}..HEAD" if tag else "HEAD"
    try:
        out = _run_git(
            "log",
            "--no-merges",
            "--pretty=format:%s (%h)",
            log_range,
        )
    except subprocess.CalledProcessError:
        return []
    if not out:
        return []
    return [line for line in out.splitlines() if line.strip()]


def _read_manifest_version() -> str:
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    return str(data["version"])


def _write_manifest_version(version: str) -> None:
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    data["version"] = version
    MANIFEST.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _section_heading(version: str) -> str:
    return f"## [{version}]"


def _build_section(version: str, commits: list[str]) -> str:
    lines = [f"## [{version}] - {date.today().isoformat()}", ""]
    if commits:
        lines.extend(f"- {commit}" for commit in commits)
    else:
        lines.append("- (keine Commits seit letztem Release-Tag)")
    lines.append("")
    return "\n".join(lines)


def _changelog_has_version(version: str) -> bool:
    if not CHANGELOG.exists():
        return False
    content = CHANGELOG.read_text(encoding="utf-8")
    return bool(
        re.search(
            rf"^{re.escape(_section_heading(version))}\b",
            content,
            flags=re.MULTILINE,
        )
    )


def _merge_changelog(new_section: str) -> None:
    header = "# Changelog\n\n"
    intro = "Alle wesentlichen Änderungen an dieser Integration.\n\n"

    if not CHANGELOG.exists():
        CHANGELOG.write_text(header + intro + new_section, encoding="utf-8")
        return

    content = CHANGELOG.read_text(encoding="utf-8")
    match = re.search(r"^## \[", content, flags=re.MULTILINE)
    if match:
        prefix = content[: match.start()]
        rest = content[match.start() :]
    else:
        prefix = content if content.endswith("\n") else content + "\n"
        rest = ""

    if not prefix.strip():
        prefix = header + intro
    elif not prefix.lstrip().startswith("# Changelog"):
        prefix = header + intro + prefix

    CHANGELOG.write_text(prefix + new_section + rest, encoding="utf-8")


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: bump.bat <version>", file=sys.stderr)
        print("Example: bump.bat 1.3.1", file=sys.stderr)
        return 1

    version = sys.argv[1].strip()
    if not VERSION_RE.match(version):
        print(
            f"Ungültige Version: {version!r} (erwartet z. B. 1.3.1)",
            file=sys.stderr,
        )
        return 1

    if not MANIFEST.is_file():
        print(f"manifest.json nicht gefunden: {MANIFEST}", file=sys.stderr)
        return 1

    try:
        _run_git("rev-parse", "--is-inside-work-tree")
    except subprocess.CalledProcessError:
        print("Kein Git-Repository.", file=sys.stderr)
        return 1

    old_version = _read_manifest_version()
    if version == old_version:
        print(
            f"Warnung: Version ist bereits {old_version}; manifest wird trotzdem aktualisiert."
        )

    if _changelog_has_version(version):
        print(
            f"Abbruch: CHANGELOG.md enthält bereits {_section_heading(version)}.",
            file=sys.stderr,
        )
        return 1

    tag = _last_tag()
    commits = _commits_since(tag)

    _write_manifest_version(version)
    section = _build_section(version, commits)
    _merge_changelog(section)

    print(f"[Debug bump]: manifest {old_version} -> {version}")
    print(f"[Debug bump]: letzter Tag: {tag or '(keiner)'}")
    print(f"[Debug bump]: {len(commits)} Commit(s) in CHANGELOG übernommen")
    print()
    print("Fertig:")
    print(f"  - {MANIFEST.relative_to(REPO_ROOT)}")
    print(f"  - {CHANGELOG.relative_to(REPO_ROOT)}")
    print()
    print("CHANGELOG.md bei Bedarf anpassen, dann:")
    print("  git add .")
    print(f'  git commit -m "chore: release {version}"')
    print("  git push origin main")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
