"""Build deepseek_conversation.zip for HACS zip_release (see hacs.json, release.yml).

HACS extracts zip_release archives directly into custom_components/<domain>/.
The zip must therefore contain integration files at the archive root (manifest.json, …),
not wrapped in an extra deepseek_conversation/ folder.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "custom_components" / "deepseek_conversation"
OUT = REPO_ROOT / "deepseek_conversation.zip"


def main() -> None:
    if not SRC.is_dir():
        raise SystemExit(f"Missing integration directory: {SRC}")

    if OUT.exists():
        OUT.unlink()

    count = 0
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(SRC.rglob("*")):
            if path.is_dir():
                continue
            if "__pycache__" in path.parts or path.suffix == ".pyc":
                continue
            arc = path.relative_to(SRC).as_posix()
            zf.write(path, arc)
            count += 1

    print(f"[Debug build_hacs_zip]: wrote {OUT.name} ({OUT.stat().st_size} bytes, {count} files)")
    _validate_zip(OUT)


def _validate_zip(path: Path) -> None:
    """Fail fast if the archive does not match HACS zip_release expectations."""
    required = ("manifest.json", "__init__.py")
    banned_prefixes = ("custom_components/", "deepseek_conversation/")
    with zipfile.ZipFile(path) as zf:
        names = zf.namelist()
        missing = [entry for entry in required if entry not in names]
        if missing:
            raise SystemExit(f"Zip missing required paths at archive root: {missing}")
        if any(
            name.startswith(prefix) for name in names for prefix in banned_prefixes
        ):
            raise SystemExit(
                "Zip must contain integration files at archive root; "
                "HACS already extracts into custom_components/<domain>/"
            )


if __name__ == "__main__":
    main()
