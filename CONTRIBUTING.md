# Contributing

Thanks for helping improve DeepSeek Conversation.

## Branches

**Open pull requests against `dev`, not `main`.** `dev` is where development happens; `main` is the release branch, merged from `dev` after testing. Pushing to `main` with a changed `manifest.json` triggers the [release workflow](.github/workflows/release.yml) (Git tag, GitHub Release, `deepseek_conversation.zip` for HACS).

## Workflow

1. Fork the repo (external contributors) or branch from `dev`.
2. Make your changes on `dev` (or a feature branch based on `dev`).
3. Ensure [CI](.github/workflows/ci.yml) passes (HACS validation + hassfest + unit tests).
   Run them locally with `python -m unittest discover -s tests` — no Home Assistant needed.

   Those run against stubs and so cannot see a change in core itself. For anything
   touching the `llm` helpers, tool schemas or `ToolInput`, run `hatest.bat` as well:
   it builds `.venv-ha` (git-ignored, ~1 GB) with Home Assistant installed and runs the
   same suite, at which point `tests/test_real_home_assistant.py` stops skipping. Add a
   version to pin one (`hatest.bat 2026.9.0`), or `skip-setup` to reuse the environment.
4. Open a PR targeting **`dev`** with a clear title and short description.
5. Maintainers merge to `dev`, test on a Home Assistant instance, then open a release PR **`dev` → `main`** when ready to ship.

## Commit messages

Use [Conventional Commits](https://www.conventionalcommits.org/) in English:

```
<type>: <short summary in imperative mood>
```

**Types used in this repo:**

| Type | When to use |
|------|-------------|
| `feat` | New user-facing behaviour or capability |
| `fix` | Bug fix |
| `docs` | README, CHANGELOG, comments only |
| `chore` | Tooling, CI, scripts, housekeeping (no product change) |

**Rules:**

- One line for the subject; add a body only when the *why* is not obvious.
- Lowercase after the colon; no period at the end.
- Describe the change, not the file list (`fix: reject vision on official API`, not `fix: vision.py`).
- Imperative mood: `add`, `fix`, `remove`, not `added` / `fixes`.
- Avoid `@` in messages (GitHub may treat it as a user mention).

**Examples:**

```
feat: trim Assist history by user rounds for context management
fix: allow voluptuous-openapi 0.4.x for Home Assistant 2026.7
chore: add HACS zip_release asset for GitHub download counts
```

## Versioning and changelog

Releases are driven by `custom_components/deepseek_conversation/manifest.json`. Maintainers use `bump.bat` / `scripts/bump.py` on `dev`, then merge into `main`. Contributors do not need to bump the version — describe user-visible changes in the PR instead.

## Code expectations

- Match existing style and structure under `custom_components/deepseek_conversation/`.
- Keep changes focused; avoid unrelated refactors in the same PR.
- Debug logs for browser/HA troubleshooting: prefix with `[Debug <scriptname>]:` where applicable.
- Update translations (`translations/`) when adding or changing user-facing strings in the UI or services.
- `hacs.json` defines HACS metadata; release zips are built by `scripts/build_hacs_zip.py` in CI, so do not commit `deepseek_conversation.zip` (it is gitignored).

## Questions

Open a [GitHub issue](https://github.com/leofleischmann/Homeassistant-Deepseek-Integration/issues) for bugs or feature ideas before large changes.
