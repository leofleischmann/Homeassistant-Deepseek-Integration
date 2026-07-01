# Contributing

Thanks for helping improve DeepSeek Conversation.

## Branches

| Branch | Purpose |
|--------|---------|
| **`dev`** | All day-to-day development and pull requests |
| **`main`** | Release branch only; updated by merging `dev` when a version ships |

**Open pull requests against `dev`, not `main`.**

`main` is merged from `dev` after testing. Pushing to `main` with a changed `manifest.json` triggers the [release workflow](.github/workflows/release.yml) (Git tag, GitHub Release, `deepseek_conversation.zip` for HACS).

## Workflow

1. Fork the repo (external contributors) or branch from `dev`.
2. Make your changes on `dev` (or a feature branch based on `dev`).
3. Ensure [CI](.github/workflows/ci.yml) passes (HACS validation + hassfest).
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

**Examples from this project:**

```
feat: trim Assist history by user rounds for context management
fix: allow voluptuous-openapi 0.4.x for Home Assistant 2026.7
docs: rewrite README with clearer scope and without vision
chore: add HACS zip_release asset for GitHub download counts
```

## Versioning and changelog

Releases are driven by `custom_components/deepseek_conversation/manifest.json`.

Maintainers use `bump.bat` / `scripts/bump.py` on `dev` to bump the version and draft `CHANGELOG.md`, then merge `dev` into `main` for the GitHub release.

Contributors do not need to bump the version unless asked; describe user-visible changes in the PR instead.

## Code expectations

- Match existing style and structure under `custom_components/deepseek_conversation/`.
- Keep changes focused; avoid unrelated refactors in the same PR.
- Debug logs for browser/HA troubleshooting: prefix with `[Debug <scriptname>]:` where applicable.
- Update translations (`translations/`) when adding or changing user-facing strings in the UI or services.

## HACS and releases

- `hacs.json` defines HACS metadata; release zips are built by `scripts/build_hacs_zip.py` in CI.
- Do not commit `deepseek_conversation.zip` (it is gitignored).

## Questions

Open a [GitHub issue](https://github.com/leofleischmann/Homeassistant-Deepseek-Integration/issues) for bugs or feature ideas before large changes.
