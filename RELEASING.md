# Versioning and releases

l9s uses [semantic versioning](https://semver.org): `MAJOR.MINOR.PATCH`, in `VERSION`.

## Which number to bump

Before 1.0 (now):

| Change | Bump | Example |
|---|---|---|
| A new feature, command, setting or deployment | minor | 0.14.5 → 0.15.0 (l9s turbo) |
| A fix, a doc change that ships, a dependency update | patch | 0.14.4 → 0.14.5 |
| Something that breaks a setup, a command or a file l9s keeps | minor, with a **Breaking:** line in the changelog | |

From 1.0, a breaking change needs a major bump, and something that goes away is announced (changelog and
`l9s doctor`) at least one minor release before it's removed. See "Compatibility" in the README.

## How a change ships

1. Work on a branch and open a pull request to `main`. CI runs on it (lint, unit tests, the installer).
2. Add what changed for users to `CHANGELOG.md` under `## Unreleased` at the top, while you work.
3. To release with the PR: set `VERSION` to the new version and rename `## Unreleased` to `## <version>`.
   `tests/unit/test_release.py` fails if the two don't match, or if versions in the changelog don't go down.
4. Merging a commit that changes `VERSION` to `main` starts `.github/workflows/release.yml`: CI, then the
   hardware gate (the live suite on a GB10 and a Mac), then a **beta** release `v<version>` with that changelog
   section as its notes.
5. `.github/workflows/promote.yml` makes a beta **stable** (GitHub's "latest": what new installs and automatic
   updates get) after 3 days without an open issue labelled `regression` that names its version.

Changes merged without a `VERSION` change wait under `## Unreleased` for the next release.
Dependency updates (`.github/workflows/deps.yml`) bump the patch version themselves.
