# Main protection and repository settings

Audit on 2026-09-11: GitHub reported `main` at
`f9d4be69432ee813fccdde609420cdee9eb15580`, **`protected=false`**, with no required
status-check enforcement. Protection was not applied by this task. The connected
GitHub tools expose no repository-administration mutation; writing these
instructions does not enforce them. Recheck live settings after any owner change.

An administrator should create an active ruleset in **Settings → Rules →
Rulesets**, targeting `refs/heads/main`, or use equivalent branch protection:

1. Require a pull request before merging and require conversation resolution.
   Prefer at least one independent approval when an additional maintainer is
   available; agree a workable review policy for a single-maintainer repository.
2. Require all exact check names below, from the GitHub Actions integration, and
   require the branch to be up to date before merging.
3. Block force pushes and branch deletion. Apply enforcement to administrators;
   leave normal bypass actors empty. Define any emergency bypass separately.
4. Keep ordinary CI permissions read-only. Any future publisher must use a
   separate protected environment with owner-approved trusted publishing.

## Exact checks observed on PR #15

All 20 passed at the verified implementation checkpoint in
[run 34576612885](https://github.com/venkatkota2/QFin/actions/runs/34576612885).
The PR timeline records checks and head SHA for the final merge as well.

- `Ruff and strict mypy`
- `minimum dependencies / Python 3.11`
- `native ASan and UBSan / clang`
- `native ASan and UBSan / gcc`
- `native parity and strict C++ warnings`
- `portable binary wheels / cibuildwheel / macos-latest`
- `portable binary wheels / cibuildwheel / ubuntu-latest`
- `portable binary wheels / cibuildwheel / windows-latest`
- `representative examples`
- `sdist clean install`
- `tests / macos-latest / Python 3.12`
- `tests / ubuntu-latest / Python 3.11`
- `tests / ubuntu-latest / Python 3.12`
- `tests / ubuntu-latest / Python 3.13`
- `tests / windows-latest / Python 3.12`
- `wheel / macos-latest / Python 3.12`
- `wheel / ubuntu-latest / Python 3.11`
- `wheel / ubuntu-latest / Python 3.12`
- `wheel / ubuntu-latest / Python 3.13`
- `wheel / windows-latest / Python 3.12`

Performance and large statistical/mutation studies are manual/weekly evidence
workflows, not noisy per-PR timing gates. Ordinary CI uses `contents: read`,
SHA-pinned actions, and `persist-credentials: false`. Only the separate
release-candidate attestation job requests `id-token: write` and
`attestations: write`; no publication job exists.

The owner must separately choose licensing and obtain appropriate legal review;
this repository currently has no license grant or declared license metadata.
No PyPI publication, GitHub Release or tag was created by the hardening task.
The manual release-candidate workflow and its attestation step are configured,
but were not dispatched during this task. The tested wheel/sdist bytes and their
SHA-256 collection were actually validated; see [release engineering](release-engineering.md).
