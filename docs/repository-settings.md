# Main protection and repository settings

On 2026-09-09, main was at
`3dba3b3803f54891d12ad2c138b4dcc276de9d42` and GitHub reported `protected=false`.
The connected GitHub App excludes repository administration, so protection could
not be applied by this hardening task. CI itself is configured and exercised;
repository enforcement requires a maintainer with administration rights.

In **Settings → Rules → Rulesets**, create an active branch ruleset targeting
`refs/heads/main` (or configure equivalent **Branches → Branch protection**):

1. Require a pull request before merging. Prefer at least one independent review
   when another maintainer is available; do not set an impossible approval rule
   for a single-maintainer repository without an agreed review process.
2. Require status checks and require the branch to be up to date. Add the exact
   checks below from the CI workflow; select the GitHub Actions integration as the
   source where supported.
3. Block force pushes and branch deletion. Apply enforcement to administrators
   and leave ordinary bypass actors empty. Any exceptional emergency bypass should
   have an explicit maintainer process.
4. Require conversation resolution. Keep build/test permissions separate from
   any future release publishing permissions.

Required checks observed on PR #14:

- `Ruff and strict mypy`
- `native parity and strict C++ warnings`
- `native ASan and UBSan`
- `minimum dependencies / Python 3.11`
- `sdist clean install`
- `representative examples`
- `tests / ubuntu-latest / Python 3.11`
- `tests / ubuntu-latest / Python 3.12`
- `tests / ubuntu-latest / Python 3.13`
- `tests / macos-latest / Python 3.12`
- `tests / windows-latest / Python 3.12`
- `wheel / ubuntu-latest / Python 3.11`
- `wheel / ubuntu-latest / Python 3.12`
- `wheel / ubuntu-latest / Python 3.13`
- `wheel / macos-latest / Python 3.12`
- `wheel / windows-latest / Python 3.12`

The performance evidence workflow is scheduled/manual and should not become a
noisy timing gate. Correctness remains strictly gated. CI has explicit
`contents: read`, checkout credential persistence disabled, and immutable SHA
pins for [checkout v7.0.1](https://github.com/actions/checkout/releases/tag/v7.0.1),
[setup-python v7.0.0](https://github.com/actions/setup-python/releases/tag/v7.0.0)
and [upload-artifact v7.0.1](https://github.com/actions/upload-artifact/releases/tag/v7.0.1).
These use Node 24 and were checked against the official action metadata.

No PyPI upload workflow exists or is added. Do not infer a package publication
from a merge. Before any future publication, resolve licensing and use a separate
least-privilege release workflow with trusted publishing and provenance as
appropriate. Wheel checksums accompany the CI artifacts.
