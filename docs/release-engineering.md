# Release candidate engineering

Version 1.1.2 is a hardening milestone. A merge is not a publication. This work
does not create a tag, GitHub Release or PyPI upload. The repository has no license
file or declared license metadata; the owner must choose licensing and obtain any
necessary legal review before external distribution/adoption. No license has been
invented or restored by this task.

CI builds editable installs, ordinary wheels, a clean-install source distribution,
and nine portable wheels: CPython 3.11/3.12/3.13 on manylinux x86_64, macOS arm64
and Windows AMD64. Each portable wheel is repaired by cibuildwheel, installed into
an isolated environment, and exercised through the public smoke example, malformed
native tests and checked-in independent references. Core wheel tests also prove
PennyLane, Qiskit and QuantLib are absent. Optional imports remain isolated from
the NumPy/SciPy core. macOS Intel, Linux ARM, musllinux, free-threaded CPython and
unlisted interpreters are not validated by this matrix.

The Windows wheel test dependency uses an exact Hypothesis pin because Windows
command processing can interpret `<`/`>` in unquoted version ranges as file
redirection. The development extra retains its normal supported range.

`release-validation.yml` is manually dispatched with an exact expected version.
It checks metadata and any existing tag, runs the complete reusable CI workflow,
then downloads the already-tested portable wheels and sdist. `release_artifacts.py`
verifies every SHA-256, rejects duplicate/unsafe names, requires nine wheels and one
sdist, and copies those exact bytes into the candidate bundle. It never rebuilds
the candidate after testing. A separate job requests only the permissions needed
to attest those bytes and archives them. This workflow has no publication job.

Ordinary CI has `contents: read`, no persisted checkout credentials, SHA-pinned
actions and no publishing credentials. Attestation alone requests `id-token: write`
and `attestations: write`. Future publishing should consume the verified bundle,
check hashes/attestations, use a protected environment and trusted publishing, and
require the owner's separate release authorization. Do not upload a fresh rebuild
under an already-tested version.

The wheel selectors and isolated test semantics follow the
[cibuildwheel options](https://cibuildwheel.pypa.io/en/stable/options/).
Attestation follows [GitHub artifact provenance](https://docs.github.com/en/actions/concepts/security/artifact-attestations).
These describe the configured mechanism; the hardening report distinguishes actual
CI artifacts from any manual workflow or attestation step not executed in this task.
