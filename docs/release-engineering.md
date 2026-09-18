# Release candidate engineering

Version 1.1.2.post1 updates installation and packaging only. A merge is not a publication. This work
does not create a tag, GitHub Release or PyPI upload. The repository has no license
file or declared license metadata; the owner must choose licensing and obtain any
necessary legal review before external distribution/adoption. No license has been
invented or restored by this task.

CI builds editable installs, ordinary wheels, a clean-install source distribution,
a universal pure-Python wheel, and nine portable native wheels:
CPython 3.11/3.12/3.13 on manylinux x86_64, macOS arm64
and Windows AMD64. Each portable wheel is repaired by cibuildwheel, installed into
an isolated environment, and exercised through the public smoke example, malformed
native tests and checked-in independent references. Core wheel tests also prove
PennyLane, Qiskit and QuantLib are absent. Optional imports remain isolated from
the NumPy/SciPy core. macOS Intel, Linux ARM, musllinux, free-threaded CPython and
unlisted interpreters are not validated by this matrix.

`python-package.yml` deliberately supplies an invalid C++ compiler, builds the
sdist and then a wheel from it, and installs that wheel in an isolated environment
on Linux, macOS and Windows. It checks every package module, metadata version,
the typing marker, both CLI entry points, representative existing calculations,
and existing financial/API regression tests. It requires the extension and all
optional packages to be absent. The ordinary smoke requires the extension to be
present. All native jobs set `QFIN_REQUIRE_NATIVE=1`, and a separate failing-build
check proves that this requirement disables automatic fallback. Neither a source
tree nor an editable install can satisfy the installed-wheel smoke check.
After the core-only checks, the same universal wheel's `[quantum]` extra is
installed and its complete financial-to-quantum workflow is tested on local
PennyLane/Lightning simulators, including the executable European-option example.

The Windows wheel test dependency uses an exact Hypothesis pin because Windows
command processing can interpret `<`/`>` in unquoted version ranges as file
redirection. The development extra retains its normal supported range.

The declared and tested SciPy floor is now 1.11.1, excluding the yanked 1.11.0
release. The old minimum-install warning remains historical evidence in the
[archive](history/hardening-1.1.2/minimum-dependency-warning.txt).

Windows portable builds use the v143 compiler toolset and explicitly select a
compatible installed MSVC redistributable. The repair script verifies the bundled
runtime's architecture/version, records hashes, and owns extraction cleanup. These
packaging-only fixes were taken from the pending hardening follow-up without its
financial or quantum-risk calculation changes.

`release-validation.yml` is manually dispatched with an exact expected version.
It checks metadata and any existing tag, runs the complete reusable CI workflow,
then downloads the already-tested native wheels, universal wheel and sdist. `release_artifacts.py`
verifies every SHA-256, rejects duplicate/unsafe names, requires ten wheels and one
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
