# Release candidate engineering

Version 1.1.3 is a follow-up hardening milestone. A merge is not a publication. This work
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

The supported SciPy floor and minimum-dependency CI are now **1.11.1**. Upstream
identifies that patch as the [licensing fix for 1.11.0](https://docs.scipy.org/doc/scipy-1.11.1/release/1.11.1-notes.html).
The yanked 1.11.0 release is no longer permitted by QFin dependency metadata or
installed by the minimum-version job. QFin does not vendor SciPy; this technical
dependency correction does not choose QFin's own license. The old warning is
retained as [historical evidence](history/hardening-1.1.2/minimum-dependency-warning.txt).

Windows wheel repair uses `tools/repair_windows_wheel.py` to select an AMD64
MSVC C++ runtime from an installed Visual Studio redistributable directory,
rather than accepting a stale DLL on `PATH`. It checks the extension and runtime
PE linker versions, requires the runtime to be at least as new, passes its path
explicitly to delvewheel, treats repair warnings as errors, and inspects the
runtime bundled in the repaired wheel. Missing, wrong-architecture, older, or
unverifiable runtimes stop the build. Runtime provenance and hashes are retained
with the wheel artifacts. This addresses the observed 14.51/14.40 mismatch;
it does not certify every host's other loaded DLLs or bypass redistribution review.
See Microsoft's [runtime compatibility restriction](https://learn.microsoft.com/en-us/cpp/porting/binary-compat-2015-2017?view=msvc-170)
and [PE header specification](https://learn.microsoft.com/en-us/windows/win32/debug/pe-format).

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
