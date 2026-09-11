# 1.1.2 verification evidence

These records support [the hardening report](../../hardening-1.1.2.md),
[statistical interpretation](../../statistical-validation.md), and
[performance conclusions](../../performance.md). They retain observed values;
raw measurements are not silently relabelled as later clean builds.

| Files | Scope and provenance |
| --- | --- |
| `baseline.json`, `coverage-summary.json` | Untouched starting main `f9d4be6`, final local source, and clean checkpoint CI coverage. Coverage keeps per-file summaries; raw CI coverage remains in its Actions artifact. |
| `ci-46ef9.json` | All 20 jobs, artifact identifiers, archive digests and selected log evidence for the implementation checkpoint. |
| `minimum-dependency-warning.txt` | Inspected final CI warning that the intentionally tested SciPy 1.11.0 compatibility floor is yanked for License Violation; test success is not licensing clearance. |
| `artifact-SHA256SUMS-46ef9.txt`, `artifact-contexts-46ef9.json` | Nine portable wheels and one sdist actually downloaded and verified/collected without rebuild, at `46ef9f8`. This is not a later-build manifest or a signed attestation. |
| `compile-commands.json`, `source-digests.json` | Actual baseline/final compiler commands and a SHA-256 inventory of final financial Python/C++ sources. The last financial change is the revalidated NumPy liability auto policy. |
| `reference-errors.json` | 1073 independent stored configurations, 1454 engine-expanded executions and 53 unit/engine-specific error summaries; includes exact dates/schedules and QuantLib dated bonds. The 26 inline extensions are additionally gated by pytest. |
| `mutations.json` | Isolated unmodified control and all nine killed numerical mutations. |
| `ruff.txt`, `mypy.txt`, `coverage-gates.txt`, `property-stress.txt` | Actual final local gate output. Complete local suite: 2232 passed, two optional Qiskit skips; clean Linux CI includes those two tests. |
| `statistical-workflows.json` | 80 actual-circuit cells × 100 repetitions. Captured with dirty parent `7f683dd` and pre-bump 1.1.1 metadata. |
| `statistical-ambiguity.json` | 16 actual-circuit cells × 100 repetitions; dirty parent `ef23f99`, 1.1.2 metadata. Includes the 0/100 conditional CVaR coverage failure. |
| `mlae.json` | Separate fixed-binomial 45-cell × 200-fit study and five-repetition optimizer timing; not full sampled workflows. |
| `baseline-native.json`, `current-native.json` | 33-row 1.1.1/1.1.2 public matrices on the same Intel host, three repetitions. Not every row is a NumPy/native comparison; reference labels are authoritative. |
| `paired-benchmark.json` | Targeted baseline/current/current/baseline process recheck, five timings and one warm-up per block, ten timed samples per version. |
| `large-liability-dispatch.json`, `current-key-rate.json` | Seven-repetition large-liability check and three-repetition full key-rate sweep used for final auto-policy decisions. |
| `baseline-memory.json`, `current-memory.json` | Seventeen fresh-process workload/engine/chunk combinations per version; raw timing samples and peak RSS include imports and inputs. |
| `implementation-alternatives.json` | Five-repetition par-yield/reduction alternatives and numerical cancellation probes; not an incremental release speedup claim. |

The paired recheck uses `tools/memory_benchmark.py::workload` for 10000 bonds,
2000 rate scenarios with chunk 16, and 1000×20 ALM paths with chunk 256, each in
NumPy and native engines. Its dated workload prices 1000 references to a 4%
bond issued 2020-01-31 and maturing 2040-01-31, unadjusted EOM, valued at
2024-04-30 on `[0,10,60]` nodes and `[.02,.03,.04]` zero rates. Liability workloads
use the same curve, one 20-year 4% bond and equally spaced times from .01 to 60
with unit amounts; counts are 2048/4096/8192 in the paired test and
32768/65536/262144 in the additional large check. Inputs are built before timing.

Use separate installations of baseline and current source; run the public
workloads in the recorded alternating order with OMP/OpenBLAS/MKL threads set to
one and `PYTHONHASHSEED=0`. Benchmarking a single shared machine cannot establish
portable significance. The final source digest inventory lets later readers
check that measured numerical code corresponds to the delivered implementation;
the statistical records preserve their own earlier development-state labels.

The final immutable branch/merge SHAs and final-head/post-merge CI are recorded
in [PR #15](https://github.com/venkatkota2/QFin/pull/15). No release, tag, PyPI
upload, license grant or administrator protection mutation is represented by
these evidence files.
