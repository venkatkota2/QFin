# Performance and automatic engine policy — 1.1.3

## Follow-up: reuse ALM period-boundary valuations

The NumPy ALM path implementation now reuses each period's closing bond index and
short rate as the next period's opening values. It also reuses the initial bond
valuation. This removes redundant valuations without removing finite-result,
dimensional or memory checks. Native code and automatic dispatch are unchanged.

The paired comparison used untouched starting main
`f14de6598523ecefc79e925214a926cc720d3e79` (1.1.2) and the 1.1.3 implementation,
in baseline/candidate/candidate/baseline process order. Each block had one warm-up
and five measurements: ten timings per version/chunk. The workload was 1000
zero-rate-shock scenarios, 20 periods and 20 bonds, through the public
`project_paths(..., engine="numpy")` call. Construction was outside timed calls.
The same interpreter and dependencies were used: Python 3.12.14, NumPy 2.5.3,
SciPy 1.18.1, Linux 6.18.35 x86_64/glibc 2.39, Intel Xeon Platinum 8573C,
single OpenBLAS/MKL/OMP threads.

| Chunk | 1.1.2 median s | 1.1.3 median s | 1.1.3 min–max s | 1.1.3 std. dev. s | Median reduction |
| --- | ---: | ---: | --- | ---: | ---: |
| 16 | 0.645871 | 0.390445 | 0.367224–0.565238 | 0.053952 | 39.5% |
| 256 | 0.405097 | 0.204200 | 0.191750–0.212067 | 0.006587 | 49.6% |

All nine returned array SHA-256 digests match bit-for-bit across both versions,
both chunks and all four blocks. Regressions also check scenario/path parity,
chunk invariance, and exactly two initial plus two per-period present valuations.
These results describe this workload on this host, not a universal speedup or
evidence to change native crossover policy. No new peak-RSS claim is made.

The [raw paired record](history/hardening-1.1.3/paired-alm-reuse.json) retains its
actual candidate parent `f14de65` and dirty-tree flag. It records the imported
source path and ALM source-file hash for each block. Both source trees ran under
installed package metadata 1.1.3; the baseline label refers to the verified source
checkout, not that metadata. The candidate source hash matches the implementation
committed in `061c4ce`. The harness verifies the selected source to avoid an
editable-install finder silently importing the other checkout.

```bash
python tools/benchmark_alm_reuse.py --baseline /path/to/f14de65-checkout --output alm.json
```

## Retained 1.1.2 evidence and unchanged dispatch policy

The remaining measurements below belong to the 1.1.2 campaign. In particular,
the old NumPy ALM-path timing is superseded by the controlled follow-up above;
it must not be read as the current 1.1.3 timing. Other kernels and dispatch policy
were not changed in 1.1.3.

Correctness and explicit failure take precedence over timing. These measurements
include public API validation, buffer preparation, allocation and the Python/C++
boundary; model construction is outside timed calls. They do not establish a
universal C++ speedup. [The 1.1.1 report](history/performance-1.1.1.md) is historical.

## Environment and comparison design

The before/after runs used the untouched starting main
`f9d4be69432ee813fccdde609420cdee9eb15580` (1.1.1) and the hardened 1.1.2
implementation. Raw records retain their actual capture time and version. These
runs were on Linux 6.18.35 x86_64/glibc 2.39, Intel Xeon Platinum 8573C, Python
3.12.14, NumPy 2.5.2, SciPy 1.18.1, GCC 13.3.0, PennyLane 0.45.1 and Lightning
0.45.0. OpenBLAS/MKL/OMP thread counts were one. Earlier AMD-host baseline records
are not used to calculate these comparisons.

The initial 33-row matrix uses three timed repetitions per row and compares
NumPy/native or explicitly labelled alternatives. Shared-host variation produced
some transient twofold differences, including between auto and NumPy paths that
execute the same code. A targeted recheck used baseline/current/current/baseline
process order, one warm-up and five measurements per block (ten samples per
version/workload), under the same single-thread settings. The large slowdowns did
not reproduce. Absolute times and dispersion from that recheck follow.

| Public workload | Baseline median s | 1.1.2 median s | 1.1.2 min–max s | 1.1.2 std. dev. s | Current/baseline |
| --- | ---: | ---: | --- | ---: | ---: |
| 1000 dated bonds / NumPy | 0.060757 | 0.062591 | 0.057960–0.070164 | 0.003554 | 1.030 |
| 10000 bonds / NumPy | 0.113968 | 0.118733 | 0.112344–0.158867 | 0.013202 | 1.042 |
| 10000 bonds / native | 0.111590 | 0.112467 | 0.107865–0.122644 | 0.004502 | 1.008 |
| 2000 rate scenarios, chunk 16 / NumPy | 0.030021 | 0.031503 | 0.028812–0.032642 | 0.001330 | 1.049 |
| 2000 rate scenarios, chunk 16 / native | 0.021805 | 0.021237 | 0.020524–0.027862 | 0.002055 | 0.974 |
| 1000 × 20 ALM paths, chunk 256 / NumPy | 0.267860 | 0.293570 | 0.266208–0.329030 | 0.015656 | 1.096 |
| 1000 × 20 ALM paths, chunk 256 / native | 0.245973 | 0.233005 | 0.227681–0.281877 | 0.019517 | 0.947 |

The NumPy ALM-path median increased about 10% in this sample. Additional finite
output and dimensional preflights are retained for correctness; variable run
ranges overlap, so this does not isolate the cost of each check. Native ALM paths
were faster in this workload but do not have sufficient portable crossover
evidence to change their conservative automatic policy. No broad twofold
regression remained reproducible in the targeted recheck. Small percentage
differences are observations, not statistically established improvements.

## 1.1.2 NumPy/native comparisons

These are same-version comparisons, distinct from incremental release changes.
Each uses three timed repetitions. Numerical differences are in the result's
units (price/surplus/DV01 currency, yield rate, or life cashflow currency).

| Workload | NumPy s | Native s | Maximum absolute difference |
| --- | ---: | ---: | ---: |
| Yield solving: 1,000 bonds | 0.141319 | 0.029139 | 0 |
| Life projection: 10,000 policies | 9.217386 | 0.020258 | 1.86e-09 |
| ALM scenarios: 1,000 bonds x 1,000 scenarios x 9 nodes | 0.944476 | 0.754109 | 6.4e-10 |
| Risk aggregation: 10,000 weighted losses | 0.000951 | 0.000715 | 4.12e-17 |
| Key rate: 5,000 bonds x 33 nodes | 0.313704 | 0.195996 | 5.68e-14 |
| Key rate: 10,000 bonds x 33 nodes | 0.835912 | 0.390722 | 5.68e-14 |

The full large key-rate sweep preserves the native advantage at the conservative
8-million cashflow-visit threshold. Its 1000-bond/17-node recheck measured
0.03391 s NumPy versus 0.02463 s native; this smaller result is less portable and
does not justify lowering the threshold. The life comparison is the existing
Python policy/year loop versus the existing native loop, not a new 1.1.2 speedup.

The liability crossover was revalidated at 2048, 4096, 8192, 32768, 65536 and
262144 standalone cashflows. At 4096 the paired medians were 0.000517 s NumPy and
0.000534 s native. At 65536, seven further repetitions gave 0.005798 s versus
0.007319 s; at 262144, 0.031757 s versus 0.034497 s. This does not support the old
4096-flow native crossover, so **automatic standalone-liability valuation now
uses NumPy**. Explicit native execution remains available. This changes engine
selection and its report metadata, preserving financial semantics and tolerances.

## Static dispatch decisions

| Work | Auto choice with compatible native extension |
| --- | --- |
| Ordinary bond pricing and standalone ALM liabilities | NumPy; no stable broad native crossover |
| Yield inversion | Native for nonempty batches |
| Key-rate risk | Native at 8,000,000 cashflow/scenario visits; NumPy below |
| Rate/indexed scenarios | Native for nonempty compatible work |
| Life projection and life scenarios | Native for nonempty compatible work |
| Weighted risk | Native; intermediate near-ties are not portable gains |
| Multi-period ALM paths and standalone mortality | NumPy |

Policies live in `_dispatch.py`, with deterministic internal decision reasons.
There is no runtime calibration, threading framework or hardware-specific tuning.
All existing `engine` overrides remain. Native curves require linear-zero
interpolation and flat-zero extrapolation; unsupported explicit requests fail,
and auto falls back to the supported NumPy semantics.

## Peak resident memory

Seventeen fresh-process measurements cover bonds, rate scenarios, ALM paths,
life scenarios, streamed factor validation and MLAE. There is one warm-up and
three timed runs per worker; both chunks 16 and 256 are recorded where applicable.
The selected table shows chunk 256. Peak RSS includes interpreter, imported
packages, input arrays, warm-up and all timed calls; it is not pure kernel memory.
The last column subtracts the pre-call high-water mark and is only a diagnostic,
not a count of allocations or peak simultaneous live memory.

| Workload / engine | Baseline peak MiB | 1.1.2 peak MiB | Increase over pre-call high-water MiB |
| --- | ---: | ---: | ---: |
| bonds / numpy | 127.5 | 131.7 | 33.2 |
| bonds / native | 109.8 | 113.9 | 15.1 |
| rate_scenarios / numpy | 103.1 | 107.5 | 8.6 |
| rate_scenarios / native | 95.1 | 98.9 | 0.2 |
| alm_paths / numpy | 107.0 | 110.4 | 10.5 |
| alm_paths / native | 98.0 | 102.3 | 2.4 |
| life_scenarios / numpy | 95.9 | 100.1 | 0.1 |
| life_scenarios / native | 95.8 | 100.1 | 0.0 |
| factor_validation / numpy | 94.8 | 98.3 | 0.0 |
| mlae / numpy | 94.5 | 98.6 | 0.4 |

Most fresh-process peaks rose roughly 4 MiB alongside their import/pre-call floor;
there is no claimed reduction in total RSS. NumPy rate-scenario peak was 99.5 MiB
at chunk 16 and 107.5 MiB at chunk 256; ALM paths were 101.9 and 110.4 MiB. These
observations support bounded working blocks but not a hard process-memory cap.
Linux peak RSS was measured; the Windows/macOS reporting code is present but this
campaign does not claim measured RSS on those platforms. See [memory](memory.md)
for complexity, aggregate output sizes, 512 MiB guards and simulator limitations.

## Retained implementation alternatives and floating-point policy

The 1.1.2 par-yield comparison measured 0.00002196 s for one floating schedule
versus 0.00021286 s for two derived bonds; dated values were 0.00006972 s versus
0.00083999 s. Five repetitions gave zero result difference. These measure an
existing implementation choice, not an incremental 1.1.2 optimization. The MLAE
31-threshold comparison was 0.05317 s versus 0.52241 s for the dense search, with
5.66e-6 maximum amplitude difference. Its interval study is documented separately.
Prefix-sum cashflow reductions remain rejected because of cancellation; no
numerical tolerance was loosened to obtain a speedup.

The effective current native flags include `-O3 -DNDEBUG -std=c++20 -fPIC
-fvisibility=hidden -Wall -Wextra -Wpedantic -fno-fast-math -ffp-contract=off
-Werror -Wshadow -Wconversion -Wsign-conversion -flto=auto -fno-fat-lto-objects`.
CMake build type owns optimization; sanitizer builds append `-O1 -g` and disable
IPO. MSVC uses `/fp:strict`. Actual baseline and final compile commands are archived;
the baseline benchmark's missing inline flags are not silently filled in.

## Reproduction and raw records

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONHASHSEED=0
export QFIN_COMPILE_COMMANDS=build/verification/compile_commands.json
python examples/native_benchmark.py --repeats 3 --output native.md --json-output native.json
python examples/native_benchmark.py --full --section key-rate --repeats 3 --output key-rate.md --json-output key-rate.json
python tools/memory_benchmark.py --repeats 3 --output memory.json
python examples/hardening_benchmark.py --repeats 5 --output alternatives.md --json-output alternatives.json
```

Evidence: [before](history/hardening-1.1.2/baseline-native.json),
[after](history/hardening-1.1.2/current-native.json),
[alternating recheck](history/hardening-1.1.2/paired-benchmark.json),
[large liability crossover](history/hardening-1.1.2/large-liability-dispatch.json),
[key-rate sweep](history/hardening-1.1.2/current-key-rate.json),
[baseline RSS](history/hardening-1.1.2/baseline-memory.json),
[current RSS](history/hardening-1.1.2/current-memory.json),
[implementation alternatives](history/hardening-1.1.2/implementation-alternatives.json),
and [effective compiler commands](history/hardening-1.1.2/compile-commands.json).
