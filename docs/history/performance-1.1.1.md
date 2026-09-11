# Performance and dispatch

Measurements are end-to-end public API calls unless explicitly labelled a private
kernel investigation. They include schedule generation, buffer conversion,
allocation and the Python/C++ boundary. Model inputs are constructed before timing.
Each JSON row records the workload, references, absolute times, numerical
difference, repetition count and environment. A native comparison is not
necessarily a before/after release comparison.

The main runner was Linux x86_64 on AMD EPYC 9V74, Python 3.12.14, QFin 1.1.1,
NumPy 2.5.2, SciPy 1.18.1, GCC 13.3.0, PennyLane 0.45.1 and Lightning 0.45.0.
OMP/OpenBLAS/MKL thread counts were set to one; QFin native kernels are single
threaded. Compiler settings are recorded in the JSON reports. Earlier records
retain their own Python patch version and environment; they are not relabelled.

## Before/after execution changes

These compare the previous execution architecture against the same financial
calculation using the hardened implementation. They do not compare different
products or omit object conversion costs.

| Calculation | Before (s) | After (s) | Repetitions | Maximum difference |
| --- | ---: | ---: | ---: | ---: |
| Floating par yield; two derived bonds → one schedule | 0.000190362 | 0.000025087 | 5 | 0 |
| Dated par yield; two derived bonds → one schedule | 0.000616231 | 0.000073578 | 5 | 0 |
| Key rate; 1 bond, 5 nodes; repeated repricing → prepared native | 0.001510993 | 0.000062082 | 3 | 6.70e-15 |
| Key rate; 100 bonds, 5 nodes | 0.018422321 | 0.002370113 | 3 | 4.26e-14 |
| Key rate; 1,000 bonds, 17 nodes | 0.532763344 | 0.031016821 | 3 | 5.68e-14 |
| Key rate; 10,000 bonds, 33 nodes | 8.273657942 | 0.436920111 | 3 | 5.68e-14 |
| 10,000-bond public pricing; repeat/bincount → current reduction | 0.137317111 | 0.119005335 | 5 | 1.42e-14 |
| 31 MLAE fits; dense search → all-mode refinement plus regions | 0.5170875 | 0.052362911 | 5 | 5.66e-6 amplitude |

The key-rate gains largely come from eliminating repeated schedule/curve/buffer
work. Comparing the already-prepared engines gives a much smaller difference:
5,000 bonds/17 nodes measured **0.251386 s NumPy versus 0.126740 s native**;
10,000 bonds/33 nodes measured **0.936767 s versus 0.436920 s**. Small and medium
cases were less stable across repeated runs.

See [final scenario/key-rate rows](../scenario-final-performance.md),
[par-yield and reduction alternatives](../hardening-performance.md), and
[MLAE timing/coverage](../mlae-validation.md), each with accompanying JSON.

## Current public API matrix

The [complete current matrix](../native-performance-current.md) includes:

- Pricing at 1, 100, 1,000, 10,000 and 100,000 bonds; annual/semiannual/quarterly
  coupons and mixed maturities. Ordinary native pricing has no stable broad win.
- Yield inversion at 100 through 100,000 bonds: the largest case measured
  **12.506504 s NumPy/Python versus 2.943251 s native**.
- ALM base valuation at 100 through 10,000 assets. At 10,000, NumPy measured
  **0.098413 s versus 0.103154 s native**; NumPy won this run.
- Scenarios at 100×1,000, 1,000×1,000, 1,000×10,000 and 10,000×1,000 assets/scenarios.
  The 1,000×10,000 case measured **11.435073 s versus 7.792020 s**, with maximum
  surplus difference `7.28e-10`. Largest scenario rows use one timed run per engine;
  their speedups have less replication evidence than the smaller median rows.
- Life at 1,000, 10,000 and 100,000 model points. The largest Python/NumPy reference
  was one timed run: **87.413358 s versus 0.178059 s native**. This compares an
  existing Python policy/year loop to its existing native counterpart, not the
  incremental hardening speedup, and does not establish production actuarial equivalence.
- Weighted risk from 1,000 to 1,000,000 losses. The million-loss case measured
  **0.149734 s versus 0.109280 s**; near-equal middle cases varied between runs.
- Eight dated/floating curve configurations spanning all four interpolation modes,
  positive and negative rates, and flat-forward extrapolation. Auto and NumPy
  follow the same methodology; timing fluctuations there are not different kernels.
- A representative existing five-data-qubit circuit: **0.003596 s default.qubit
  versus 0.002110 s lightning.qubit**, with probability difference `1.11e-16`.

## Automatic engine policy

| Work | Current automatic choice when the extension is available |
| --- | --- |
| Ordinary curve/YTM bond pricing | NumPy; inconsistent broad native crossover |
| Batch yield inversion | Native for nonempty work |
| Key-rate risk | Native from 8,000,000 cash-flow/scenario visits; otherwise NumPy |
| ALM base | Pricing policy above; standalone liability batch native from 4,096 cash flows; combined engine reported |
| Rate and indexed scenarios | Native for nonempty compatible work |
| Life projection/scenarios | Native for nonempty compatible work; mixed analytics labelled |
| Weighted risk | Native; middle-size near-ties are not portable gains |
| Multi-period ALM paths | NumPy by default; explicit native remains supported |
| Standalone mortality interpolation | NumPy |

All native curve choices require linear-zero interpolation with flat-zero
extrapolation. Other supported curves use NumPy; an explicit unsupported native
request fails. The key-rate threshold is deliberately conservative: repeat runs
at roughly 1.8 million visits ranged from near parity to a modest win, while
roughly 8.8 million visits supported the larger crossover. Constants are policies
for the measured workload, not promises on every CPU or cash-flow shape.

## Numerical and memory tradeoffs

Prefix-sum differencing was rejected: it loses a one-unit segment after a
trillion-scale prefix and accumulates larger errors in public pricing. `reduceat`
keeps empty-segment handling and avoids a repeated index array for large buffers.
At 10,000 bonds, overall traced peak allocation remained about **27.54 MB** in both
implementations because other valuation arrays dominate. There is no claimed
large end-to-end peak-memory reduction from this change alone.

For 30,000 weighted mixed-sign terms near 1e12 with one-unit adjustments, naive
and Kahan reductions missed about **0.203451** currency units and NumPy pairwise
missed **0.193929** against `math.fsum`. Neumaier and QFin's selective fsum path
matched that reference. The Python selective path cost **0.001641 s** versus
**0.000007 s** for pairwise reduction in this stress case; it is deliberately
reserved for severe cancellation. These are reduction microbenchmarks, not
public-API speedup claims. The native compensated implementation remains separately
covered by parity tests.

The [native output investigation](../native-output-performance.md) compared an
installed audited-main extension with current code on one million tiny scenarios:
**0.011196 s before versus 0.012009 s after**. Current code also performs stronger
validation, so this is not evidence that removing a copy alone made compute slower
or faster. NumPy-owned output removes one scenario-length vector and transfer
(8 MB for one million doubles), with straightforward lifetime and exception safety.
Small outputs retain simple copies; capsule ownership transfer was not introduced.

NumPy scenario intermediates are bounded to 2,097,152 elements per matrix (16 MiB),
with several matrices potentially live simultaneously. A public chunk size is an
upper bound, not a reservation. Key-rate matrices have a separate 8-million-element
chunk target. Life and multi-period scenarios retain aggregate outputs and chunked
execution; no new scenario×instrument×period or scenario×policy×year cube is used.

## Reproducing and comparing

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python examples/native_benchmark.py --full --repeats 3 \
  --output native-performance.md --json-output native-performance.json
python examples/hardening_benchmark.py --repeats 5 \
  --output hardening-performance.md --json-output hardening-performance.json
python examples/mlae_benchmark.py --repeats 5 --coverage-repetitions 200 \
  --output mlae-validation.md --json-output mlae-validation.json
```

Use identical environment/thread settings when comparing releases. Compare rows
by benchmark, problem size and engine, and retain absolute times and numerical
differences. Scheduled/manual CI uploads the machine-readable evidence; ordinary
CI gates correctness and coverage, not minor noisy timing movement. Investigate
large changes before adjusting dispatch. No OpenMP, TBB or new threading framework
was added.
