# QFin 0.6 ALM and life performance

These are measured end-to-end public-API timings. Validation, conversion, chunk dispatch, and result construction are included; no number is fabricated.

## Environment

- Measurement date (UTC): 2026-08-29
- OS: Linux-6.18.35-x86_64-with-glibc2.39
- CPU: INTEL(R) XEON(R) PLATINUM 8573C
- Architecture: x86_64
- Python: 3.12.13
- QFin: 0.6.0
- NumPy: 2.5.2
- QFin native: qfin-native (C++20)
- C++ compiler: GNU 13.3.0
- Native timings: median of 3 runs after one warm-up
- Large NumPy references: one timed run
- Native threading: deterministic single-threaded execution (no OpenMP)

## Results

| Workload | NumPy (s) | QFin C++ (s) | Speedup | Max difference | Returned arrays | Peak chunk estimate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 100 bonds x 100 scenarios x 10 years | 0.070643 | 0.064708 | 1.09x | 2.037e-10 | 0.074 MiB | n/a |
| 100 bonds x 1,000 scenarios x 10 years | 0.566937 | 0.645784 | 0.88x | 2.328e-10 | 0.740 MiB | n/a |
| 100 bonds x 10,000 scenarios x 10 years | 6.513893 | 7.054609 | 0.92x | 2.910e-10 | 7.401 MiB | n/a |
| 1 model point / 100 policies x 1 scenario x 1 year | 0.000204 | 0.000083 | 2.46x | 0.000e+00 | 0.000 MiB | 0.000 MiB |
| 1 model point / 100 policies x 1 scenario x 20 years | 0.001275 | 0.000092 | 13.84x | 0.000e+00 | 0.000 MiB | 0.002 MiB |
| 10 model points / 1,000 policies x 10 scenarios x 20 years | 0.067333 | 0.000244 | 275.42x | 7.451e-09 | 0.000 MiB | 0.015 MiB |
| 25 model points / 2,500 policies x 20 scenarios x 20 years | 0.352339 | 0.000733 | 480.85x | 4.470e-08 | 0.001 MiB | 0.031 MiB |
| 100 model points / 10,000 policies x 250 scenarios x 20 years | 15.951797 | 0.033957 | 469.76x | 2.980e-07 | 0.010 MiB | 0.103 MiB |
| 1,000 model points / 100,000 policies x 100 scenarios x 20 years | 64.028196 | 0.139467 | 459.09x | 2.384e-06 | 0.004 MiB | 0.122 MiB |

## Interpretation

The ALM kernel returns scenario-by-period portfolio aggregates; it never returns a scenario-by-instrument cube. The life kernel returns five scenario-level aggregates and chunks both scenarios and model points, so represented policy counts do not multiply the result shape after grouping.

`Max difference` is the largest absolute difference across the compared aggregate outputs. Tests enforce relative parity tolerances, so absolute differences must be interpreted against portfolio-scale monetary values.

Multi-period ALM speedups range from 0.88x to 1.09x in this run. Because some rows remain close to 1.0x, the gain is not stable enough for automatic dispatch: `engine="auto"` conservatively stays on NumPy and native remains an explicit profiling override.

The native life kernel is faster at the smallest measured non-empty workload (2.46x) and the benefit grows for larger grouped books. Non-empty life and life-scenario workloads therefore select native execution automatically when the extension is available.

The generated factors are synthetic independent benchmark draws, not a calibrated economic-scenario model. Timings are environment-specific and are not performance guarantees.

Reproduce this report with:

```bash
python examples/alm_life_benchmark.py --full --repeats 3 \
  --output docs/alm-life-performance.md
```
