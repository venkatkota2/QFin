# QFin performance evidence

Measured end-to-end public-API timings, including object-to-buffer conversion and Python/C++ boundary costs.

## Environment

- Measurement date (UTC): 2026-09-09T05:31:21.710533+00:00
- OS: Linux-6.18.35-x86_64-with-glibc2.39
- CPU: AMD EPYC 9V74 80-Core Processor
- Architecture: x86_64
- Python: 3.12.14
- QFin: 1.1.1
- NumPy: 2.5.2
- SciPy: 1.18.1
- Thread environment: OMP=1; OpenBLAS=1; MKL=1
- PennyLane: 0.45.1
- PennyLane-Lightning: 0.45.0
- C++ compiler: GNU 13.3.0
- Compiler flags: -O3 -DNDEBUG -std=c++20 -fPIC -fvisibility=hidden -O3 -Wall -Wextra -Wpedantic -Werror -Wshadow -Wconversion -Wsign-conversion -flto=auto -fno-fat-lto-objects
- Requested repetitions: median of 3
- Per-row reference and QFin repetition counts are recorded in JSON; some large reference cases use one timed run.
- Native threading: deterministic single-threaded execution (no OpenMP)

This matrix combines the recorded runs from the same environment. Individual rows retain their original measurement timestamps in JSON; scenario/key-rate rows use the final validation build.

## Results

| Workload | Problem | Reference | Accelerated | Reference (s) | Accelerated (s) | Speedup | Max difference |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: |
| Fixed-income public API | 1 bond | NumPy | QFin C++ | 0.000112 | 0.000035 | 3.20x | 0.000e+00 |
| Fixed-income pure Python | 1 bond | Python | QFin C++ | 0.000073 | 0.000035 | 2.09x | 0.000e+00 |
| Fixed-income public API | 100 bonds | NumPy | QFin C++ | 0.001381 | 0.001072 | 1.29x | 2.842e-14 |
| Fixed-income pure Python | 100 bonds | Python | QFin C++ | 0.075686 | 0.001072 | 70.59x | 5.684e-14 |
| Fixed-income public API | 1,000 bonds | NumPy | QFin C++ | 0.010932 | 0.010782 | 1.01x | 5.684e-14 |
| Fixed-income pure Python | 1,000 bonds | Python | QFin C++ | 0.777370 | 0.010782 | 72.10x | 5.684e-14 |
| Fixed-income public API | 10,000 bonds | NumPy | QFin C++ | 0.114580 | 0.116571 | 0.98x | 5.684e-14 |
| Fixed-income public API | 100,000 bonds | NumPy | QFin C++ | 1.234544 | 1.142911 | 1.08x | 5.684e-14 |
| Yield solving | 100 bonds | Python/NumPy | QFin C++ | 0.011978 | 0.002569 | 4.66x | 0.000e+00 |
| Yield solving | 1,000 bonds | Python/NumPy | QFin C++ | 0.126379 | 0.029943 | 4.22x | 0.000e+00 |
| Yield solving | 10,000 bonds | Python/NumPy | QFin C++ | 1.310756 | 0.282962 | 4.63x | 6.821e-13 |
| Yield solving | 100,000 bonds | Python/NumPy | QFin C++ | 12.506504 | 2.943251 | 4.25x | 9.095e-13 |
| ALM base valuation | 100 assets | NumPy | QFin C++ | 0.001176 | 0.001070 | 1.10x | 1.819e-12 |
| ALM base valuation | 1,000 assets | NumPy | QFin C++ | 0.009778 | 0.010304 | 0.95x | 1.819e-12 |
| ALM base valuation | 10,000 assets | NumPy | QFin C++ | 0.098413 | 0.103154 | 0.95x | 1.819e-12 |
| Life projection | 1,000 policies | Python/NumPy | QFin C++ | 0.844689 | 0.001687 | 500.63x | 1.164e-10 |
| Life projection | 10,000 policies | Python/NumPy | QFin C++ | 8.916255 | 0.016634 | 536.02x | 1.863e-09 |
| Life projection | 100,000 policies | Python/NumPy | QFin C++ | 87.413358 | 0.178059 | 490.92x | 0.000e+00 |
| Key-rate risk | 1 bonds x 5 nodes | Brute-force public repricing | Prepared QFin C++ | 0.001511 | 0.000062 | 24.34x | 6.702e-15 |
| Key-rate engine comparison | 1 bonds x 5 nodes | Prepared NumPy | Prepared QFin C++ | 0.000125 | 0.000062 | 2.01x | 0.000e+00 |
| Key-rate risk | 100 bonds x 5 nodes | Brute-force public repricing | Prepared QFin C++ | 0.018422 | 0.002370 | 7.77x | 4.263e-14 |
| Key-rate engine comparison | 100 bonds x 5 nodes | Prepared NumPy | Prepared QFin C++ | 0.002929 | 0.002370 | 1.24x | 4.263e-14 |
| Key-rate risk | 100 bonds x 17 nodes | Brute-force public repricing | Prepared QFin C++ | 0.046913 | 0.003997 | 11.74x | 5.684e-14 |
| Key-rate engine comparison | 100 bonds x 17 nodes | Prepared NumPy | Prepared QFin C++ | 0.004247 | 0.003997 | 1.06x | 5.684e-14 |
| Key-rate risk | 1,000 bonds x 17 nodes | Brute-force public repricing | Prepared QFin C++ | 0.532763 | 0.031017 | 17.18x | 5.684e-14 |
| Key-rate engine comparison | 1,000 bonds x 17 nodes | Prepared NumPy | Prepared QFin C++ | 0.035244 | 0.031017 | 1.14x | 5.684e-14 |
| Key-rate risk | 1,000 bonds x 33 nodes | Brute-force public repricing | Prepared QFin C++ | 0.785042 | 0.050741 | 15.47x | 5.684e-14 |
| Key-rate engine comparison | 1,000 bonds x 33 nodes | Prepared NumPy | Prepared QFin C++ | 0.057893 | 0.050741 | 1.14x | 5.684e-14 |
| Key-rate risk | 5,000 bonds x 17 nodes | Brute-force public repricing | Prepared QFin C++ | 2.186350 | 0.126740 | 17.25x | 5.684e-14 |
| Key-rate engine comparison | 5,000 bonds x 17 nodes | Prepared NumPy | Prepared QFin C++ | 0.251386 | 0.126740 | 1.98x | 5.684e-14 |
| Key-rate risk | 5,000 bonds x 33 nodes | Brute-force public repricing | Prepared QFin C++ | 4.091591 | 0.204550 | 20.00x | 5.684e-14 |
| Key-rate engine comparison | 5,000 bonds x 33 nodes | Prepared NumPy | Prepared QFin C++ | 0.346407 | 0.204550 | 1.69x | 5.684e-14 |
| Key-rate risk | 10,000 bonds x 33 nodes | Brute-force public repricing | Prepared QFin C++ | 8.273658 | 0.436920 | 18.94x | 5.684e-14 |
| Key-rate engine comparison | 10,000 bonds x 33 nodes | Prepared NumPy | Prepared QFin C++ | 0.936767 | 0.436920 | 2.14x | 5.684e-14 |
| ALM scenarios | 100 bonds x 1,000 scenarios x 5 nodes | NumPy | QFin C++ | 0.104445 | 0.063893 | 1.63x | 1.273e-11 |
| ALM scenarios | 1,000 bonds x 1,000 scenarios x 9 nodes | NumPy | QFin C++ | 1.130055 | 0.710407 | 1.59x | 6.403e-10 |
| ALM scenarios | 1,000 bonds x 10,000 scenarios x 17 nodes | NumPy | QFin C++ | 11.435073 | 7.792020 | 1.47x | 7.276e-10 |
| ALM scenarios | 10,000 bonds x 1,000 scenarios x 17 nodes | NumPy | QFin C++ | 13.221703 | 7.909861 | 1.67x | 4.587e-08 |
| Risk aggregation | 1,000 weighted losses | NumPy | QFin C++ | 0.000072 | 0.000029 | 2.51x | 5.063e-14 |
| Risk aggregation | 10,000 weighted losses | NumPy | QFin C++ | 0.000834 | 0.000767 | 1.09x | 2.220e-16 |
| Risk aggregation | 100,000 weighted losses | NumPy | QFin C++ | 0.011494 | 0.012049 | 0.95x | 4.441e-16 |
| Risk aggregation | 1,000,000 weighted losses | NumPy | QFin C++ | 0.149734 | 0.109280 | 1.37x | 4.441e-16 |
| Curve semantics | 1,000 floating bonds; linear_zero | NumPy | auto | 0.019490 | 0.013728 | 1.42x | 0.000e+00 |
| Curve semantics | 1,000 dated bonds; linear_zero | NumPy | auto | 0.079079 | 0.087609 | 0.90x | 0.000e+00 |
| Curve semantics | 1,000 floating bonds; linear_discount | NumPy | auto | 0.015030 | 0.014008 | 1.07x | 0.000e+00 |
| Curve semantics | 1,000 dated bonds; linear_discount | NumPy | auto | 0.076777 | 0.098999 | 0.78x | 0.000e+00 |
| Curve semantics | 1,000 floating bonds; log_linear_discount | NumPy | auto | 0.011987 | 0.012362 | 0.97x | 0.000e+00 |
| Curve semantics | 1,000 dated bonds; log_linear_discount | NumPy | auto | 0.075680 | 0.091358 | 0.83x | 0.000e+00 |
| Curve semantics | 1,000 floating bonds; monotone_zero | NumPy | auto | 0.012281 | 0.013347 | 0.92x | 0.000e+00 |
| Curve semantics | 1,000 dated bonds; monotone_zero | NumPy | auto | 0.078428 | 0.078483 | 1.00x | 0.000e+00 |
| Quantum simulation | 5 data qubits, power 0 | default.qubit | PennyLane-Lightning C++ | 0.003596 | 0.002110 | 1.70x | 1.110e-16 |

## Numerical acceptance

`Max difference` is the largest absolute difference over the compared outputs. Numerical and financial-unit tolerances vary by quantity and workload; see tests/native, tests/finance and docs/validation.md. A small absolute difference alone is not an acceptance criterion. Correctness tests are run before accepting a measured implementation change.

## Interpretation

Native dispatch is useful only when the eliminated inner loop exceeds buffer-conversion cost. The benchmark deliberately exposes crossover cases; a speedup below 1.0x means the reference was faster for that measured size.
The 100- and 10,000-bond rows do not bracket a stable NumPy/native crossover in this run; use the displayed measurements directly.
Life projection removes the policy-by-year Python loop; observed speedups range from 490.92x to 536.02x.
Chunked ALM scenario valuation observed speedups from 1.47x to 1.67x.
Risk auto-dispatch uses native when available. The displayed comparisons show whether that policy fits this runner; near-equal timings are not a portable speedup claim.
The quantum row compares PennyLane devices only; Lightning C++ remains the quantum simulator and is independent of QFin's finance C++ extension.
SciPy has no separate row because these reference cases use vectorized NumPy or an explicit batch bisection; no SciPy primitive is used.
Timings are environment-specific measurements, not performance guarantees.
