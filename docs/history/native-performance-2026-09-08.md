# QFin native performance

These are measured end-to-end public-API timings; no result is fabricated. Object-to-buffer conversion and Python/C++ boundary costs are included.

## Environment

- Measurement date (UTC): 2026-09-08
- OS: Linux-6.18.35-x86_64-with-glibc2.39
- CPU: AMD EPYC 9V74 80-Core Processor
- Architecture: x86_64
- Python: 3.12.13
- QFin: 1.1.1
- NumPy: 2.5.2
- SciPy: 1.18.1
- Thread environment: OMP=1; OpenBLAS=1; MKL=1
- PennyLane: 0.45.1
- PennyLane-Lightning: 0.45.0
- QFin native: qfin-native (C++20)
- C++ compiler: GNU 13.3.0
- Requested repetitions: median of 3
- Large Python/NumPy references: one timed run
- Quantum device rows: median of at least 5 runs
- Native threading: deterministic single-threaded execution (no OpenMP)

## Results

| Workload | Problem | Reference | Accelerated | Reference (s) | Accelerated (s) | Speedup | Max difference |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: |
| Fixed-income public API | 1 bond | NumPy | QFin C++ | 0.000097 | 0.000028 | 3.48x | 0.000e+00 |
| Fixed-income pure Python | 1 bond | Python | QFin C++ | 0.000083 | 0.000028 | 2.98x | 0.000e+00 |
| Fixed-income public API | 100 bonds | NumPy | QFin C++ | 0.001173 | 0.001404 | 0.84x | 2.842e-14 |
| Fixed-income pure Python | 100 bonds | Python | QFin C++ | 0.075533 | 0.001404 | 53.82x | 5.684e-14 |
| Fixed-income public API | 1,000 bonds | NumPy | QFin C++ | 0.011052 | 0.011085 | 1.00x | 5.684e-14 |
| Fixed-income pure Python | 1,000 bonds | Python | QFin C++ | 0.828300 | 0.011085 | 74.72x | 5.684e-14 |
| Fixed-income public API | 10,000 bonds | NumPy | QFin C++ | 0.129467 | 0.119884 | 1.08x | 5.684e-14 |
| Fixed-income public API | 100,000 bonds | NumPy | QFin C++ | 1.388217 | 1.247230 | 1.11x | 5.684e-14 |
| Yield solving | 100 bonds | Python/NumPy | QFin C++ | 0.013263 | 0.002799 | 4.74x | 0.000e+00 |
| Yield solving | 1,000 bonds | Python/NumPy | QFin C++ | 0.132039 | 0.031080 | 4.25x | 0.000e+00 |
| Yield solving | 10,000 bonds | Python/NumPy | QFin C++ | 1.346354 | 0.297733 | 4.52x | 6.821e-13 |
| Yield solving | 100,000 bonds | Python/NumPy | QFin C++ | 13.244258 | 3.378468 | 3.92x | 9.095e-13 |
| ALM base valuation | 100 assets | NumPy | QFin C++ | 0.001321 | 0.001072 | 1.23x | 1.819e-12 |
| ALM base valuation | 1,000 assets | NumPy | QFin C++ | 0.011544 | 0.011742 | 0.98x | 1.819e-12 |
| ALM base valuation | 10,000 assets | NumPy | QFin C++ | 0.112560 | 0.111015 | 1.01x | 1.819e-12 |
| Life projection | 1,000 policies | Python/NumPy | QFin C++ | 0.932518 | 0.003879 | 240.39x | 1.164e-10 |
| Life projection | 10,000 policies | Python/NumPy | QFin C++ | 8.883163 | 0.016365 | 542.81x | 1.863e-09 |
| Life projection | 100,000 policies | Python/NumPy | QFin C++ | 88.513803 | 0.182192 | 485.83x | 0.000e+00 |
| ALM scenarios | 100 bonds x 1,000 scenarios x 5 nodes | NumPy | QFin C++ | 0.081984 | 0.057398 | 1.43x | 1.273e-11 |
| ALM scenarios | 1,000 bonds x 1,000 scenarios x 9 nodes | NumPy | QFin C++ | 0.880815 | 0.623699 | 1.41x | 6.403e-10 |
| ALM scenarios | 1,000 bonds x 10,000 scenarios x 17 nodes | NumPy | QFin C++ | 8.005449 | 6.700020 | 1.19x | 7.276e-10 |
| ALM scenarios | 10,000 bonds x 1,000 scenarios x 17 nodes | NumPy | QFin C++ | 10.262940 | 6.605984 | 1.55x | 4.587e-08 |
| Risk aggregation | 1,000 weighted losses | NumPy | QFin C++ | 0.000078 | 0.000027 | 2.87x | 5.063e-14 |
| Risk aggregation | 10,000 weighted losses | NumPy | QFin C++ | 0.000918 | 0.000715 | 1.28x | 2.220e-16 |
| Risk aggregation | 100,000 weighted losses | NumPy | QFin C++ | 0.011105 | 0.009193 | 1.21x | 4.441e-16 |
| Risk aggregation | 1,000,000 weighted losses | NumPy | QFin C++ | 0.147119 | 0.107658 | 1.37x | 4.441e-16 |
| Key-rate risk | 1 bonds x 5 nodes | Brute-force public repricing | Prepared QFin C++ | 0.001703 | 0.000075 | 22.60x | 1.421e-14 |
| Key-rate risk | 100 bonds x 5 nodes | Brute-force public repricing | Prepared QFin C++ | 0.015498 | 0.001573 | 9.85x | 5.684e-14 |
| Key-rate risk | 100 bonds x 17 nodes | Brute-force public repricing | Prepared QFin C++ | 0.044061 | 0.003231 | 13.64x | 5.684e-14 |
| Key-rate risk | 1,000 bonds x 17 nodes | Brute-force public repricing | Prepared QFin C++ | 0.493380 | 0.034485 | 14.31x | 5.684e-14 |
| Key-rate risk | 10,000 bonds x 33 nodes | Brute-force public repricing | Prepared QFin C++ | 7.974438 | 0.602876 | 13.23x | 5.684e-14 |
| Curve semantics | 1,000 floating bonds; linear_zero | NumPy | auto | 0.015202 | 0.013263 | 1.15x | 0.000e+00 |
| Curve semantics | 1,000 dated bonds; linear_zero | NumPy | auto | 0.087947 | 0.086422 | 1.02x | 0.000e+00 |
| Curve semantics | 1,000 floating bonds; linear_discount | NumPy | auto | 0.013372 | 0.014012 | 0.95x | 0.000e+00 |
| Curve semantics | 1,000 dated bonds; linear_discount | NumPy | auto | 0.085615 | 0.104841 | 0.82x | 0.000e+00 |
| Curve semantics | 1,000 floating bonds; log_linear_discount | NumPy | auto | 0.012738 | 0.011977 | 1.06x | 0.000e+00 |
| Curve semantics | 1,000 dated bonds; log_linear_discount | NumPy | auto | 0.076572 | 0.079620 | 0.96x | 0.000e+00 |
| Curve semantics | 1,000 floating bonds; monotone_zero | NumPy | auto | 0.013683 | 0.015371 | 0.89x | 0.000e+00 |
| Curve semantics | 1,000 dated bonds; monotone_zero | NumPy | auto | 0.079419 | 0.081108 | 0.98x | 0.000e+00 |
| Quantum simulation | 5 data qubits, power 0 | default.qubit | PennyLane-Lightning C++ | 0.004298 | 0.002352 | 1.83x | 1.110e-16 |

## Numerical acceptance

`Max difference` is the largest absolute difference over the compared outputs. Parity tests use relative tolerance `1e-13` for fixed-income, ALM, scenario, and life outputs (`1e-11` for finite-difference DV01). Weighted expected shortfall uses an absolute `1e-10` large-batch bound; analytical cases and all other risk statistics use tighter tolerances.

## Interpretation

Native dispatch is useful only when the eliminated inner loop exceeds buffer-conversion cost. The benchmark deliberately exposes crossover cases; a speedup below 1.0x means the reference was faster for that measured size.
The measured NumPy/native fixed-income crossover is between 100 and 10,000 bonds for this mixed-maturity workload.
Life projection removes the policy-by-year Python loop; observed speedups range from 240.39x to 542.81x.
Chunked ALM scenario valuation observed speedups from 1.19x to 1.55x.
Risk auto-dispatch uses native when available. The displayed comparisons show whether that policy fits this runner; near-equal timings are not a portable speedup claim.
The quantum row compares PennyLane devices only; Lightning C++ remains the quantum simulator and is independent of QFin's finance C++ extension.
SciPy has no separate row because these reference cases use vectorized NumPy or an explicit batch bisection; no SciPy primitive is used.
Timings are environment-specific measurements, not performance guarantees.
