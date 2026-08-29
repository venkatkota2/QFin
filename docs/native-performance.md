# QFin native performance

These are measured end-to-end public-API timings; no result is fabricated. Object-to-buffer conversion and Python/C++ boundary costs are included.

## Environment

- Measurement date (UTC): 2026-08-29
- OS: Linux-6.18.35-x86_64-with-glibc2.39
- CPU: AMD EPYC 9V74 80-Core Processor
- Architecture: x86_64
- Python: 3.12.13
- QFin: 0.4.0
- NumPy: 2.3.5
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
| Fixed-income public API | 1 bond | NumPy | QFin C++ | 0.000060 | 0.000016 | 3.69x | 0.000e+00 |
| Fixed-income pure Python | 1 bond | Python | QFin C++ | 0.000051 | 0.000016 | 3.12x | 0.000e+00 |
| Fixed-income public API | 100 bonds | NumPy | QFin C++ | 0.000859 | 0.000793 | 1.08x | 2.842e-14 |
| Fixed-income pure Python | 100 bonds | Python | QFin C++ | 0.026979 | 0.000793 | 34.01x | 5.684e-14 |
| Fixed-income public API | 10,000 bonds | NumPy | QFin C++ | 0.090563 | 0.084965 | 1.07x | 2.842e-14 |
| Fixed-income pure Python | 10,000 bonds | Python | QFin C++ | 2.871603 | 0.084965 | 33.80x | 5.684e-14 |
| Fixed-income public API | 100,000 bonds | NumPy | QFin C++ | 0.943547 | 0.898020 | 1.05x | 2.842e-14 |
| Fixed-income pure Python | 100,000 bonds | Python | QFin C++ | 28.310689 | 0.898020 | 31.53x | 5.684e-14 |
| Yield solving | 100 bonds | Python/NumPy | QFin C++ | 0.012038 | 0.002578 | 4.67x | 0.000e+00 |
| Yield solving | 10,000 bonds | Python/NumPy | QFin C++ | 1.205628 | 0.287429 | 4.19x | 0.000e+00 |
| ALM base valuation | 100 assets | NumPy | QFin C++ | 0.000977 | 0.000848 | 1.15x | 1.819e-12 |
| ALM base valuation | 1,000 assets | NumPy | QFin C++ | 0.008110 | 0.008039 | 1.01x | 1.819e-12 |
| ALM base valuation | 10,000 assets | NumPy | QFin C++ | 0.085653 | 0.085569 | 1.00x | 1.819e-12 |
| Life projection | 1,000 policies | Python/NumPy | QFin C++ | 0.533946 | 0.000923 | 578.37x | 1.164e-10 |
| Life projection | 10,000 policies | Python/NumPy | QFin C++ | 5.186785 | 0.008787 | 590.31x | 1.863e-09 |
| Life projection | 100,000 policies | Python/NumPy | QFin C++ | 51.775355 | 0.087451 | 592.05x | 1.490e-08 |
| ALM scenarios | 1,000 bonds x 1,000 scenarios | NumPy | QFin C++ | 0.333906 | 0.259681 | 1.29x | 2.063e-08 |
| ALM scenarios | 1,000 bonds x 10,000 scenarios | NumPy | QFin C++ | 4.070378 | 3.099346 | 1.31x | 2.758e-08 |
| Risk aggregation | 1,000 weighted losses | NumPy | QFin C++ | 0.000049 | 0.000023 | 2.16x | 7.088e-13 |
| Risk aggregation | 10,000 weighted losses | NumPy | QFin C++ | 0.000781 | 0.000734 | 1.06x | 8.082e-14 |
| Risk aggregation | 100,000 weighted losses | NumPy | QFin C++ | 0.011791 | 0.043572 | 0.27x | 9.661e-12 |
| Quantum simulation | 5 data qubits, power 0 | default.qubit | PennyLane-Lightning C++ | 0.012940 | 0.002272 | 5.70x | 1.110e-16 |

## Numerical acceptance

`Max difference` is the largest absolute difference over the compared outputs. Parity tests use relative tolerance `1e-13` for fixed-income, ALM, scenario, and life outputs (`1e-11` for finite-difference DV01). Weighted expected shortfall uses an absolute `1e-10` large-batch bound; analytical cases and all other risk statistics use tighter tolerances.

## Interpretation

Native dispatch is useful only when the eliminated inner loop exceeds buffer-conversion cost. The benchmark deliberately exposes crossover cases; a speedup below 1.0x means the reference was faster for that measured size.
The 100- and 10,000-bond rows do not bracket a stable NumPy/native crossover in this run; use the displayed measurements directly.
Life projection removes the policy-by-year Python loop; observed speedups range from 578.37x to 592.05x.
Chunked ALM scenario valuation observed speedups from 1.29x to 1.31x.
Native tail-risk aggregation was slower in the largest measured case;
automatic risk dispatch therefore remains on NumPy.
The quantum row compares PennyLane devices only; Lightning C++ remains the quantum simulator and is independent of QFin's finance C++ extension.
SciPy has no separate row because these reference cases use vectorized NumPy or an explicit batch bisection; no SciPy primitive is used.
Timings are environment-specific measurements, not performance guarantees.
