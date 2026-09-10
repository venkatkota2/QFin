# QFin performance evidence

Measured end-to-end public-API timings, including object-to-buffer conversion and Python/C++ boundary costs.

## Environment

- Measurement date (UTC): 2026-09-09T17:21:53.256653+00:00
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
- Requested repetitions: median of 5
- Per-row reference and QFin repetition counts are recorded in JSON; some large reference cases use one timed run.
- Native threading: deterministic single-threaded execution (no OpenMP)

## Results

| Workload | Problem | Reference | Accelerated | Reference (s) | Accelerated (s) | Speedup | Max difference |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: |
| Par yield | floating | Two derived bonds and full repricing | One schedule | 0.000190 | 0.000025 | 7.59x | 0.000e+00 |
| Par yield | dated | Two derived bonds and full repricing | One schedule | 0.000616 | 0.000074 | 8.38x | 0.000e+00 |
| Public bond reduction alternatives | 1 bonds | repeat + bincount | reduceat | 0.000085 | 0.000092 | 0.92x | 0.000e+00 |
| Public bond reduction alternatives | 1 bonds | repeat + bincount | current dispatch | 0.000085 | 0.000083 | 1.03x | 0.000e+00 |
| Public bond reduction alternatives | 1 bonds | repeat + bincount | prefix differences | 0.000085 | 0.000085 | 0.99x | 0.000e+00 |
| Public bond reduction alternatives | 100 bonds | repeat + bincount | reduceat | 0.001308 | 0.001218 | 1.07x | 1.421e-14 |
| Public bond reduction alternatives | 100 bonds | repeat + bincount | current dispatch | 0.001308 | 0.001373 | 0.95x | 0.000e+00 |
| Public bond reduction alternatives | 100 bonds | repeat + bincount | prefix differences | 0.001308 | 0.001180 | 1.11x | 5.969e-12 |
| Public bond reduction alternatives | 1,000 bonds | repeat + bincount | reduceat | 0.012215 | 0.011507 | 1.06x | 1.421e-14 |
| Public bond reduction alternatives | 1,000 bonds | repeat + bincount | current dispatch | 0.012215 | 0.011189 | 1.09x | 1.421e-14 |
| Public bond reduction alternatives | 1,000 bonds | repeat + bincount | prefix differences | 0.012215 | 0.012357 | 0.99x | 9.499e-11 |
| Public bond reduction alternatives | 10,000 bonds | repeat + bincount | reduceat | 0.137317 | 0.120946 | 1.14x | 1.421e-14 |
| Public bond reduction alternatives | 10,000 bonds | repeat + bincount | current dispatch | 0.137317 | 0.119005 | 1.15x | 1.421e-14 |
| Public bond reduction alternatives | 10,000 bonds | repeat + bincount | prefix differences | 0.137317 | 0.147423 | 0.93x | 7.524e-10 |

## Numerical acceptance

`Max difference` is the largest absolute difference over the compared outputs. Numerical and financial-unit tolerances vary by quantity and workload; see tests/native, tests/finance and docs/validation.md. A small absolute difference alone is not an acceptance criterion. Correctness tests are run before accepting a measured implementation change.

## Interpretation

Native dispatch is useful only when the eliminated inner loop exceeds buffer-conversion cost. The benchmark deliberately exposes crossover cases; a speedup below 1.0x means the reference was faster for that measured size.
The quantum row compares PennyLane devices only; Lightning C++ remains the quantum simulator and is independent of QFin's finance C++ extension.
SciPy has no separate row because these reference cases use vectorized NumPy or an explicit batch bisection; no SciPy primitive is used.
Timings are environment-specific measurements, not performance guarantees.

## Cancellation microbenchmark

| Method | Seconds | Absolute error |
| --- | ---: | ---: |
| naive Python | 0.001357 | 0.203451 |
| NumPy pairwise | 0.000007 | 0.193929 |
| Kahan Python | 0.002308 | 0.203451 |
| Neumaier Python | 0.002894 | 0 |
| QFin selective fsum | 0.001641 | 0 |

The fsum reference is 3,333.333333333333 for the supplied float products. Private reduction timings are not public-API speedups. Allocation peaks and the prefix-cancellation counterexample are recorded in JSON.
