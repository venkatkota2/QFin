# QFin performance evidence

Measured end-to-end public-API timings, including object-to-buffer conversion and Python/C++ boundary costs.

## Environment

- Measurement date (UTC): 2026-09-09T17:24:01.015120+00:00
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

## Results

| Workload | Problem | Reference | Accelerated | Reference (s) | Accelerated (s) | Speedup | Max difference |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: |
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

## Numerical acceptance

`Max difference` is the largest absolute difference over the compared outputs. Numerical and financial-unit tolerances vary by quantity and workload; see tests/native, tests/finance and docs/validation.md. A small absolute difference alone is not an acceptance criterion. Correctness tests are run before accepting a measured implementation change.

## Interpretation

Native dispatch is useful only when the eliminated inner loop exceeds buffer-conversion cost. The benchmark deliberately exposes crossover cases; a speedup below 1.0x means the reference was faster for that measured size.
Chunked ALM scenario valuation observed speedups from 1.47x to 1.67x.
The quantum row compares PennyLane devices only; Lightning C++ remains the quantum simulator and is independent of QFin's finance C++ extension.
SciPy has no separate row because these reference cases use vectorized NumPy or an explicit batch bisection; no SciPy primitive is used.
Timings are environment-specific measurements, not performance guarantees.
