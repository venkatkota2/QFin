# QFin performance evidence

Measured end-to-end public-API timings, including object-to-buffer conversion and Python/C++ boundary costs.

## Environment

- Measurement date (UTC): 2026-09-09T05:35:31.312607+00:00
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
| Key-rate risk | 1 bonds x 5 nodes | Brute-force public repricing | Prepared QFin C++ | 0.001707 | 0.000060 | 28.33x | 6.702e-15 |
| Key-rate engine comparison | 1 bonds x 5 nodes | Prepared NumPy | Prepared QFin C++ | 0.000167 | 0.000060 | 2.77x | 0.000e+00 |
| Key-rate risk | 100 bonds x 5 nodes | Brute-force public repricing | Prepared QFin C++ | 0.017423 | 0.001596 | 10.92x | 4.263e-14 |
| Key-rate engine comparison | 100 bonds x 5 nodes | Prepared NumPy | Prepared QFin C++ | 0.002488 | 0.001596 | 1.56x | 4.263e-14 |
| Key-rate risk | 100 bonds x 17 nodes | Brute-force public repricing | Prepared QFin C++ | 0.047350 | 0.002461 | 19.24x | 5.684e-14 |
| Key-rate engine comparison | 100 bonds x 17 nodes | Prepared NumPy | Prepared QFin C++ | 0.002994 | 0.002461 | 1.22x | 5.684e-14 |
| Key-rate risk | 1,000 bonds x 17 nodes | Brute-force public repricing | Prepared QFin C++ | 0.476149 | 0.026324 | 18.09x | 5.684e-14 |
| Key-rate engine comparison | 1,000 bonds x 17 nodes | Prepared NumPy | Prepared QFin C++ | 0.036522 | 0.026324 | 1.39x | 5.684e-14 |
| Key-rate risk | 1,000 bonds x 33 nodes | Brute-force public repricing | Prepared QFin C++ | 0.830206 | 0.042452 | 19.56x | 5.684e-14 |
| Key-rate engine comparison | 1,000 bonds x 33 nodes | Prepared NumPy | Prepared QFin C++ | 0.051077 | 0.042452 | 1.20x | 5.684e-14 |
| Key-rate risk | 5,000 bonds x 17 nodes | Brute-force public repricing | Prepared QFin C++ | 2.294117 | 0.130027 | 17.64x | 5.684e-14 |
| Key-rate engine comparison | 5,000 bonds x 17 nodes | Prepared NumPy | Prepared QFin C++ | 0.217268 | 0.130027 | 1.67x | 5.684e-14 |
| Key-rate risk | 5,000 bonds x 33 nodes | Brute-force public repricing | Prepared QFin C++ | 4.183859 | 0.217338 | 19.25x | 5.684e-14 |
| Key-rate engine comparison | 5,000 bonds x 33 nodes | Prepared NumPy | Prepared QFin C++ | 0.431745 | 0.217338 | 1.99x | 5.684e-14 |
| Key-rate risk | 10,000 bonds x 33 nodes | Brute-force public repricing | Prepared QFin C++ | 8.150372 | 0.423941 | 19.23x | 5.684e-14 |
| Key-rate engine comparison | 10,000 bonds x 33 nodes | Prepared NumPy | Prepared QFin C++ | 0.827241 | 0.423941 | 1.95x | 5.684e-14 |

## Numerical acceptance

`Max difference` is the largest absolute difference over the compared outputs. Numerical and financial-unit tolerances vary by quantity and workload; see tests/native, tests/finance and docs/validation.md. A small absolute difference alone is not an acceptance criterion. Correctness tests are run before accepting a measured implementation change.

## Interpretation

Native dispatch is useful only when the eliminated inner loop exceeds buffer-conversion cost. The benchmark deliberately exposes crossover cases; a speedup below 1.0x means the reference was faster for that measured size.
The quantum row compares PennyLane devices only; Lightning C++ remains the quantum simulator and is independent of QFin's finance C++ extension.
SciPy has no separate row because these reference cases use vectorized NumPy or an explicit batch bisection; no SciPy primitive is used.
Timings are environment-specific measurements, not performance guarantees.
