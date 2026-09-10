# QFin performance evidence

Private batched-kernel investigation; these are not public-API speedup measurements.

## Environment

- Measurement date (UTC): 2026-09-09T17:21:54.377606+00:00
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
| Private native output investigation | 100,000 scenarios x one zero-time cash flow | Audited-main vector then array | Current NumPy-owned output | 0.001129 | 0.001235 | 0.91x | 0.000e+00 |
| Private native output investigation | 1,000,000 scenarios x one zero-time cash flow | Audited-main vector then array | Current NumPy-owned output | 0.011196 | 0.012009 | 0.93x | 0.000e+00 |

## Numerical acceptance

`Max difference` is the largest absolute difference over the compared outputs. Numerical and financial-unit tolerances vary by quantity and workload; see tests/native, tests/finance and docs/validation.md. A small absolute difference alone is not an acceptance criterion. Correctness tests are run before accepting a measured implementation change.

## Interpretation

Native dispatch is useful only when the eliminated inner loop exceeds buffer-conversion cost. The benchmark deliberately exposes crossover cases; a speedup below 1.0x means the reference was faster for that measured size.
The quantum row compares PennyLane devices only; Lightning C++ remains the quantum simulator and is independent of QFin's finance C++ extension.
SciPy has no separate row because these reference cases use vectorized NumPy or an explicit batch bisection; no SciPy primitive is used.
Timings are environment-specific measurements, not performance guarantees.

Private batched-call investigation, not a public API speedup. Current code also validates stressed-node discounts and output finiteness. The difference cannot be attributed entirely to output-copy removal. Retained change saves one scenario-length vector and copy; no capsule ownership transfer is used.
