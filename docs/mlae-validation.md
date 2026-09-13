# QFin MLAE validation

This validates QFin's existing maximum-likelihood amplitude-estimation algorithm; it does not introduce a new quantum algorithm or claim quantum advantage.

## Environment

- Os: Linux-6.18.35-x86_64-with-glibc2.39
- Architecture: x86_64
- Cpu: AMD EPYC 9V74 80-Core Processor
- Python: 3.12.13
- Qfin: 1.1.1
- Numpy: 2.5.2
- Scipy: 1.18.1
- Pennylane: 0.45.1
- Pennylane Lightning: 0.45.0
- Compiler: GNU 13.3.0
- Compiler Flags: release defaults; see cpp/CMakeLists.txt
- Thread Count: 1
- Timing Repetitions: 5
- Coverage Repetitions Per Case: 200
- Seed: 20260905

## Optimizer

| Workload | Dense reference (s) | Hardened QFin (s) | Speedup | Max amplitude difference |
| --- | ---: | ---: | ---: | ---: |
| 31 thresholds; schedule=(0,1,2,4); 1000 shots/circuit | 0.517088 | 0.052363 | 9.88x | 5.656e-06 |

The hardened timing includes the disjoint likelihood-ratio calculation and finite-sample confidence guard; the dense reference performs only the legacy global point search. Thus the comparison does not omit new uncertainty work.

## Seeded empirical 95% coverage

| Schedule | Shots | Amplitude | Guarded region | Raw LR region |
| --- | ---: | ---: | ---: | ---: |
| `(0,)` | 20 | 0.001 | 0.970 | 0.970 |
| `(0,)` | 20 | 0.010 | 0.990 | 0.990 |
| `(0,)` | 20 | 0.050 | 1.000 | 1.000 |
| `(0,)` | 20 | 0.250 | 0.955 | 0.955 |
| `(0,)` | 20 | 0.500 | 0.970 | 0.970 |
| `(0,)` | 20 | 0.750 | 0.945 | 0.945 |
| `(0,)` | 20 | 0.950 | 0.990 | 0.990 |
| `(0,)` | 20 | 0.990 | 0.995 | 0.995 |
| `(0,)` | 20 | 0.999 | 0.990 | 0.990 |
| `(0, 1, 2, 4)` | 20 | 0.001 | 1.000 | 0.905 |
| `(0, 1, 2, 4)` | 20 | 0.010 | 0.980 | 0.950 |
| `(0, 1, 2, 4)` | 20 | 0.050 | 0.990 | 0.940 |
| `(0, 1, 2, 4)` | 20 | 0.250 | 1.000 | 1.000 |
| `(0, 1, 2, 4)` | 20 | 0.500 | 0.975 | 0.920 |
| `(0, 1, 2, 4)` | 20 | 0.750 | 1.000 | 1.000 |
| `(0, 1, 2, 4)` | 20 | 0.950 | 0.995 | 0.940 |
| `(0, 1, 2, 4)` | 20 | 0.990 | 0.990 | 0.945 |
| `(0, 1, 2, 4)` | 20 | 0.999 | 1.000 | 0.875 |
| `(0, 1, 2, 4)` | 100 | 0.001 | 0.995 | 0.945 |
| `(0, 1, 2, 4)` | 100 | 0.010 | 0.990 | 0.940 |
| `(0, 1, 2, 4)` | 100 | 0.050 | 0.995 | 0.970 |
| `(0, 1, 2, 4)` | 100 | 0.250 | 1.000 | 1.000 |
| `(0, 1, 2, 4)` | 100 | 0.500 | 0.980 | 0.965 |
| `(0, 1, 2, 4)` | 100 | 0.750 | 1.000 | 1.000 |
| `(0, 1, 2, 4)` | 100 | 0.950 | 0.995 | 0.970 |
| `(0, 1, 2, 4)` | 100 | 0.990 | 0.995 | 0.975 |
| `(0, 1, 2, 4)` | 100 | 0.999 | 0.995 | 0.945 |
| `(0, 1, 2, 4)` | 1000 | 0.001 | 0.985 | 0.940 |
| `(0, 1, 2, 4)` | 1000 | 0.010 | 1.000 | 0.970 |
| `(0, 1, 2, 4)` | 1000 | 0.050 | 0.980 | 0.960 |
| `(0, 1, 2, 4)` | 1000 | 0.250 | 0.995 | 0.995 |
| `(0, 1, 2, 4)` | 1000 | 0.500 | 0.990 | 0.955 |
| `(0, 1, 2, 4)` | 1000 | 0.750 | 1.000 | 1.000 |
| `(0, 1, 2, 4)` | 1000 | 0.950 | 0.985 | 0.945 |
| `(0, 1, 2, 4)` | 1000 | 0.990 | 0.975 | 0.960 |
| `(0, 1, 2, 4)` | 1000 | 0.999 | 0.990 | 0.960 |
| `(0, 2, 5)` | 100 | 0.001 | 0.970 | 0.945 |
| `(0, 2, 5)` | 100 | 0.010 | 0.990 | 0.965 |
| `(0, 2, 5)` | 100 | 0.050 | 0.970 | 0.925 |
| `(0, 2, 5)` | 100 | 0.250 | 0.990 | 0.970 |
| `(0, 2, 5)` | 100 | 0.500 | 1.000 | 0.930 |
| `(0, 2, 5)` | 100 | 0.750 | 0.990 | 0.940 |
| `(0, 2, 5)` | 100 | 0.950 | 0.980 | 0.960 |
| `(0, 2, 5)` | 100 | 0.990 | 0.985 | 0.930 |
| `(0, 2, 5)` | 100 | 0.999 | 0.990 | 0.965 |

The raw region uses the one-parameter Wilks likelihood-ratio cutoff and is reported as diagnostic metadata. The main interval is its union with a simultaneous exact Clopper-Pearson set, which is deliberately conservative near boundaries and at low shot counts. This fixed-experiment interval is unchanged. Since 1.1.3, adaptive VaR/CVaR uses a separate workflow failure budget and quantile-error propagation; see [statistical validation](statistical-validation.md) for the sampling assumptions and exclusions.
