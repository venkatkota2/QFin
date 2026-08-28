# QFin 0.4 performance evidence

These are simulator/development benchmarks, not quantum-advantage claims. The
results below were produced on the hosted Linux development runner using Python
3.12.13, NumPy 2.5.2, PennyLane 0.45.1, and PennyLane Lightning 0.45.0. Absolute
timings will vary by CPU, BLAS, thread settings, and workload.

## ALM scenarios

Command:

```bash
python examples/benchmark_alm.py
```

The model contains 20 fixed-rate bond positions, 10 life-policy cohorts, and a
41-year cash-flow horizon. A run across 100,000 parallel rate shifts produced:

```text
scenarios=100,000
seconds=0.566181
scenarios_per_second=176,622
```

The scenario engine pre-aggregates cash flows, reuses base discount factors,
and evaluates matrix blocks under a 64 MiB temporary-memory ceiling. The exact
throughput is machine-specific; the reproducible property is that valuation is
vectorized by block rather than looped once per scenario.

## Walsh objective compiler

Command:

```bash
python examples/benchmark_walsh.py
```

For 65,536 objective points, the local run compared the earlier per-block
Python loop with the new reshape/vectorized transform:

```text
reference_seconds=0.127930
optimized_seconds=0.002094
speedup=61.11x
max_error=0.000e+00
```

The benchmark fixes the random seed and verifies exact numerical equality. It
measures only the Walsh-Hadamard transform, not full circuit construction or
simulation.

## PennyLane runtime

Every MLAE schedule now creates one PennyLane device and reuses it across all
Grover powers. `lightning.qubit` remains the default compiled C++ state-vector
device. Tests assert one device creation per schedule and compare every quantum
ALM value with the exact classical scenario expectation. Cold import/plugin
cost and circuit complexity can still dominate small examples, so QFin does
not report a universal Lightning speedup factor.
