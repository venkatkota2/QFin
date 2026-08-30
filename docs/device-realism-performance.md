# QFin 0.7 device-realism performance

All values below were measured by the public QFin API. Timings are medians of 5 runs after one warm-up.

## Environment

- OS: Linux-6.18.35-x86_64-with-glibc2.39
- CPU: AMD EPYC 9V74 80-Core Processor
- Python: 3.12.13
- QFin: 0.7.0
- NumPy: 2.5.2
- PennyLane: 0.45.1
- PennyLane-Lightning: 0.45.0
- Qiskit: 2.5.2
- Circuit: compressed three-data-qubit European-call objective
- Native gate set: RX, RY, RZ, CNOT

## Ideal simulator execution

| Grover power | default.qubit (s) | lightning.qubit (s) | Lightning speedup | Probability difference |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0.002641 | 0.002078 | 1.27x | 0.000e+00 |
| 1 | 0.008664 | 0.005660 | 1.53x | 2.220e-16 |

## Decomposition and routing

| Target | Analysis time (s) | Routed gates, powers 0+1 | Max depth | Inserted SWAPs | Max two-qubit gates |
| --- | ---: | ---: | ---: | ---: | ---: |
| all_to_all | 0.021470 | 311 | 191 | 0 | 96 |
| linear | 0.115056 | 479 | 275 | 56 | 225 |

The targets are synthetic research topologies, not vendor devices. Analysis time is compiler preprocessing, not quantum execution time.

## Synthetic noise and mitigation

| Experiment | Probability | Absolute error vs ideal |
| --- | ---: | ---: |
| Ideal | 0.216384486 | 0.000e+00 |
| Local noise | 0.220856961 | 4.472e-03 |
| Linear ZNE | 0.217641170 | 1.257e-03 |

End-to-end three-scale noise-analysis time: 0.069246 s.

This uses analytic `default.mixed`, local per-wire depolarizing probability 0.001 after each gate, readout bit-flip probability 0.002, global folding at 1x/3x/5x, and first-order extrapolation. It is not a hardware prediction.

## Interoperability

- Linear-target OpenQASM export: 0.092684 s, 385 gates, SHA-256 `79fe11987f1bb0940904edca40f3a32a630f4b59771cf7eba272a6c3b5591915`.
- Qiskit parse: 0.079388 s, 390 operations including terminal measurements.

## Interpretation

These measurements establish numerical parity and expose topology/noise costs. They do not establish quantum advantage, hardware feasibility, or a stable simulator speedup at every small circuit size.

Reproduce with:

```bash
python examples/device_realism_benchmark.py --repeats 5 \
  --output docs/device-realism-performance.md
```
