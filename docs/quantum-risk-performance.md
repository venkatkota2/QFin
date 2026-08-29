# QFin Quantum-Risk Simulator Performance

These are measured simulator timings, not a hardware or quantum-advantage claim.

## Environment

- CPU/platform: AMD EPYC 9V74 80-Core Processor
- OS: Linux-6.18.35-x86_64-with-glibc2.39
- Python: 3.12.13
- QFin: 0.5.0
- NumPy: 2.5.2
- PennyLane: 0.45.1
- PennyLane-Lightning: 0.45.0
- Repeats: 3 (median reported)
- Shots per circuit: 1000
- MLAE schedule: `(0, 1, 2)`
- Data qubits: 5 (32 encoded grid points)

## Results

| Problem | Device | Wall time (s) | Speedup | Quantum estimate | Classical reference | Circuits | Max state error |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Tail probability | default.qubit | 0.412689 | 1.00x | 0.18480861 | 0.18750000 | 3 | 2.498e-16 |
| VaR | default.qubit | 1.766280 | 1.00x | 0.99407869 | 0.99407869 | 12 | 2.498e-16 |
| CVaR | default.qubit | 2.143679 | 1.00x | 3.02062802 | 2.98666561 | 15 | 2.498e-16 |
| Tail probability | lightning.qubit | 0.118210 | 3.49x | 0.18967061 | 0.18750000 | 3 | 3.331e-16 |
| VaR | lightning.qubit | 0.270025 | 6.54x | 0.99407869 | 0.99407869 | 12 | 3.331e-16 |
| CVaR | lightning.qubit | 0.348164 | 6.16x | 2.97504305 | 2.98666561 | 15 | 3.331e-16 |

The VaR/CVaR workflow uses hybrid binary search. Its circuit count depends on
the number of occupied encoded loss points. CVaR adds one normalized tail-excess
objective after threshold search.

The generic empirical probability tree and objective multiplexers require `O(2**data_qubits)` rotations. The benchmark demonstrates correctness and backend integration; it does not establish an asymptotic speedup.
