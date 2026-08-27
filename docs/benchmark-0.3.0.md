# QFin 0.3.0 numerical demonstration

Generated on 2026-08-27 with:

```bash
python examples/recruiter_benchmark.py
```

Each case uses PennyLane `lightning.qubit`, maximum-likelihood amplitude
estimation, 1,000 shots for each Grover power in `(0, 1, 2)`, and a fixed seed.
`Shots` is the total across the three circuits. Logical qubit counts are
pre-transpilation estimates.

| Case | Qubits | Black-Scholes | QFin | Abs. error | Shots | Walsh terms | Retained | Logical qubits |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ATM call | 5 | 9.4134 | 9.3942 | 0.0192 | 3,000 | 30 | 93.8% | 7 |
| ATM put | 4 | 6.4580 | 6.4122 | 0.0458 | 3,000 | 15 | 93.8% | 6 |
| OTM call | 6 | 10.6707 | 10.4807 | 0.1900 | 3,000 | 58 | 90.6% | 8 |
| ITM put | 4 | 14.2648 | 14.0320 | 0.2328 | 3,000 | 13 | 81.2% | 6 |

These simulator results demonstrate a reproducible finance-to-circuit
workflow. They do not establish quantum advantage, hardware performance, or
production pricing accuracy.
