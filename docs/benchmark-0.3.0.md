# QFin 0.3.0 numerical demonstration

Generated on 2026-08-27 with:

```bash
python examples/recruiter_benchmark.py
```

Each case uses PennyLane `default.qubit`, maximum-likelihood amplitude
estimation, 1,000 shots for each Grover power in `(0, 1, 2)`, and a fixed seed.
`Shots` is the total across the three circuits. Logical qubit counts are
pre-transpilation estimates.

| Case | Qubits | Black-Scholes | QFin | Abs. error | Shots | Walsh terms | Retained | Logical qubits |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ATM call | 6 | 9.4134 | 9.3692 | 0.0442 | 3,000 | 53 | 82.8% | 8 |
| ATM put | 5 | 6.4580 | 6.4277 | 0.0303 | 3,000 | 26 | 81.2% | 7 |
| OTM call | 6 | 10.6707 | 10.8008 | 0.1301 | 3,000 | 58 | 90.6% | 8 |
| ITM put | 5 | 14.2648 | 14.2341 | 0.0308 | 3,000 | 17 | 53.1% | 7 |

These simulator results demonstrate a reproducible finance-to-circuit
workflow. They do not establish quantum advantage, hardware performance, or
production pricing accuracy.
