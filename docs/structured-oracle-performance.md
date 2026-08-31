# QFin 0.9 structured-oracle performance

All values below were produced by `examples/structured_oracle_benchmark.py`.
They are medians of end-to-end public compiler/runtime calls; no result is fabricated.

## Environment

- CPU: x86_64
- OS: Linux-6.18.35-x86_64-with-glibc2.39
- Python: 3.12.13
- QFin: 0.9.0
- NumPy: 2.5.2
- PennyLane: 0.45.1
- PennyLane-Lightning: 0.45.0
- Repeats: 3

## Streaming construction and validation

| Qubits/factor | Joint points | Marginal points | Compile + validate (s) | Structured values | Generic values | Generic / structured | Oracle p error | Max loss error |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 4 | 4 | 0.000788 | 16 | 11 | 0.69x | 0.000e+00 | 3.625e-01 |
| 2 | 16 | 8 | 0.000685 | 26 | 47 | 1.81x | 0.000e+00 | 1.861e+00 |
| 3 | 64 | 16 | 0.002490 | 58 | 191 | 3.29x | 0.000e+00 | 3.702e-01 |
| 4 | 256 | 32 | 0.005023 | 105 | 767 | 7.30x | 0.000e+00 | 7.813e-02 |
| 5 | 1,024 | 64 | 0.012656 | 183 | 3,071 | 16.78x | 1.557e-02 | 6.469e-02 |

Validation streams every encoded point in bounded chunks. Memory is bounded, but validation time remains exponential in total factor qubits.

## Portable target comparison (two 1-qubit factors, power 0)

| Topology | Analysis (s) | Structured gates | Generic gates | Structured depth | Generic depth | Structured swaps | Generic swaps | Gate ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| all_to_all | 0.273455 | 1,457 | 26 | 955 | 20 | 0 | 0 | 56.04x |
| linear | 4.809226 | 2,789 | 29 | 2,166 | 23 | 444 | 1 | 96.17x |

The generic comparison is deliberately limited to four joint points. At this tiny size reversible arithmetic can use more gates than a lookup-style loader; the structured benefit is avoiding exponentially stored joint probability/payoff data, not a promise of lower gate count or quantum advantage.

## Power-0 simulator execution

| Device | Median (s) | Probability | Absolute difference |
| --- | ---: | ---: | ---: |
| default.qubit | 0.016799 | 0.500000000000 | 1.665e-16 |
| lightning.qubit | 0.012689 | 0.500000000000 | 1.277e-15 |

QFin constructs the finance-specific arithmetic. PennyLane-Lightning performs the compiled state-vector simulation. These timings are simulator measurements, not hardware or fault-tolerant runtime estimates.
