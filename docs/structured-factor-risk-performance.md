# QFin 1.0 structured factor-risk benchmark

Measured on 2026-08-31 with
`python examples/structured_factor_risk_benchmark.py`. These are observed
wall-clock results from one environment, not promised performance.

- OS: Linux 6.18.35 x86_64, glibc 2.39
- CPU reported by `platform`: x86_64
- Python: 3.12.13
- NumPy: 2.5.2
- QFin: 1.0.0
- Native compiler: GNU 13.3.0
- Quantum device: `lightning.qubit`

## Classical exact-reference scaling

| Factors x marginal points | Joint points | Generic NumPy (s) | Streamed exact (s) | Generic stored values | Structured input values | VaR difference | CVaR difference |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 3 x 2 | 8 | 0.000092 | 0.001925 | 16 | 12 | 0.000e+00 | 0.000e+00 |
| 3 x 4 | 64 | 0.000072 | 0.002093 | 128 | 24 | 0.000e+00 | 5.329e-14 |
| 3 x 8 | 512 | 0.000098 | 0.002778 | 1,024 | 48 | 0.000e+00 | 2.265e-13 |
| 3 x 16 | 4,096 | 0.000286 | 0.007747 | 8,192 | 96 | 0.000e+00 | 1.599e-13 |
| 4 x 16 | 65,536 | 0.008797 | 0.220957 | 131,072 | 128 | 0.000e+00 | 8.698e-12 |

Generic timing includes joint loss/probability construction plus NumPy risk
aggregation. Structured timing is the exact memory-bounded reference, which
uses repeated streamed CDF passes. The data show no speed crossover: the
materialized NumPy path is faster at every measured size. The structured
reference exists to bound memory and validate the reversible loss register,
not to claim classical acceleration.

The stored-value columns isolate model inputs: generic storage counts one loss
and probability per joint point; structured storage counts each marginal grid
and probability. Runtime chunks and the bounded fixed-point code histogram are
additional working memory and are reported separately by the compiled model.

## Structured circuit microbenchmark

Circuit timings are medians of five runs after one warm-up execution.

| Measurement | Result |
| --- | ---: |
| Compile | 0.00383715 s |
| Tail comparator circuit | 0.00218447 s |
| Excess-bit circuit | 0.00644295 s |
| Tail probability absolute difference | 7.21645e-16 |
| Excess-bit probability absolute difference | 1.16573e-15 |
| Loss qubits | 4 |
| Maximum runtime qubits | 11 |

The excess circuit is wider and includes comparison plus controlled
subtraction, so this tiny simulator case is slower than the threshold-only
circuit. These timings exclude the first device warm-up but include QNode
construction/execution inside each call. They are simulator measurements, not
hardware latency or evidence of quantum advantage.

## Interpretation

- Numerical parity held to at most `8.698e-12` across the measured classical
  cases.
- Factorized input storage grows with the sum of marginal sizes, while a
  generic input loss table grows with their product.
- The bounded-memory validation deliberately pays substantial repeated-pass
  time. Automatic dispatch does not replace small NumPy risk aggregation with
  this path.
- Reversible CVaR has a visible circuit cost: one excess register and one MLAE
  objective per loss bit.

Re-run the script on another system before making local performance decisions.
