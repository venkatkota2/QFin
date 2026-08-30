# QFin 0.8 scalable-representation performance

All timings below come from public QFin APIs and are medians of 5 runs after one warm-up.

## Environment

- OS: Linux-6.18.35-x86_64-with-glibc2.39
- CPU: AMD EPYC 9V74 80-Core Processor
- Python: 3.12.13
- QFin: 0.8.0
- NumPy: 2.5.2
- Factor marginals: standard normal probability encodings
- Optimization solver: SciPy SLSQP with analytical gradient

## Factorized construction

| Factors | Qubits/factor | Joint points | Stored marginal points | Factorized angles | Flattened angles | Factorized build (s) | Flattened build (s) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 3 | 64 | 16 | 14 | 63 | 0.000088 | 0.000309 |
| 3 | 4 | 4,096 | 48 | 45 | 4,095 | 0.000118 | 0.016702 |
| 4 | 5 | 1,048,576 | 128 | 124 | 1,048,575 | 0.000143 | not materialized |

The largest flattened case is deliberately not allocated. Its angle and memory counts are analytical properties of the represented dimensions, not fabricated timings. Factorized construction stores and prepares each marginal independently.

## Classical mean-variance baseline

| Assets | Solve time (s) | Budget residual | Utility improvement vs feasible start | Iterations |
| ---: | ---: | ---: | ---: | ---: |
| 10 | 0.000838 | 0.000e+00 | 1.490457e-02 | 17 |
| 25 | 0.002591 | 2.220e-16 | 2.560660e-02 | 19 |
| 50 | 0.010826 | 4.441e-16 | 2.772690e-02 | 21 |

## Block-encoding feasibility analysis

A 50x50 covariance analysis took 0.000487 s. Hermitian=True, PSD=True, condition number=11.7005.

QFin does not construct a block-encoding oracle or execute QSVT. This timing covers classical feasibility metadata only.

## Interpretation

The factorized loader removes joint probability-table construction where independence or latent-factor structure permits. It does not remove the cost of a general multivariate payoff oracle, and it is not evidence of quantum advantage. Optimization remains classical because QFin has no validated quantum portfolio optimizer.

Reproduce with:

```bash
python examples/scalable_representation_benchmark.py --repeats 5 \
  --output docs/scalable-representation-performance.md
```
