# Reproducibility contract

QFin 1.1.2 distinguishes repeatability within an environment from numerical
agreement across environments. Pin package versions, retain inputs and conventions,
record `qfin.system_info()`, and keep the result's engine/methodology metadata.

| Calculation | Contract |
| --- | --- |
| Deterministic financial valuation | Same inputs and execution path are repeatable within one environment. Across CPU/compiler/BLAS versions, use documented numerical and financial-unit tolerances; bitwise equality is not promised. |
| Scenario simulation and empirical bootstrap | An explicit seed repeats draws under the same NumPy version, RNG and call sequence. Bootstrap results retain seed, resample count, sample size and interval method. Changing batching or RNG versions can change draws. |
| Quantum shot execution | Seed, device, PennyLane/Lightning versions, shot count, schedule, representation and likelihood precision jointly define an experiment. Different simulators need not produce identical samples from the same seed. |
| Automatic classical dispatch | Cheap static thresholds in `_dispatch.py`; no machine benchmark during import, construction or execution. Explicit `engine` overrides remain supported. |

Quantum pricing, empirical risk, structured tail and structured risk result
`to_dict()` methods include `provenance` and `interval_semantics`. Existing fields,
positional constructors and defaults remain compatible. Provenance includes package
versions, resolved device, seed, shots per circuit, MLAE powers, representation,
data qubits, likelihood-grid size and precision settings. It describes execution;
it is not a guarantee that the selected seed achieves an error target. Missing
distribution metadata is represented by `null` rather than a guessed version.

For comparable measurements set `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`,
`MKL_NUM_THREADS=1` and `PYTHONHASHSEED=0` before starting Python. QFin's own C++
kernels remain single threaded. These environment variables do not assert control
over every third-party device implementation. Preserve the native compiler and
effective compile commands as well as the package list.

QFin does not enable fast-math. GCC/Clang builds use `-fno-fast-math` and
`-ffp-contract=off`; MSVC uses `/fp:strict`. Build type owns optimization flags;
sanitizer builds append `-O1 -g` and disable IPO. This supports reliable finite-value
checks and reproducible reductions without claiming portable bit patterns.

The [statistical validation](statistical-validation.md) explains conditional and
local intervals. Logical resource reports describe circuit counts, not measured
hardware performance or quantum advantage.
