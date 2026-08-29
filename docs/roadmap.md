# Roadmap

## 0.1 to 0.3 — European-option quantum vertical slice (complete)

- calls and puts under Black-Scholes;
- probability and inverse-CDF quantile representations;
- structured and sparse Walsh/Pauli payoff circuits;
- PennyLane execution with `lightning.qubit` by default;
- MLAE, error decomposition, validation, and logical resources.

## 0.4 — native finance and ALM foundation (complete)

- scikit-build-core, CMake, pybind11, and a wheel-compatible C++20 extension;
- NumPy fallback and explicit `auto`/`numpy`/`native` execution modes;
- yield curves, fixed/zero-coupon cash flows, batch pricing, yield solving,
  duration, convexity, DV01, accrued interest, and clean/dirty prices;
- asset/liability portfolios, funding and immunization measures;
- chunked parallel, twist, and key-rate scenarios;
- mortality tables and annual term-life projection;
- weighted loss aggregation, VaR, and expected shortfall;
- ALM scenario losses connected to the existing distribution representation;
- capability metadata that distinguishes classical, native, representation,
  and quantum-algorithm availability;
- Python/native parity tests and reproducible end-to-end benchmarks.

## 0.5 — risk algorithm and compiler policy

- tail-probability, VaR, and CVaR quantum oracle design;
- distribution/oracle error budgets and classical confidence intervals;
- compiler selection across option pricing and loss-distribution problems;
- state-preparation complexity and classical preprocessing cost reports;
- correlation/dependence assumptions for multi-factor portfolio losses.

## 0.6 — richer ALM and life products

- credit-spread, equity, inflation, mortality, and lapse scenario factors;
- stochastic rates and multi-period reinvestment/rebalancing;
- participating, universal-life, annuity, and multi-state policy foundations;
- assumption sets, model-point grouping, attribution, and sensitivity reports;
- chunked scenario × policy execution without full-cube allocation.

## 0.7 — device realism

- device gate-set decomposition and topology-aware resource reports;
- noise models, error mitigation, and repeatable backend benchmarks;
- additional PennyLane devices only when implemented and tested;
- Qiskit export and hardware-provider capability checks.

## Later research

- portfolio optimization and capital allocation;
- block encoding, QSVT, quantum PDE/SDE and linear-system methods;
- calibration, stochastic volatility, and path-dependent products;
- OpenMP/SIMD/Kokkos/CUDA only after portable-kernel profiling;
- production actuarial governance, validation, and audit workflows.

Every phase must retain classical validation, explicit error accounting, and a
clear distinction between simulator feasibility and quantum advantage.
