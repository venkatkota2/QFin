# Roadmap

## 0.1 — complete European-option vertical slice

- Calls and puts under Black–Scholes
- Normal, lognormal, and empirical distribution objects
- Automatic bounds, grid, and qubit selection
- PennyLane simulator circuits
- Maximum-likelihood amplitude estimation
- Black–Scholes validation and resource/error reports

## 0.2 — structured circuits (complete)

- Binary probability-tree distribution loading
- Multiplexed, gate-decomposable payoff rotations
- Gate-level zero-state and good-state reflections
- Structured backend as default; dense backend retained for validation
- Structured-versus-dense numerical equivalence tests
- PennyLane device-level circuit specifications

## 0.3 — quantile and compressed payoff circuits (complete)

- Inverse-CDF lognormal representation prepared with `n` Hadamards and no
  per-grid distribution angles
- Tolerance-controlled sparse Walsh/Pauli payoff circuits
- Separate representation, payoff-approximation, estimation, and total errors
- Explicit term caps and convergence diagnostics
- Compressed backend as default; exact earlier backends retained for validation

## 0.4 — fixed-income and life ALM (complete)

- Continuously compounded discount curves and fixed-rate bond portfolios
- Term and whole-life expected cash-flow projection from mortality tables
- Asset/liability present value, funding ratio, duration gap, and convexity
- Bounded-memory vectorized parallel-rate scenario engine
- Quantum shortfall probability and expected-shortfall compilation
- PennyLane Lightning device reuse, selectable precision, and portable fallback

## 0.5 — broader risk and actuarial modelling

- Lapse, expense inflation, reserves, reinsurance, and policyholder options
- Key-rate and stochastic interest-rate scenarios
- Discrete loss distributions, VaR, and CVaR
- Portfolio aggregation with explicit economic/mortality dependence assumptions

## 0.6 — device realism and compiler policy

- Device gate-set decomposition and topology-aware resource reports
- Noise models, error mitigation, and repeatable backend benchmarks
- Pluggable representation and algorithm selection policies

## Later research

- Correlated multi-factor models and multiple time steps
- Stochastic rates and stochastic volatility
- Optimization and capital allocation
- Block encoding, QSVT, quantum PDE/SDE methods, and calibration
- Qiskit export and hardware-provider capability checks

Each phase must retain classical validation, error accounting, and a clear
distinction between simulator feasibility and hardware practicality.
