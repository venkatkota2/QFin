# Changelog

## Unreleased

- Aligned package and README licensing metadata with the repository's current
  no-license state.
- Added a reproducible multi-case numerical demonstration and a CI coverage
  regression floor.

## 0.3.0 — 2026-08-24

- Added inverse-CDF quantile encoding, preparing the terminal lognormal
  register with one Hadamard per data qubit and no distribution-angle table.
- Added a sparse Walsh/Pauli payoff compiler controlled by financial price
  error, angle RMSE, and an optional hard term cap.
- Added the compressed PennyLane backend and retained the v0.2 structured and
  v0.1 dense implementations as exact references.
- Separated representation, payoff-approximation, circuit-estimation, and
  total benchmark errors in compiler and runtime results.
- Expanded resource reports with distribution gate counts and payoff
  compression ratios.

## 0.2.0 — 2026-08-24

- Replaced the default dense Householder circuit with a binary probability-tree
  distribution loader made from multiplexed Y rotations.
- Added uniformly controlled payoff rotations and gate-level Grover
  reflections with one reusable work qubit.
- Retained the v0.1 dense backend as an explicit numerical reference.
- Added structured-versus-dense equivalence tests and PennyLane circuit specs.
- Expanded resource reporting to distinguish parameters, work qubits, and
  avoided dense-unitary entries.

## 0.1.0 — 2026-08-24

- Added the European call/put compiler, lognormal discretization, MLAE,
  Black-Scholes validation, the PennyLane simulator backend, CLI, and tests.
