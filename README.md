# QFin

QFin is an experimental Python framework that translates familiar financial
models into quantum representations and executable quantum circuits. It sits
above PennyLane: users describe an option and market model, while QFin chooses
the terminal distribution, truncation bounds, grid, qubit count, payoff
normalization, amplitude-estimation circuit, and validation report.

Version `0.3.0` solves one problem end to end: European call and put pricing
under Black–Scholes using maximum-likelihood amplitude estimation (MLAE) on
PennyLane's `default.qubit` simulator. Its default representation labels
equal-probability inverse-CDF points with a uniform quantum register and
compiles the payoff into a tolerance-controlled sparse Walsh/Pauli circuit.
No arbitrary dense unitary or probability-loading angle table is constructed.

## Install

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
python -m pip install -e ".[quantum]"
```

Python 3.11 or newer is supported. NumPy and SciPy are core dependencies;
PennyLane is optional so the financial and compilation layers can be used
without a quantum runtime.

## First price

```python
import qfin

market = qfin.BlackScholes(
    spot=100,
    rate=0.04,
    volatility=0.20,
)
option = qfin.EuropeanCall(strike=105, maturity=1.0)

model = qfin.compile(
    option,
    market,
    target_error=0.10,
    max_qubits=8,
)

print(model.explain())
result = model.run(shots=2_000, schedule=(0, 1, 2, 4), seed=7)

print(f"Quantum estimate:   {result.value:.4f}")
print(f"Black–Scholes:      {result.classical_value:.4f}")
print(f"Absolute error:     {result.absolute_error:.4f}")
print(result.resources.to_dict())
```

Inspect the generated circuit directly:

```python
backend = model.to_pennylane()  # mode="compressed" is selected automatically
print(backend.draw(power=1))
print(backend.circuit_specs(power=1))
```

Or run the bundled command:

```bash
qfin price --kind call --spot 100 --strike 105 --maturity 1 \
  --rate 0.04 --volatility 0.20 --shots 2000 --seed 7
```

## What the compiler does

1. Converts the Black–Scholes market into the terminal lognormal distribution.
2. Selects a truncated quantile domain and midpoint inverse-CDF quadrature.
3. Increases qubits until the payoff expectation stabilizes or the configured
   simulator limit is reached.
4. Normalizes the payoff into an objective-qubit amplitude.
5. Loads the equal-probability quantile labels with one Hadamard per data qubit.
6. Fits the payoff rotation with magnitude-ordered Walsh coefficients until
   both the financial error and angle-RMSE tolerances are met.
7. Implements each retained coefficient as a commuting Pauli-string rotation.
8. Constructs Grover reflections from X, H, Z, and multi-controlled-X gates.
9. Runs shot-based MLAE and maps the amplitude back to a discounted price.
10. Separately reports representation, payoff-approximation, estimation, and
    total errors against Black–Scholes.

## Compressed and reference backends

`model.to_pennylane()` returns the v0.3 compressed backend for the default
quantile representation. The exact v0.2 structured backend and v0.1 dense
Householder backend remain available as numerical references:

```python
compressed = model.to_pennylane(mode="compressed")
structured = model.to_pennylane(mode="structured")
dense = model.to_pennylane(mode="dense")
```

The compressed circuit is intentionally approximate and reports its error
against the exact discrete expectation. The structured and dense circuits
remain exactly equivalent. To retain the v0.2 probability-mass encoding in a
compiled model, use `representation_method="probability"`; its automatic
backend is `structured`.

Compression can be bounded explicitly:

```python
model = qfin.compile(
    option,
    market,
    target_error=0.10,
    payoff_angle_tolerance=0.10,
    payoff_max_terms=64,
)
print(model.payoff_approximation.met_tolerance)
```

The compiler returns the best circuit within the cap and clearly reports when
the requested tolerances were not reached.

## Honest scope

This release is a research prototype, not a production pricer and not evidence
of quantum advantage. Quantile state preparation is `O(n)`, but interpreting
the basis through the inverse CDF moves distribution complexity into the
payoff function. Walsh compression is problem- and tolerance-dependent and can
still retain all `2**n` coefficients. The report never labels a capped fit as
converged unless it meets both requested criteria. Resource counts are logical
and PennyLane-level rather than hardware-transpiled counts. QFin does not yet
support calibration, path-dependent products, stochastic volatility,
portfolio risk, hardware noise, Qiskit export, or production controls.

See [docs/circuit-design.md](docs/circuit-design.md) for the v0.3 circuit,
[docs/architecture.md](docs/architecture.md) for package design, and
[docs/roadmap.md](docs/roadmap.md) for the next research steps.

## Reproducible numerical demonstration

[docs/benchmark-0.3.0.md](docs/benchmark-0.3.0.md) compares several call and
put cases with Black-Scholes and reports QFin estimates, absolute errors,
shots, retained Walsh terms, and logical-resource counts. The table is emitted
by [examples/recruiter_benchmark.py](examples/recruiter_benchmark.py); it is a
simulator demonstration, not evidence of quantum advantage.

## Development

```bash
python -m pip install -e ".[dev]"
ruff check .
mypy src/qfin
pytest --cov=qfin --cov-fail-under=78
python -m build
```
