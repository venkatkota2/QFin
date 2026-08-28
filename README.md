# QFin

[![CI](https://github.com/venkatkota2/QFin/actions/workflows/ci.yml/badge.svg)](https://github.com/venkatkota2/QFin/actions/workflows/ci.yml)

QFin is an experimental Python framework that translates familiar financial
models into quantum representations and executable quantum circuits. It sits
above PennyLane: users describe an option-pricing or asset-liability problem,
while QFin builds the financial cash flows and scenarios, chooses the quantum
representation, and returns circuit, error, validation, and resource reports.

Version `0.4.0` retains the end-to-end European call/put pipeline and adds
fixed-rate bonds, term and whole-life expected cash flows, duration/convexity
matching, high-throughput parallel-rate scenarios, and quantum ALM shortfall
estimation. Maximum-likelihood amplitude estimation (MLAE) runs on PennyLane's
compiled C++ `lightning.qubit` simulator by default. Uniform option quantiles
and uniform ALM scenario sets use parameter-free Hadamard loading plus a
tolerance-controlled sparse Walsh/Pauli objective circuit.

## Install

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
python -m pip install -e ".[quantum]"
```

Python 3.11 or newer is supported. NumPy and SciPy are core dependencies;
PennyLane is optional so the financial and compilation layers can be used
without a quantum runtime. The distribution name is `qfin-quantum` because the
`qfin` name on PyPI belongs to a different project; the Python import remains
`qfin`.

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

`lightning.qubit` is the default C++ simulator. Any installed PennyLane device
can be selected explicitly, for example
`model.run(device_name="default.qubit")`.

Or run the bundled command:

```bash
qfin price --kind call --spot 100 --strike 105 --maturity 1 \
  --rate 0.04 --volatility 0.20 --shots 2000 --seed 7 \
  --device lightning.qubit
```

## Fixed-income and life ALM

```python
import numpy as np
import qfin

curve = qfin.DiscountCurve.flat(0.04)
assets = qfin.FixedIncomePortfolio((
    qfin.BondPosition(qfin.FixedRateBond(1_000, 0.04, 10, 2), 600),
    qfin.BondPosition(qfin.FixedRateBond(1_000, 0.045, 20, 2), 450),
))

mortality = qfin.MortalityTable.illustrative_gompertz_makeham()
liabilities = qfin.LifePolicyPortfolio((
    qfin.PolicyPosition(qfin.TermLifePolicy(40, 25, 100_000), 80),
    qfin.PolicyPosition(qfin.WholeLifePolicy(50, 50_000), 40),
))

alm = qfin.AssetLiabilityModel(assets, liabilities, curve, mortality)
print(alm.evaluate().to_dict())

shocks = np.array([-0.03, -0.02, -0.01, 0.0, 0.01, 0.02, 0.03, 0.04])
scenarios = alm.run_parallel_shocks(shocks)
print(scenarios.expected_shortfall)

compiled_risk = qfin.compile_alm(
    alm,
    shocks,
    metric="expected_shortfall",
    target_error=25_000,
)
result = compiled_risk.run(shots=2_000, seed=7)
print(result.to_dict())
```

Classical cash-flow projection and scenario valuation stay in vectorized NumPy,
where they belong. PennyLane is used for the quantum-relevant scenario
expectation step. Equal-probability power-of-two scenario sets use the
compressed Hadamard/Walsh circuit; non-uniform or padded sets use the exact
probability-tree backend. See [docs/alm.md](docs/alm.md) for assumptions,
performance design, and current actuarial limitations.

## What the compiler does

1. Converts the Black–Scholes market into the terminal lognormal distribution.
2. Selects a truncated quantile domain and midpoint inverse-CDF quadrature.
3. Increases qubits until the discrete price is inside the combined domain and
   discretization allocation against the Black–Scholes MVP benchmark, or the
   configured simulator limit is reached.
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

This release is a research prototype, not a production pricer, actuarial
valuation engine, or evidence of quantum advantage. Quantile/scenario loading
can be `O(n)` for uniform registers, but objective synthesis is problem- and
tolerance-dependent and can still retain all `2**n` coefficients. ALM currently
uses deterministic expected mortality cash flows and parallel rate shifts; it
does not model lapses, reserves, policyholder options, credit, stochastic
rates, regulatory capital, or rebalancing. Resource counts are logical and
PennyLane-level rather than hardware-transpiled counts.

## Stack and performance

- NumPy and SciPy handle vectorized distribution, quadrature, Walsh-transform,
  likelihood, cash-flow, mortality, and chunked scenario calculations.
- PennyLane provides the circuit and backend abstraction.
- PennyLane Lightning supplies the compiled C++ state-vector simulator used by
  default. One device is reused across an MLAE schedule; `default.qubit`
  remains available for portability and debugging.
- QFin stays pure Python because profiling shows circuit simulation dominates
  the current classical compiler work. A QFin-specific Rust or C++ extension
  would add packaging complexity without addressing the current bottleneck.

See [docs/circuit-design.md](docs/circuit-design.md) for the v0.3 circuit,
[docs/architecture.md](docs/architecture.md) for package design, and
[docs/alm.md](docs/alm.md) for the v0.4 ALM model, and
[docs/roadmap.md](docs/roadmap.md) for the next research steps.

## Reproducible numerical demonstration

[docs/benchmark-0.3.0.md](docs/benchmark-0.3.0.md) compares several call and
put cases with Black-Scholes and reports QFin estimates, absolute errors,
shots, retained Walsh terms, and logical-resource counts. The table is emitted
by [examples/recruiter_benchmark.py](examples/recruiter_benchmark.py); it is a
simulator demonstration, not evidence of quantum advantage.

[docs/benchmark-0.4.0.md](docs/benchmark-0.4.0.md) records the reproducible ALM
scenario-throughput and Walsh-compiler microbenchmarks, including environment
and interpretation caveats.

## Development

```bash
python -m pip install -e ".[dev]"
ruff check .
mypy src/qfin
pytest --cov=qfin --cov-fail-under=78
python -m build
```
