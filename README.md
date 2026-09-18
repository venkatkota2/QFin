# QFin

[![CI](https://github.com/venkatkota2/QFin/actions/workflows/ci.yml/badge.svg)](https://github.com/venkatkota2/QFin/actions/workflows/ci.yml)

QFin is an experimental Python library for compiling supported financial models
into logical quantum programs and simulating them locally with PennyLane and
PennyLane-Lightning. The intended workflow is **financial model → QFin
financial-to-quantum compilation → logical quantum program → PennyLane/Lightning
simulation**. A physical quantum computer or hardware account is not required.

The existing classical finance utilities use NumPy/SciPy and optional private
C++20 kernels. Results report methodology, execution engine, accuracy and limits.

The 1.1.2.post1 packaging update makes the existing library installable with or
without a C++ toolchain. Financial models, calculations and public APIs are unchanged.

## Install

Install directly from this repository with Python 3.11–3.13:

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
python -m pip install "qfin-quantum[quantum] @ git+https://github.com/venkatkota2/QFin.git"
python -m qfin --version
```

NumPy and SciPy are installed automatically. Source installs build the C++20
extension when possible and otherwise use the existing NumPy engines. To skip
compilation explicitly, add `-Cwheel.cmake=false` to the install command. Native
binary wheels include the extension and need no compiler on the user's machine.

The command above includes PennyLane and Lightning. For just the classical
finance utilities, omit the quantum extra:

```bash
python -m pip install "qfin-quantum @ git+https://github.com/venkatkota2/QFin.git"
```

The distribution is `qfin-quantum`; the import is `qfin`. This repository has not
been published to PyPI: use the URL above, a local checkout, or a tested wheel.
See the [installation guide](docs/installation.md) for Windows, compiler-free
installation, native acceleration, extras and troubleshooting.

For development from a repository checkout:

```bash
python -m pip install -e .               # core library
python -m pip install -e ".[quantum]"     # PennyLane and Lightning simulation
python -m pip install -e ".[qiskit]"      # existing circuit export
python -m pip install -e ".[validation]"  # independent QuantLib comparisons
python -m pip install -e ".[dev]"         # development checks
```

Verify the installation from any directory:

```python
import qfin

curve = qfin.YieldCurve([0, 1, 5], [0.03, 0.03, 0.03])
bond = qfin.FixedRateBond(maturity=5, coupon_rate=0.04)
print(qfin.price_bonds([bond], curve).dirty_prices)
print(qfin.system_info()["native_extension"])
```

## Dated fixed income through ALM risk

```python
import qfin

curve = qfin.YieldCurve(
    [0, 1, 5, 10, 30],
    [0.02, 0.025, 0.03, 0.035, 0.04],
    valuation_date="2026-01-31",
    interpolation="log_linear_discount",
    extrapolation="flat_forward",
)
bond = qfin.FixedRateBond.from_dates(
    "2025-01-31", "2031-01-31", 0.04,
    frequency=2, day_count="30/360", end_of_month=True,
)
prices = qfin.price_bonds(bond, curve)
print(prices.clean_prices, prices.accrued_interest, prices.dirty_prices)
print(qfin.key_rate_risk(bond, curve).key_rate_dv01)

# Omitted dated settlement uses the curve valuation date consistently.
assets = qfin.AssetPortfolio([bond], quantities=[100])
liabilities = qfin.LiabilityPortfolio.from_arrays([3, 5], [4_000, 5_000])
model = qfin.ALMModel(assets, liabilities, curve)
base = model.evaluate()
scenarios = qfin.RateScenarioSet.parallel(curve, [-0.01, 0.0, 0.01])
stressed = model.run_scenarios(scenarios, chunk_size=2)
losses = stressed.loss_distribution()
summary = qfin.compile(qfin.CVaR(losses, confidence=0.95), backend="classical").run()
print(base.surplus, stressed.surplus, summary.cvar)
print(curve.explain())
```

A rate scenario adds continuous zero-rate shocks to curve nodes and then retains
the curve's interpolation and extrapolation. Scenario valuation matches individual
`curve.shifted(shock)` repricing. Native curve kernels support only linear-zero
interpolation with flat-zero extrapolation; automatic dispatch uses NumPy for the
other supported methods. Mixed dated/floating batches and inconsistent settlement
clocks are rejected.

## Financial model to local quantum simulation

```python
import qfin

market = qfin.BlackScholes(spot=100, rate=0.04, volatility=0.20)
option = qfin.EuropeanCall(strike=105, maturity=1.0)
compiled = qfin.compile(option, market, target_error=0.10, max_qubits=8)
print(compiled.explain())
result = compiled.run(
    shots=2_000, schedule=(0, 1, 2, 4), seed=7,
    device_name="lightning.qubit",
)
print(result.value, result.confidence_interval_95)
```

`compiled` contains the logical problem, encoding, algorithm and resource/error
metadata; `run()` executes its circuits in the local Lightning simulator. To
inspect the circuit, use
`compiled.to_pennylane(device_name="lightning.qubit").draw(power=0)`.

This example requires the quantum extra. Finite and structured-factor risk retain
their existing explicit `run_quantum()` paths. MLAE uses deterministic likelihood
optimization and guarded confidence regions. Target-error units follow the
objective: currency for option prices, loss units for VaR/CVaR and probability
units for tail probability. Unknown encoding refinement error is `None`, not zero.

PennyLane-Lightning performs quantum simulation. QFin C++ performs finance loops;
it does not implement a simulator. `system_info()` distinguishes these capabilities.
No quantum advantage, production hardware runtime or fault-tolerant resource
claim is made. Adaptive risk-search intervals retain explicit statistical limits.

## Documentation and examples

Start with the [current documentation index](docs/README.md), including
[financial conventions](docs/financial-conventions.md), [curves](docs/curves.md),
[fixed income](docs/fixed-income.md), [ALM](docs/alm.md), [life](docs/life.md),
[risk](docs/risk.md), [compiler](docs/compiler.md) and [quantum risk](docs/quantum-risk.md).

The [public API contract](docs/public-api.md) keeps financial models, results and
utilities at the top level. Eight implementation aliases are deprecated with
canonical namespace replacements; they remain through 1.x, with removal no earlier
than 2.0. The CLI intentionally remains a small European-option demonstration.

[Performance](docs/performance.md) records measured public-API timings and dispatch
choices. NumPy remains the default where native acceleration lacks a consistent
win. [Validation](docs/validation.md) describes properties, parity, numerical and
financial tolerances, sanitizers and installed-wheel CI. Historical milestones and
benchmarks are [preserved separately](docs/history/README.md).

Runnable examples remain in `examples/`, including `fixed_income_accuracy.py`,
`multiperiod_alm.py`, `life_products.py`, `quantum_risk_pipeline.py`,
`structured_factor_risk.py` and `device_realism.py`. See [Contributing](CONTRIBUTING.md)
for checks and benchmark commands.

## Limitations and licensing

QFin is research grade. It is not calibrated by default, and its annual policy and
ALM models are not replacements for commercial actuarial platforms. The full
[limitations](docs/limitations.md) remain part of the public contract.

The repository currently has no license grant. Reuse requiring an explicit grant
is blocked until the maintainer chooses and authorizes licensing terms. The
hardening work does not invent or add a license.

The [hardening report](docs/hardening-1.1.2.md) records numerical repairs, measured performance, API compatibility, verification and remaining limitations.
