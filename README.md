# QFin

[![CI](https://github.com/venkatkota2/QFin/actions/workflows/ci.yml/badge.svg)](https://github.com/venkatkota2/QFin/actions/workflows/ci.yml)

QFin is an experimental finance-specific Python library with a private C++20
finance core and a compiler for existing PennyLane quantum workflows. Financial
users work with curves, bonds, portfolios, policies and loss distributions.
Results report their methodology, execution engine, accuracy metadata and limits.

The 1.1.2 hardening milestone improves numerical correctness, settlement
consistency, reproducibility, API ownership, testing and packaging. It adds no
financial products, stochastic models, quantum algorithms or backends.

## Install

From a repository checkout:

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
python -m pip install -e .
```

Python 3.11+ is required. Core calculations require NumPy and SciPy. A source or
editable install builds the native extension and needs a C++20 compiler; an
installed wheel contains the extension. Optional extras remain separate:

```bash
python -m pip install -e ".[quantum]"     # PennyLane and Lightning simulation
python -m pip install -e ".[qiskit]"      # existing circuit export
python -m pip install -e ".[validation]"  # independent QuantLib comparisons
python -m pip install -e ".[dev]"         # development checks
```

The distribution is `qfin-quantum`; the import is `qfin`. Read the installed
version from `qfin.__version__` or `qfin --version`.

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

## Existing quantum workflows

```python
market = qfin.BlackScholes(spot=100, rate=0.04, volatility=0.20)
option = qfin.EuropeanCall(strike=105, maturity=1.0)
compiled = qfin.compile(option, market, target_error=0.10, max_qubits=8)
print(compiled.explain())
result = compiled.run(shots=2_000, schedule=(0, 1, 2, 4), seed=7)
print(result.value, result.confidence_interval_95)
```

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
