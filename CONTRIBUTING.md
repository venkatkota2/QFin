# Contributing

QFin is intentionally narrow while the core abstraction is being validated.
Before adding a financial product or algorithm, open an issue describing:

- the financial inputs and outputs;
- the mathematical and classical reference model;
- the proposed quantum representation and algorithm;
- error sources and resource assumptions; and
- tests that distinguish a real implementation from a placeholder.

Set up the development environment with `python -m pip install -e ".[dev]"`.
Editable installs compile the C++20 extension through scikit-build-core; users
do not run CMake manually. Set `QFIN_REQUIRE_NATIVE=1` when developing or testing
the native backend so compiler errors fail the build. Without that requirement,
source installs may fall back to the existing NumPy engines. See
[installation](docs/installation.md) for compiler-free setup. Then run the
complete validation set:

```bash
ruff check .
mypy src/qfin
pytest --cov=qfin --cov-branch --cov-report=term-missing --cov-report=json:coverage.json --cov-fail-under=90
python examples/check_coverage.py coverage.json
python -m build
python examples/native_benchmark.py
python examples/quantum_risk_benchmark.py --repeats 1 --shots 500
```

Every native kernel requires a Python/NumPy oracle, analytical cases where
available, malformed-input tests, explicit tolerances, and measured evidence
that batching justifies crossing the extension boundary.

Every quantum-risk change additionally requires a finite-distribution
reference, exact state/objective-amplitude tests, deterministic seeded
simulator tests, logical resource accounting, and documentation that separates
simulator feasibility from hardware or advantage claims.

The current hardening milestone is feature frozen. Prioritize numerical and
financial invariants, independent oracles, deterministic native parity, strict
types and measured public-API performance. Do not relax tolerances or checks to
accept an optimization. See [current validation](docs/validation.md).

The global line gate is 93%, branch gate 81%, with additional numerical module
floors. Hypothesis uses deterministic `ci` (50 examples/property) and `stress`
(500) profiles: `HYPOTHESIS_PROFILE=stress pytest tests/property`. Native generated
buffer tests add fixed 100-example campaigns per property.

Independent fixture generators import QuantLib/Decimal, never QFin for expected
values. Review fixture provenance and financial materiality when regenerating;
ordinary tests consume checked-in values without QuantLib. API snapshots in
`tests/api/snapshots` are reviewed compatibility baselines, not routine golden
outputs to overwrite after a failure. Run `tools/mutation_check.py --output
mutations.json` for the bounded numerical campaign and see [reproducibility](docs/reproducibility.md),
[statistical validation](docs/statistical-validation.md), [memory](docs/memory.md)
and [release gates](docs/release-engineering.md).
