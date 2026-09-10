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
do not run CMake manually. Then run the complete validation set:

```bash
ruff check .
mypy src/qfin
pytest --cov=qfin --cov-report=term-missing --cov-fail-under=90
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
