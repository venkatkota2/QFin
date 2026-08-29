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
pytest --cov=qfin --cov-report=term-missing --cov-fail-under=78
python -m build
python examples/native_benchmark.py
```

Every native kernel requires a Python/NumPy oracle, analytical cases where
available, malformed-input tests, explicit tolerances, and measured evidence
that batching justifies crossing the extension boundary.
