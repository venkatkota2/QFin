# Contributing

QFin is intentionally narrow while the core abstraction is being validated.
Before adding a financial product or algorithm, open an issue describing:

- the financial inputs and outputs;
- the mathematical and classical reference model;
- the proposed quantum representation and algorithm;
- error sources and resource assumptions; and
- tests that distinguish a real implementation from a placeholder.

Set up the development environment with `python -m pip install -e ".[dev]"`,
then run `ruff check .`, `mypy src/qfin`, `pytest`, and `python -m build`.

