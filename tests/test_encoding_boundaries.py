"""Encoding factories reject invalid probabilities, objectives and register plans."""

from collections.abc import Callable
from dataclasses import replace
from itertools import product

import numpy as np
import pytest

import qfin
from qfin.representation import (
    FactorizedDistributionEncoding,
    IntegerPolynomialPlan,
    IntegerQuadraticTerm,
    LinearFactorTransform,
    MaterializedFactorGrid,
    QuantumObjectiveEncoding,
    cdf_objective,
    tail_excess_objective,
    tail_probability_objective,
)


def _encoding() -> qfin.DistributionEncoding:
    return qfin.encode(qfin.EmpiricalDistribution([0., 1., 2., 3.]),
                       qubits=2, min_qubits=2, max_qubits=2)


@pytest.mark.parametrize(
    "changes,message",
    [
        ({"grid": [0, 1]}, "2\\*\\*qubits"),
        ({"grid": [0, 1, 2, np.inf]}, "grid values"),
        ({"probabilities": [.2, .2, .2, .2]}, "sum to one"),
        ({"probabilities": [-.1, .1, .5, .5]}, "non-negative"),
        ({"probabilities": [np.nan, 0, .5, .5]}, "sum to one"),
        ({"lower_bound": 4}, "bounds"),
        ({"upper_bound": np.inf}, "bounds"),
        ({"tail_probability": 1}, "tail_probability"),
        ({"discretization_error": np.nan}, "discretization_error"),
        ({"discretization_error": -1}, "discretization_error"),
        ({"mean_error": -1}, "mean_error"),
        ({"objective": ""}, "labels"),
    ],
)
def test_distribution_constructor_validates_its_own_metadata(
    changes: dict[str, object], message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        replace(_encoding(), **changes)


@pytest.mark.parametrize("encoder", [qfin.encode, qfin.encode_quantiles])
@pytest.mark.parametrize("fixed", [False, True])
@pytest.mark.parametrize("objective", ["unknown", lambda grid: np.nan * grid,
                                      lambda _grid: np.ones(1)])
def test_objective_validation_does_not_depend_on_refinement(
    encoder: Callable[..., qfin.DistributionEncoding], fixed: bool, objective: object,
) -> None:
    with pytest.raises(ValueError, match="objective"):
        encoder(qfin.EmpiricalDistribution([0., 1.]), objective=objective,
                qubits=2 if fixed else None, min_qubits=2, max_qubits=3)


@pytest.mark.parametrize(
    "changes,message",
    [
        ({"normalized_values": [0, 1]}, "match"),
        ({"normalized_values": [0, .5, 1, np.nan]}, "finite"),
        ({"normalized_values": [0, .5, 1, 1.01]}, "\\[0, 1\\]"),
        ({"financial_scale": -1}, "financial_scale"),
        ({"financial_offset": np.inf}, "financial_offset"),
        ({"label": ""}, "label"),
        ({"threshold": np.nan}, "threshold"),
    ],
)
def test_objective_constructor_preserves_financial_amplitude_domain(
    changes: dict[str, object], message: str,
) -> None:
    objective = QuantumObjectiveEncoding(_encoding(), np.arange(4)/3, 3, -1, "test")
    with pytest.raises(ValueError, match=message):
        replace(objective, **changes)


def test_exact_objective_maps_and_empty_tails() -> None:
    encoding = _encoding()
    for factory in (cdf_objective, tail_probability_objective, tail_excess_objective):
        with pytest.raises(ValueError, match="threshold"):
            factory(encoding, np.inf)
    cdf = cdf_objective(encoding, 1)
    tail = tail_probability_objective(encoding, 1)
    assert cdf.exact_amplitude + tail.exact_amplitude == pytest.approx(1)
    assert cdf.to_dict()["inclusive"]
    for amplitude in (-.01, 1.01, np.nan):
        with pytest.raises(ValueError, match="amplitude"):
            cdf.value_from_amplitude(amplitude)
    empty_tail = tail_excess_objective(encoding, 100)
    assert empty_tail.exact_value == 0
    assert empty_tail.financial_scale == 0
    np.testing.assert_array_equal(empty_tail.normalized_values, np.zeros(4))


@pytest.mark.parametrize(
    "factory,message",
    [
        (lambda: LinearFactorTransform(np.ones(2), np.zeros(2), ("a", "b")), "array"),
        (lambda: LinearFactorTransform([[np.inf]], [0], ("a",)), "finite"),
        (lambda: LinearFactorTransform([[1]], [0, 1], ("a",)), "offset"),
        (lambda: LinearFactorTransform(np.eye(2), [0, 0], ("a",)), "output_names"),
        (lambda: LinearFactorTransform(np.eye(2), [0, 0], ("a", "a")), "unique"),
        (lambda: MaterializedFactorGrid([[0], [1]], [.5], ("a",)), "same points"),
        (lambda: MaterializedFactorGrid([[0], [1]], [.5, .5], ()), "value_names"),
        (lambda: MaterializedFactorGrid([[0], [np.inf]], [.5, .5], ("a",)), "finite"),
        (lambda: MaterializedFactorGrid([[0], [1]], [.5, -.5], ("a",)), "probabilities"),
        (lambda: FactorizedDistributionEncoding((), ()), "at least one"),
        (lambda: FactorizedDistributionEncoding((_encoding(),), ()), "factor_names"),
        (lambda: FactorizedDistributionEncoding((_encoding(),)*2, ("a", "a")), "unique"),
        (lambda: FactorizedDistributionEncoding((_encoding(),), ("a",), " "), "dependence"),
        (lambda: FactorizedDistributionEncoding(
            (_encoding(),), ("a",), transform=LinearFactorTransform([[1, 1]], [0], ("a",))
        ), "one column"),
        (lambda: qfin.encode_independent_factors([]), "must not be empty"),
        (lambda: qfin.encode_independent_factors([qfin.EmpiricalDistribution([0, 1])],
                                               method="unsupported"), "method"),
    ],
)
def test_factorized_factories_defend_dimensions_and_probabilities(
    factory: Callable[[], object], message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        factory()


def test_integer_polynomial_bit_expansion_matches_exhaustive_integer_oracle() -> None:
    plan = IntegerPolynomialPlan((2, 2), 6, 25, (-2, 1),
                                 (IntegerQuadraticTerm(0, 0, 3), IntegerQuadraticTerm(0, 1, -2)))
    for x, y in product(range(4), repeat=2):
        bits = (x >> 1, x & 1, y >> 1, y & 1)
        actual = sum(coefficient * np.prod([bit for bit, used in zip(bits, mask, strict=True)
                                           if used], dtype=int)
                     for mask, coefficient in plan.bit_coefficients)
        assert actual == 25 - 2*x + y + 3*x*x - 2*x*y
        assert 0 <= actual < 2**plan.output_qubits
    with pytest.raises(ValueError, match="range"):
        IntegerPolynomialPlan((2,), 2, 0, (2,))
    with pytest.raises(ValueError, match="linear"):
        IntegerPolynomialPlan((2,), 2, 0, (1, 1))
    with pytest.raises(ValueError, match="unavailable"):
        IntegerPolynomialPlan((2,), 4, 0, (1,), (IntegerQuadraticTerm(0, 1, 1),))
    with pytest.raises(ValueError, match="non-zero"):
        IntegerQuadraticTerm(0, 0, 0)
