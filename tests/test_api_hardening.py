"""Cross-module API ownership, validation, version, and deprecation invariants."""

from __future__ import annotations

import warnings
from collections.abc import Callable
from importlib.metadata import version
from typing import Any

import numpy as np
import pytest

import qfin
from qfin.algorithms import CircuitObservation, maximum_likelihood_amplitude_estimate
from qfin.representation import (
    LinearFactorTransform,
    MaterializedFactorGrid,
    QuantumObjectiveEncoding,
)
from qfin.resources import estimate_resources


def _assert_constructor_owns_array(
    source: np.ndarray[Any, Any],
    constructor: Callable[[], np.ndarray[Any, Any]],
) -> None:
    original = source.copy()
    original_shape = source.shape
    original_writeable = source.flags.writeable
    stored = constructor()
    np.testing.assert_array_equal(source, original)
    assert source.shape == original_shape
    assert source.flags.writeable is original_writeable
    assert not np.shares_memory(stored, source)
    assert not stored.flags.writeable


def test_immutable_models_copy_caller_owned_arrays() -> None:
    losses = np.array([-1.0, 2.0, 5.0])
    _assert_constructor_owns_array(
        losses,
        lambda: qfin.LossDistribution(losses).losses,
    )

    scenario_values = np.array([[0.1, -0.2], [0.3, 0.4]])
    _assert_constructor_owns_array(
        scenario_values,
        lambda: qfin.FactorScenarios(
            scenario_values,
            ("rates", "equity"),
            "test dependence",
        ).values,
    )

    correlation = np.array([[1.0, 0.25], [0.25, 1.0]])
    _assert_constructor_owns_array(
        correlation,
        lambda: qfin.GaussianFactorModel(("a", "b"), correlation).correlation,
    )

    matrix = np.array([[1.0, 2.0], [-1.0, 0.5]])
    _assert_constructor_owns_array(
        matrix,
        lambda: LinearFactorTransform(matrix, np.zeros(2), ("x", "y")).matrix,
    )

    grid_values = np.array([[1.0], [2.0]])
    _assert_constructor_owns_array(
        grid_values,
        lambda: MaterializedFactorGrid(
            grid_values,
            np.array([0.25, 0.75]),
            ("loss",),
        ).values,
    )

    normalized = np.array([0.0, 1.0])
    distribution = qfin.DistributionEncoding(
        grid=np.array([0.0, 1.0]),
        probabilities=np.array([0.5, 0.5]),
        qubits=1,
        lower_bound=0.0,
        upper_bound=1.0,
        tail_probability=0.0,
        discretization_error=None,
        mean_error=0.0,
        objective="expectation",
    )
    _assert_constructor_owns_array(
        normalized,
        lambda: QuantumObjectiveEncoding(
            distribution,
            normalized,
            1.0,
            0.0,
            "test",
        ).normalized_values,
    )


_NON_INTEGER_VALUES: tuple[object, ...] = (
    True,
    False,
    1.5,
    2.1,
    np.float32(2.0),
)


@pytest.mark.parametrize("value", _NON_INTEGER_VALUES)
def test_public_integer_boundaries_reject_coercion(value: object) -> None:
    with pytest.raises(ValueError, match="integer"):
        qfin.FixedRateBond(1.0, 0.02, frequency=value)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="integer"):
        CircuitObservation(power=value, successes=0, shots=10)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="integer"):
        qfin.GaussianFactorModel(("x",), np.ones((1, 1))).simulate(
            value  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="integer"):
        estimate_resources(2, schedule=(0, value), shots=10)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="integer"):
        estimate_resources(2, shots=value)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="integer"):
        maximum_likelihood_amplitude_estimate(
            (CircuitObservation(0, 5, 10),),
            grid_size=value,  # type: ignore[arg-type]
        )


def test_version_comes_from_installed_distribution_metadata() -> None:
    assert qfin.__version__ == version("qfin-quantum")


@pytest.mark.parametrize(
    ("name", "canonical_module"),
    (
        ("IntegerPolynomialPlan", "qfin.representation.arithmetic"),
        ("ProbabilityTreePreparation", "qfin.circuits.state_preparation"),
        ("StatePreparationCost", "qfin.representation.strategies"),
        ("WalshTerm", "qfin.circuits.walsh_payoff"),
    ),
)
def test_low_level_top_level_aliases_warn(name: str, canonical_module: str) -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        value = qfin.__getattr__(name)
    assert value.__module__ == canonical_module
    assert len(caught) == 1
    assert issubclass(caught[0].category, DeprecationWarning)
    assert "deprecated at the top level" in str(caught[0].message)
    assert caught[0].filename == __file__


def test_cli_reports_metadata_version(capsys: pytest.CaptureFixture[str]) -> None:
    from qfin.__main__ import main

    with pytest.raises(SystemExit) as raised:
        main(("--version",))
    assert raised.value.code == 0
    assert capsys.readouterr().out.strip() == f"qfin {qfin.__version__}"
