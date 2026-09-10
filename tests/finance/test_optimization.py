import numpy as np
import pytest

import qfin


@pytest.fixture
def problem() -> qfin.MeanVarianceProblem:
    return qfin.MeanVarianceProblem(
        expected_returns=np.array([0.05, 0.08, 0.12]),
        covariance=np.array(
            [
                [0.03, 0.004, 0.002],
                [0.004, 0.07, 0.008],
                [0.002, 0.008, 0.15],
            ]
        ),
        risk_aversion=3.0,
        asset_names=("bonds", "balanced", "equity"),
    )


def test_slsqp_solution_is_feasible_and_improves_the_baseline(
    problem: qfin.MeanVarianceProblem,
) -> None:
    result = problem.solve()
    assert result.success
    assert np.sum(result.weights) == pytest.approx(1.0, abs=1e-9)
    assert np.all(result.weights >= -1e-10)
    assert result.utility >= result.baseline_utility - 1e-12
    assert result.utility_improvement >= -1e-12
    assert result.expected_return == pytest.approx(
        problem.expected_portfolio_return(result.weights)
    )
    assert result.variance == pytest.approx(problem.portfolio_variance(result.weights))
    assert result.solver == "scipy_slsqp_continuous_mean_variance"
    metadata = qfin.compile(problem).to_dict()
    assert metadata["problem_category"] == "portfolio_optimization"
    assert metadata["compilation_converged"]
    assert metadata["target_error"] is None
    assert not metadata["quantum_execution_available"]


def test_target_return_and_custom_bounds_are_enforced() -> None:
    problem = qfin.MeanVarianceProblem(
        np.array([0.04, 0.08, 0.12]),
        np.diag([0.02, 0.06, 0.12]),
        risk_aversion=2.0,
        target_return=0.085,
        upper_bounds=[0.7, 0.7, 0.7],
    )
    result = problem.solve()
    assert result.expected_return >= 0.085 - 1e-8
    assert result.target_return_residual is not None
    assert result.target_return_residual >= -1e-8
    assert np.max(result.weights) <= 0.7 + 1e-8


def test_unbounded_closed_form_satisfies_first_order_conditions() -> None:
    problem = qfin.MeanVarianceProblem(
        np.array([0.05, 0.09, 0.11]),
        np.array([[0.04, 0.01, 0.0], [0.01, 0.08, 0.01], [0.0, 0.01, 0.12]]),
        risk_aversion=4.0,
        long_only=False,
    )
    result = problem.solve(method="closed_form")
    gradient = problem.risk_aversion * (problem.covariance @ result.weights)
    gradient -= problem.expected_returns
    assert np.sum(result.weights) == pytest.approx(1.0, abs=1e-10)
    np.testing.assert_allclose(gradient, np.full(3, gradient[0]), atol=1e-10)
    assert "closed_form" in result.solver


def test_rank_deficient_closed_form_with_duplicate_assets_is_well_posed() -> None:
    problem = qfin.MeanVarianceProblem(
        np.array([0.07, 0.07]),
        np.array([[0.04, 0.04], [0.04, 0.04]]),
        risk_aversion=3.0,
        long_only=False,
    )

    result = problem.solve(method="closed_form")

    np.testing.assert_allclose(result.weights, [0.5, 0.5], atol=1.0e-14)
    assert result.variance == pytest.approx(0.04)
    assert result.expected_return == pytest.approx(0.07)
    assert "singular_kkt" in result.solver


def test_zero_variance_asset_can_be_part_of_finite_closed_form_optimum() -> None:
    problem = qfin.MeanVarianceProblem(
        np.array([0.10, 0.05]),
        np.diag([1.0, 0.0]),
        risk_aversion=1.0,
        long_only=False,
    )

    result = problem.solve(method="closed_form")

    np.testing.assert_allclose(result.weights, [0.05, 0.95], atol=1.0e-13)
    assert np.sum(result.weights) == pytest.approx(1.0, abs=1.0e-14)


def test_budget_neutral_zero_variance_return_direction_is_unbounded() -> None:
    problem = qfin.MeanVarianceProblem(
        np.array([0.04, 0.09]),
        np.array([[0.03, 0.03], [0.03, 0.03]]),
        risk_aversion=2.0,
        long_only=False,
    )

    with pytest.raises(
        qfin.OptimizationError,
        match="unbounded mean-variance problem due to zero-variance budget-neutral",
    ):
        problem.solve(method="closed_form")


def test_expected_return_neutral_covariance_null_space_is_valid() -> None:
    covariance = np.array(
        [
            [0.01, 0.02, 0.03],
            [0.02, 0.04, 0.06],
            [0.03, 0.06, 0.09],
        ]
    )
    expected_returns = np.array([0.03, 0.06, 0.09])
    problem = qfin.MeanVarianceProblem(
        expected_returns,
        covariance,
        risk_aversion=4.0,
        long_only=False,
    )

    result = problem.solve(method="closed_form")

    assert np.sum(result.weights) == pytest.approx(1.0, abs=1.0e-13)
    gradient = problem.risk_aversion * covariance @ result.weights - expected_returns
    np.testing.assert_allclose(gradient, np.full(3, gradient[0]), atol=1.0e-12)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        (
            {
                "expected_returns": [0.05, 0.08],
                "covariance": [[0.04, 0.2], [0.2, 0.04]],
            },
            "positive semidefinite",
        ),
        (
            {
                "expected_returns": [0.05, 0.08],
                "covariance": np.eye(2),
                "lower_bounds": [0.6, 0.6],
            },
            "exceed",
        ),
    ],
)
def test_problem_validation_rejects_invalid_inputs(
    kwargs: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        qfin.MeanVarianceProblem(**kwargs)  # type: ignore[arg-type]


def test_compiler_selects_classical_and_reports_quantum_boundaries(
    problem: qfin.MeanVarianceProblem,
) -> None:
    compiled = qfin.compile(problem)
    assert isinstance(compiled, qfin.CompiledOptimizationModel)
    assert compiled.backend_name == "classical"
    assert not compiled.quantum_algorithm_available
    assert compiled.run().success
    assert compiled.resources().assets == 3
    assert not compiled.resources().quantum_representation_available
    assert compiled.block_encoding_feasibility().mathematical_qsvt_candidate
    assert "feasibility metadata only" in compiled.explain()

    capabilities = qfin.problem_capabilities(problem)
    assert capabilities.category == "portfolio_optimization"
    assert not capabilities.quantum_algorithm_available
    with pytest.raises(qfin.CompilationError, match="no implemented PennyLane"):
        qfin.compile(problem, backend="pennylane")
    with pytest.raises(qfin.CompilationError, match="no implemented quantum"):
        compiled.to_pennylane()
