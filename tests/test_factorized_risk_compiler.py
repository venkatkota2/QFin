from __future__ import annotations

import numpy as np
import pytest

import qfin

pytest.importorskip("pennylane")


def _marginal(
    grid: tuple[float, ...],
    probabilities: tuple[float, ...],
) -> qfin.DistributionEncoding:
    return qfin.DistributionEncoding(
        grid=np.asarray(grid, dtype=np.float64),
        probabilities=np.asarray(probabilities, dtype=np.float64),
        qubits=int(np.log2(len(grid))),
        lower_bound=min(grid),
        upper_bound=max(grid),
        tail_probability=0.0,
        discretization_error=0.0,
        mean_error=0.0,
        objective="test",
    )


def _two_factor_model() -> qfin.FactorizedLossModel:
    encoding = qfin.FactorizedDistributionEncoding(
        factors=(
            _marginal((-1.0, 1.0), (0.4, 0.6)),
            _marginal((-2.0, 2.0), (0.7, 0.3)),
        ),
        factor_names=("rates", "equity"),
    )
    return qfin.FactorizedLossModel(
        encoding,
        qfin.SparseExposureObjective(
            constant=0.25,
            linear={"rates": 1.5, "equity": -0.5},
            quadratic={("rates", "equity"): 0.25},
        ),
    )


def _binary_loss_model() -> qfin.FactorizedLossModel:
    encoding = qfin.FactorizedDistributionEncoding(
        factors=(_marginal((0.0, 1.0), (0.9, 0.1)),),
        factor_names=("loss",),
    )
    return qfin.FactorizedLossModel(
        encoding,
        qfin.SparseExposureObjective(linear={"loss": 1.0}),
    )


@pytest.mark.parametrize("problem_type", [qfin.FactorVaR, qfin.FactorCVaR])
def test_streamed_factor_risk_matches_materialized_test_oracle(
    problem_type: type[qfin.FactorVaR] | type[qfin.FactorCVaR],
) -> None:
    model = _two_factor_model()
    problem = problem_type(model, confidence=0.7)
    summary = qfin.evaluate_factor_risk(problem, chunk_size=2, max_points=4)
    _, losses, probabilities = model.chunk(0, model.joint_grid_points)
    expected = qfin.aggregate_risk(
        qfin.LossDistribution(losses, probabilities),
        confidence=problem.confidence,
        engine="numpy",
    )

    assert summary.var == pytest.approx(expected.var)
    assert summary.cvar == pytest.approx(expected.cvar)
    assert summary.mean == pytest.approx(expected.mean)
    assert summary.standard_deviation == pytest.approx(expected.standard_deviation)
    assert summary.evaluated_points == 4
    assert summary.streamed_point_visits >= summary.evaluated_points
    assert not summary.joint_table_materialized


def test_streamed_factor_risk_never_calls_joint_materialize(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _two_factor_model()

    def fail_materialize(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("joint materialization was called")

    monkeypatch.setattr(qfin.FactorizedDistributionEncoding, "materialize", fail_materialize)
    problem = qfin.FactorCVaR(model, confidence=0.7)
    summary = qfin.evaluate_factor_risk(problem, chunk_size=1, max_points=4)
    compiled = qfin.compile(
        problem,
        backend="classical",
        target_error=0.1,
        arithmetic_scale=4.0,
        max_factor_validation_points=4,
        factor_validation_chunk_size=1,
    )

    assert compiled.run().cvar == pytest.approx(summary.cvar)
    assert compiled.validation.to_dict()["joint_table_materialized"] is False
    assert compiled.validation.to_dict()["loss_code_bins"] <= 2**16


def test_factorized_risk_compiler_tracks_financial_unit_error_and_resources() -> None:
    problem = qfin.FactorCVaR(_binary_loss_model(), confidence=0.75)
    compiled = qfin.compile(
        problem,
        backend="pennylane",
        target_error=0.1,
        arithmetic_scale=1.0,
        max_factor_validation_points=2,
        max_factorized_wires=8,
    )
    resources = compiled.resources(schedule=(0, 2), shots=100)
    device_resources = compiled.device_resources(
        schedule=(0,),
        shots=10,
        max_total_wires=8,
    )

    assert isinstance(compiled, qfin.CompiledFactorRiskModel)
    assert compiled.oracle.loss_qubits == 1
    assert compiled.oracle_error == pytest.approx(0.0)
    assert compiled.error_budget.loss_quantization + compiled.error_budget.estimation == (
        pytest.approx(compiled.error_budget.total)
    )
    assert resources.excess_bit_objectives == 1
    assert resources.objective_evaluations == resources.threshold_objectives + 1
    assert resources.total_circuit_executions == 2 * resources.objective_evaluations
    assert resources.oracle_queries == resources.objective_evaluations * 100 * (1 + 5)
    assert resources.maximum_runtime_qubits == 5
    assert (
        device_resources.objective_evaluations
        == compiled.resources(schedule=(0,), shots=10).objective_evaluations
    )
    assert device_resources.total_routed_gates_per_objective > 0
    assert "joint probability/payoff table: not built" in compiled.explain()


def test_factorized_risk_precision_selection_uses_financial_error() -> None:
    base = _binary_loss_model()
    model = qfin.FactorizedLossModel(
        base.encoding,
        qfin.SparseExposureObjective(constant=0.26),
    )
    compiled = qfin.compile(
        qfin.FactorCVaR(model, confidence=0.75),
        backend="classical",
        target_error=0.1,
        max_factor_validation_points=2,
    )

    assert compiled.oracle.loss_scale == pytest.approx(4.0)
    assert compiled.validation.expected_shortfall_error == pytest.approx(0.01)
    assert compiled.oracle_converged


def test_reusable_threshold_and_excess_register_have_exact_circuit_parity() -> None:
    compiled = qfin.compile(
        qfin.FactorCVaR(_binary_loss_model(), confidence=0.75),
        backend="pennylane",
        target_error=0.1,
        arithmetic_scale=1.0,
        max_factor_validation_points=2,
        max_factorized_wires=8,
    )
    tail = compiled.tail_runtime(1, max_total_wires=8)
    reused = tail.for_threshold_code(2, encoded_probability=0.0)
    excess = compiled.excess_runtime(0, 0, max_total_wires=8)

    assert tail.probability() == pytest.approx(0.1, abs=1e-10)
    assert reused.probability() == pytest.approx(0.0, abs=1e-10)
    assert excess.probability() == pytest.approx(0.1, abs=1e-10)
    np.testing.assert_allclose(excess.excess_probabilities(), np.array([0.9, 0.1]), atol=1e-10)


@pytest.mark.parametrize(
    ("problem", "expected"),
    [
        (qfin.FactorVaR(_binary_loss_model(), confidence=0.75), 0.0),
        (qfin.FactorCVaR(_binary_loss_model(), confidence=0.75), 0.4),
    ],
)
def test_structured_quantum_var_cvar_matches_encoded_reference(
    problem: qfin.FactorVaR | qfin.FactorCVaR,
    expected: float,
) -> None:
    compiled = qfin.compile(
        problem,
        backend="pennylane",
        target_error=0.08,
        arithmetic_scale=1.0,
        max_factor_validation_points=2,
        max_factorized_wires=8,
    )
    result = compiled.run_quantum(
        schedule=(0, 1, 2),
        shots=4_000,
        seed=17,
        likelihood_grid_size=65_537,
        max_total_wires=8,
    )

    assert result.value == pytest.approx(expected, abs=0.04)
    assert result.classical_value == pytest.approx(expected)
    assert result.encoded_value == pytest.approx(expected)
    assert result.search.selected_code == 0
    assert result.resources.threshold_objectives == len(result.search.evaluations)
    if isinstance(problem, qfin.FactorCVaR):
        assert len(result.excess_estimates) == compiled.oracle.loss_qubits
        assert result.expected_shortfall == pytest.approx(expected, abs=0.04)
    else:
        assert not result.excess_estimates
        assert result.expected_shortfall is None


def test_factorized_risk_backend_policy_and_capability_metadata() -> None:
    problem = qfin.FactorCVaR(_binary_loss_model(), confidence=0.75)
    fallback = qfin.compile(
        problem,
        backend="auto",
        target_error=0.1,
        arithmetic_scale=1.0,
        max_factor_validation_points=2,
        max_factorized_wires=4,
    )
    capability = qfin.problem_capabilities(problem)

    assert fallback.backend_name == "classical"
    assert capability.category == "structured_tail_risk"
    assert capability.quantum_algorithm_available
    assert not capability.native_implementation_available
    assert qfin.system_info()["structured_factor_var_cvar"] is True
    with pytest.raises(qfin.ResourceLimitError, match=r"requires 5 wires"):
        qfin.compile(
            problem,
            backend="pennylane",
            target_error=0.1,
            arithmetic_scale=1.0,
            max_factor_validation_points=2,
            max_factorized_wires=4,
        )


@pytest.mark.parametrize("confidence", [0.0, 1.0, float("nan")])
def test_factorized_risk_rejects_invalid_confidence(confidence: float) -> None:
    with pytest.raises(ValueError, match="confidence"):
        qfin.FactorVaR(_binary_loss_model(), confidence=confidence)


def test_factorized_risk_validation_guards_work_and_dimensions() -> None:
    problem = qfin.FactorVaR(_two_factor_model(), confidence=0.7)
    with pytest.raises(ValueError, match="above max_points"):
        qfin.evaluate_factor_risk(problem, max_points=3)
    with pytest.raises(ValueError, match="positive"):
        qfin.evaluate_factor_risk(problem, chunk_size=0)
    compiled = qfin.compile(
        problem,
        backend="classical",
        target_error=0.1,
        arithmetic_scale=4.0,
        max_factor_validation_points=4,
    )
    with pytest.raises(ValueError, match="not 'pennylane'"):
        compiled.tail_runtime(0)
