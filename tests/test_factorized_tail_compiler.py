from math import asin, sin, sqrt

import numpy as np
import pytest

import qfin

pytest.importorskip("pennylane")


def _binary_tail_problem(
    *,
    factors: int = 2,
    objective: qfin.SparseExposureObjective | None = None,
    threshold: float = 0.0,
) -> qfin.FactorTailProbability:
    marginals = tuple(
        qfin.DistributionEncoding(
            grid=np.array([-1.0, 1.0]),
            probabilities=np.array([0.5, 0.5]),
            qubits=1,
            lower_bound=-1.0,
            upper_bound=1.0,
            tail_probability=0.0,
            discretization_error=0.0,
            mean_error=0.0,
            objective="test",
        )
        for _ in range(factors)
    )
    encoding = qfin.FactorizedDistributionEncoding(
        marginals,
        tuple(f"factor_{index}" for index in range(factors)),
    )
    selected_objective = objective or qfin.SparseExposureObjective(
        linear={name: 1.0 for name in encoding.factor_names}
    )
    return qfin.FactorTailProbability(
        qfin.FactorizedLossModel(encoding, selected_objective),
        threshold=threshold,
    )


def test_factorized_tail_compiler_runs_classical_reference() -> None:
    problem = _binary_tail_problem()
    compiled = qfin.compile(
        problem,
        backend="classical",
        target_error=0.05,
        arithmetic_scale=1.0,
    )

    assert isinstance(compiled, qfin.CompiledFactorTailModel)
    assert compiled.backend_name == "classical"
    assert compiled.run().probability == pytest.approx(0.75)
    assert compiled.validation.disagreement_probability == pytest.approx(0.0)
    assert "joint payoff table: not built" in compiled.explain()
    with pytest.raises(ValueError, match="not 'pennylane'"):
        compiled.to_pennylane()


def test_factorized_quantum_tail_matches_grover_probability() -> None:
    problem = _binary_tail_problem()
    compiled = qfin.compile(
        problem,
        backend="pennylane",
        target_error=0.05,
        arithmetic_scale=1.0,
        max_factorized_wires=12,
    )
    runtime = compiled.to_pennylane(max_total_wires=12)
    amplitude = compiled.validation.oracle_probability
    theta = asin(sqrt(amplitude))

    assert runtime.device_name == "lightning.qubit"
    assert runtime.probability(0) == pytest.approx(amplitude, abs=1e-10)
    assert runtime.probability(1) == pytest.approx(sin(3 * theta) ** 2, abs=1e-10)


def test_precision_selection_and_error_budget_are_explicit() -> None:
    problem = _binary_tail_problem(
        factors=1,
        objective=qfin.SparseExposureObjective(constant=0.49),
        threshold=0.4,
    )
    compiled = qfin.compile(problem, backend="classical", target_error=0.1)
    budget = compiled.error_budget

    assert compiled.oracle.loss_scale == pytest.approx(2.0)
    assert compiled.validation.disagreement_probability == pytest.approx(0.0)
    assert budget.transform + budget.payoff + budget.estimation == pytest.approx(
        budget.total
    )
    assert budget.oracle == pytest.approx(budget.transform + budget.payoff)
    assert set(budget.to_dict()) >= {"transform", "payoff", "oracle", "estimation"}


def test_backend_policy_falls_back_and_explicit_quantum_rejects_width() -> None:
    problem = _binary_tail_problem()
    compiled = qfin.compile(
        problem,
        backend="auto",
        arithmetic_scale=1.0,
        max_factorized_wires=4,
    )
    assert compiled.backend_name == "classical"

    with pytest.raises(qfin.ResourceLimitError, match=r"requires .* wires"):
        qfin.compile(
            problem,
            backend="pennylane",
            arithmetic_scale=1.0,
            max_factorized_wires=4,
        )


def test_target_comparison_is_measured_and_joint_materialization_is_guarded() -> None:
    compiled = qfin.compile(
        _binary_tail_problem(),
        backend="pennylane",
        arithmetic_scale=1.0,
        max_factorized_wires=12,
    )
    with pytest.raises(qfin.ResourceLimitError, match="would materialize 4 points"):
        compiled.target_comparison(max_joint_points=3, max_total_wires=12)

    comparison = compiled.target_comparison(
        schedule=(0,),
        shots=100,
        target="all_to_all",
        max_joint_points=4,
        max_total_wires=12,
    )
    assert comparison.joint_points == 4
    assert comparison.structured.total_routed_gates_per_objective > 0
    assert comparison.generic.total_routed_gates_per_objective > 0
    assert comparison.generic_joint_materialized_for_benchmark
    assert comparison.routed_gate_ratio > 0


def test_capability_metadata_distinguishes_qfin_arithmetic_from_lightning() -> None:
    problem = _binary_tail_problem()
    capability = qfin.problem_capabilities(problem)
    info = qfin.system_info()

    assert capability.category == "structured_tail_risk"
    assert capability.quantum_representation_available
    assert capability.quantum_algorithm_available
    assert not capability.native_implementation_available
    assert "Lightning" in capability.note
    assert info["structured_arithmetic_oracles"] is True
    assert info["factorized_tail_risk"] is True
    assert info["preferred_quantum_device"] == "lightning.qubit"
