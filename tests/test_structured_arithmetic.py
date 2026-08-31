import numpy as np
import pytest

import qfin

qml = pytest.importorskip("pennylane")


def _binary_factor_encoding(
    *,
    factors: int = 2,
    transform: qfin.LinearFactorTransform | None = None,
) -> qfin.FactorizedDistributionEncoding:
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
    return qfin.FactorizedDistributionEncoding(
        factors=marginals,
        factor_names=tuple(f"z{index}" for index in range(factors)),
        transform=transform,
    )


def test_integer_quadratic_plan_matches_basis_state_arithmetic() -> None:
    plan = qfin.IntegerPolynomialPlan(
        input_qubits=(2, 1),
        output_qubits=5,
        constant=2,
        linear=(3, 1),
        quadratic=(qfin.IntegerQuadraticTerm(0, 1, 2),),
    )
    device = qml.device("default.qubit", wires=8)

    @qml.qnode(device)  # type: ignore[untyped-decorator]
    def circuit(left: int, right: int) -> object:
        bits = np.array([left >> 1, left & 1, right], dtype=int)
        qml.BasisState(bits, wires=(0, 1, 2))
        plan.apply(((0, 1), (2,)), (3, 4, 5, 6, 7))
        return qml.probs(wires=(3, 4, 5, 6, 7))

    indices = np.array(
        [(left, right) for left in range(4) for right in range(2)],
        dtype=np.int64,
    )
    expected = plan.evaluate(indices)
    observed = np.array(
        [np.argmax(circuit(int(left), int(right))) for left, right in indices]
    )
    np.testing.assert_array_equal(observed, expected)
    assert plan.monomial_count < 2 ** sum(plan.input_qubits)


def test_affine_transform_has_streamed_and_circuit_parity() -> None:
    transform = qfin.LinearFactorTransform(
        matrix=np.array([[0.5, -0.25]]),
        offset=np.array([0.1]),
        output_names=("spread",),
    )
    encoding = _binary_factor_encoding(transform=transform)
    plan = qfin.compile_affine_transform(encoding, scale=16.0)
    validation = qfin.validate_affine_transform(
        encoding,
        plan,
        chunk_size=2,
        max_points=4,
    )

    assert validation.evaluated_points == 4
    assert validation.chunks == 2
    assert not validation.joint_table_materialized
    assert validation.maximum_abs_error <= validation.error_bound

    output_wires = tuple(range(2, 2 + plan.output_qubits[0]))
    device = qml.device("default.qubit", wires=2 + len(output_wires))

    @qml.qnode(device)  # type: ignore[untyped-decorator]
    def circuit(left: int, right: int) -> object:
        qml.BasisState(np.array([left, right]), wires=(0, 1))
        plan.apply(((0,), (1,)), (output_wires,))
        return qml.probs(wires=output_wires)

    indices = np.array(
        [(left, right) for left in range(2) for right in range(2)],
        dtype=np.int64,
    )
    expected_codes = plan.evaluate_codes(indices)[:, 0]
    observed_codes = np.array(
        [np.argmax(circuit(int(left), int(right))) for left, right in indices]
    )
    np.testing.assert_array_equal(observed_codes, expected_codes)


def test_quantile_grid_is_rejected_with_actionable_message() -> None:
    encoding = qfin.encode_independent_factors(
        [qfin.Normal()],
        qubits_per_factor=2,
        method="quantile",
    )
    with pytest.raises(ValueError, match="method='probability'"):
        qfin.compile_affine_transform(encoding)


def test_piecewise_loss_oracle_matches_streaming_and_circuit_probabilities() -> None:
    encoding = _binary_factor_encoding()
    objective = qfin.SparseExposureObjective(
        constant=0.25,
        linear={"z0": 0.5, "z1": -0.25},
        quadratic={("z0", "z1"): 0.125},
        piecewise=(qfin.HingeExposure("z0", threshold=0.0, slope=0.5),),
    )
    model = qfin.FactorizedLossModel(encoding, objective)
    problem = qfin.FactorTailProbability(model, threshold=0.0)
    plan = qfin.compile_structured_loss_oracle(
        encoding,
        objective,
        loss_scale=16.0,
    )
    validation = qfin.validate_structured_tail_oracle(
        problem,
        plan,
        chunk_size=2,
        max_points=4,
    )
    runtime = qfin.compile(
        problem,
        backend="pennylane",
        target_error=0.1,
        arithmetic_scale=16.0,
        max_factorized_wires=20,
    ).to_pennylane(max_total_wires=20)

    indices, _, probabilities = model.chunk(0, 4)
    codes = plan.evaluate_codes(indices)
    expected = np.bincount(
        codes,
        weights=probabilities,
        minlength=2**plan.loss_qubits,
    )
    np.testing.assert_allclose(runtime.loss_probabilities(), expected, atol=1e-10)
    assert validation.disagreement_probability == pytest.approx(0.0)
    assert runtime.probability(0) == pytest.approx(validation.oracle_probability, abs=1e-10)
    assert plan.to_dict()["joint_payoff_table_materialized"] is False


def test_streaming_reference_does_not_call_materialize(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    encoding = _binary_factor_encoding(factors=3)
    objective = qfin.SparseExposureObjective(linear={"z0": 1.0, "z2": -0.5})
    problem = qfin.FactorTailProbability(
        qfin.FactorizedLossModel(encoding, objective),
        threshold=0.0,
    )

    def fail_materialize(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("joint materialization was called")

    monkeypatch.setattr(qfin.FactorizedDistributionEncoding, "materialize", fail_materialize)
    summary = qfin.evaluate_factor_tail_probability(
        problem,
        chunk_size=3,
        max_points=8,
    )
    compiled = qfin.compile(
        problem,
        backend="classical",
        arithmetic_scale=4.0,
        max_factor_validation_points=8,
        factor_validation_chunk_size=3,
    )

    assert summary.evaluated_points == 8
    assert summary.chunks == 3
    assert compiled.run().probability == pytest.approx(summary.probability)

